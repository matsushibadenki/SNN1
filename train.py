# matsushibadenki/snn/train.py
# DIコンテナを利用した、統合学習実行スクリプト (Tokenizer移行版)
#
# 変更点:
# - 独自Vocabularyの構築・保存ロジックを完全に削除。
# - DIコンテナからHugging Face Tokenizerを取得して使用するように変更。
# - データセットの初期化、collate_fnをTokenizerベースの処理に更新。
# - 知識蒸留時のcollate_fnを簡素化。
# - チェックポイントにtokenizerの名前を保存するように変更。

import os
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
from transformers import PreTrainedTokenizerBase

from app.containers import TrainingContainer
from snn_research.data.datasets import DataFormat, get_dataset_class, SNNBaseDataset

torch.autograd.set_detect_anomaly(True)

def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collate_fn(batch, pad_id):
    inputs, targets = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=pad_id)
    return padded_inputs, padded_targets

def distillation_collate_fn(batch, tokenizer: PreTrainedTokenizerBase):
    # 生のテキストバッチをTokenizerでまとめて処理
    texts = [item['text'] for item in batch]
    
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    
    # student_targetはinput_idsを1つずらしたもの
    student_target = input_ids.clone()
    # パディング部分は損失計算から除外
    student_target[~attention_mask.bool()] = tokenizer.pad_token_id
    
    return input_ids, student_target, input_ids, attention_mask


def main_worker(rank, world_size, container, args):
    is_distributed = container.config.training.type() != "standard"
    is_distillation = container.config.training.type() == "distillation"
    
    if is_distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)

    # DIコンテナからTokenizerを取得
    tokenizer = container.tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset: SNNBaseDataset | RawTextDataset
    if is_distillation:
        # 蒸留時は、getitemがテキスト辞書を返すような特殊なDatasetが必要
        class RawTextDataset(Dataset):
            def __init__(self, file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.data = [line for line in f if line.strip()]
            def __len__(self): return len(self.data)
            def __getitem__(self, idx): return {'text': self.data[idx]}
        dataset = RawTextDataset(container.config.data.path())
    else:
        data_format = DataFormat(container.config.data.format())
        dataset_class = get_dataset_class(data_format)
        dataset = dataset_class(
            file_path=container.config.data.path(),
            tokenizer=tokenizer,
            max_seq_len=container.config.model.time_steps()
        )
    
    val_split = int(len(dataset) * container.config.data.split_ratio())
    train_dataset, _ = random_split(dataset, [len(dataset) - val_split, val_split])
    
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    
    if is_distillation:
        _collate_fn = partial(distillation_collate_fn, tokenizer=tokenizer)
    else:
        _collate_fn = partial(collate_fn, pad_id=tokenizer.pad_token_id)

    dataloader = DataLoader(train_dataset, batch_size=container.config.training.batch_size(),
                              sampler=sampler, collate_fn=_collate_fn, num_workers=2, shuffle=(sampler is None))

    # --- デバイスの自動選択ロジック ---
    if is_distributed and torch.cuda.is_available():
        device = f"cuda:{rank}"
    else:
        device = container.config.device()
        if device == "cuda" and not torch.cuda.is_available(): device = "cpu"
        if device == "mps" and not torch.backends.mps.is_available(): device = "cpu"
    print(f"Process {rank}: Selected device: {device}")
    
    model = container.snn_model().to(device)
    
    model_config = {
        'd_model': container.config.model.d_model(),
        'd_state': container.config.model.d_state(),
        'num_layers': container.config.model.num_layers(),
        'time_steps': container.config.model.time_steps(),
        'n_head': container.config.model.n_head(),
    }

    if is_distributed:
        model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    optimizer = container.optimizer(params=model.parameters())
    scheduler = container.scheduler(optimizer=optimizer) if container.config.training.use_scheduler() else None

    trainer_args = {
        "model": model, "optimizer": optimizer, "scheduler": scheduler,
        "device": device, "rank": rank,
    }
    if is_distillation:
        trainer = container.distillation_trainer(**trainer_args)
    else:
        trainer = container.standard_trainer(**trainer_args)

    if rank in [-1, 0]: print(f"\n🔥 {container.config.training.type()} 学習を開始します...")
    for epoch in range(container.config.training.epochs()):
        if is_distributed and sampler: sampler.set_epoch(epoch)
        metrics = trainer.train_epoch(dataloader)
        if rank in [-1, 0]:
            lr = scheduler.get_last_lr()[0] if scheduler else container.config.training.learning_rate()
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"Epoch {epoch+1: >3}/{container.config.training.epochs()}: {metrics_str}, lr: {lr:.6f}")
            if (epoch + 1) % container.config.training.log_interval() == 0:
                trainer.save_checkpoint(
                    container.config.model.path(), 
                    epoch, 
                    tokenizer_name=tokenizer.name_or_path, 
                    config=model_config
                )

    if is_distributed: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNNモデルの統合学習スクリプト")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="設定ファイルのパス")
    parser.add_argument("--data_path", type=str, help="データセットのパス (設定ファイルを上書き)")
    parser.add_argument("--data_format", type=str, choices=[f.value for f in DataFormat], help="データ形式 (設定ファイルを上書き)")
    args = parser.parse_args()

    container = TrainingContainer()
    container.config.from_yaml(args.config)
    if args.data_path: container.config.data.path.from_value(args.data_path)
    if args.data_format: container.config.data.format.from_value(args.data_format)
    
    set_seed(container.config.seed())

    training_type = container.config.training.type()
    if training_type in ["distributed", "distillation"] and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        world_size = torch.cuda.device_count()
        print(f"{world_size}個のGPUで '{training_type}' 学習を開始します。")
        torch.multiprocessing.spawn(main_worker, args=(world_size, container, args), nprocs=world_size, join=True)
    else:
        print(f"単一デバイスで '{training_type}' 学習を開始します。")
        main_worker(-1, 1, container, args)
