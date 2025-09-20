# matsushibadenki/snn/train.py
# DIコンテナを利用した、統合学習実行スクリプト (修正版)
#
# 変更点:
# - 知識蒸留時に使用する専用のDatasetクラスを追加し、collate_fnのバグを修正。

import os
import argparse
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP

from app.containers import TrainingContainer
from snn_research.data.datasets import DataFormat, Vocabulary, get_dataset_class

# --- (set_seed, collate_fn のコードは変更なし) ---
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

# --- 知識蒸留専用のデータセットとCollate Function ---
class DistillationDataset(Dataset):
    """知識蒸留用のシンプルなテキストデータセット"""
    def __init__(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line)['text'] for line in f if line.strip()]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def distillation_collate_fn(batch_texts, student_vocab, teacher_tokenizer):
    student_inputs, student_targets = [], []
    for text in batch_texts:
        encoded = student_vocab.encode(text)
        student_inputs.append(torch.tensor(encoded[:-1]))
        student_targets.append(torch.tensor(encoded[1:], dtype=torch.long))
    
    student_padded_inputs = pad_sequence(student_inputs, batch_first=True, padding_value=student_vocab.pad_id)
    student_padded_targets = pad_sequence(student_targets, batch_first=True, padding_value=student_vocab.pad_id)

    teacher_tokenized = teacher_tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    return student_padded_inputs, student_padded_targets, teacher_tokenized.input_ids, teacher_tokenized.attention_mask

def main_worker(rank, world_size, container, args):
    is_distributed = container.config.training.type() != "standard"
    is_distillation = container.config.training.type() == "distillation"
    
    if is_distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    vocab_path = "vocab.pth"
    if rank in [-1, 0]:
        print("📖 語彙を構築中...")
        vocab = container.vocabulary()
        dataset_class = DistillationDataset if is_distillation else get_dataset_class(DataFormat(container.config.data.format()))
        text_iterator = (item for item in dataset_class(container.config.data.path())) if is_distillation else dataset_class.extract_texts(container.config.data.path())
        vocab.build_vocab(text_iterator)
        torch.save(vocab, vocab_path)
        print(f"✅ 語彙を構築しました。語彙数: {vocab.vocab_size}")

    if is_distributed: dist.barrier()
    vocab = torch.load(vocab_path, map_location='cpu')

    dataset = (DistillationDataset(container.config.data.path()) if is_distillation 
               else get_dataset_class(DataFormat(container.config.data.format()))(container.config.data.path(), vocab))

    val_split = int(len(dataset) * container.config.data.split_ratio())
    train_dataset, _ = random_split(dataset, [len(dataset) - val_split, val_split])
    
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    
    _collate_fn = lambda b: collate_fn(b, vocab.pad_id)
    if is_distillation:
        teacher_tokenizer = container.teacher_tokenizer()
        _collate_fn = lambda b: distillation_collate_fn(b, vocab, teacher_tokenizer)

    dataloader = DataLoader(train_dataset, batch_size=container.config.training.batch_size(),
                              sampler=sampler, collate_fn=_collate_fn, num_workers=2, shuffle=(sampler is None))

    device = f"cuda:{rank}" if is_distributed else container.config.device()
    model_config = container.config.model.to_dict()
    # configからpathキーを削除
    model_config.pop('path', None)
    model = container.snn_model(vocab_size=vocab.vocab_size, **model_config).to(device)

    if is_distributed: model = DDP(model, device_ids=[rank])
    
    optimizer = container.optimizer(params=model.parameters())
    scheduler = container.scheduler(optimizer=optimizer) if container.config.training.use_scheduler() else None

    # pad_idを損失関数に設定
    container.standard_loss.kwargs['pad_id'] = vocab.pad_id
    container.distillation_loss.kwargs['student_pad_id'] = vocab.pad_id
    
    trainer = container.trainer_factory(model=model, optimizer=optimizer, scheduler=scheduler, device=device, rank=rank)

    if rank in [-1, 0]: print(f"\n🔥 {container.config.training.type()} 学習を開始します...")
    for epoch in range(container.config.training.epochs()):
        if is_distributed: sampler.set_epoch(epoch)
        metrics = trainer.train_epoch(dataloader)
        if rank in [-1, 0]:
            lr = scheduler.get_last_lr()[0] if scheduler else container.config.training.learning_rate()
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"Epoch {epoch+1: >3}/{container.config.training.epochs()}: {metrics_str}, lr: {lr:.6f}")
            if (epoch + 1) % container.config.training.log_interval() == 0:
                trainer.save_checkpoint(
                    container.config.model.path(), 
                    epoch, 
                    vocab=vocab, 
                    config=model_config
                )

    if rank in [-1, 0] and os.path.exists(vocab_path): os.remove(vocab_path)
    if is_distributed: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNNモデルの統合学習スクリプト")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="設定ファイルのパス")
    parser.add_argument("--data_path", type=str, help="データセットのパス (設定ファイルを上書き)")
    args = parser.parse_args()

    container = TrainingContainer()
    container.config.from_yaml(args.config)
    if args.data_path: container.config.data.path.from_value(args.data_path)
    
    set_seed(container.config.seed())

    training_type = container.config.training.type()
    if training_type in ["distributed", "distillation"]:
        world_size = torch.cuda.device_count()
        print(f"{world_size}個のGPUで '{training_type}' 学習を開始します。")
        torch.multiprocessing.spawn(main_worker, args=(world_size, container, args), nprocs=world_size, join=True)
    else:
        print(f"単一デバイス ({container.config.device()}) で 'standard' 学習を開始します。")
        main_worker(-1, 1, container, args)