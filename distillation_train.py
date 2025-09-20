# /path/to/your/project/distillation_train.py
# 知識蒸留を用いたSNNモデルの分散学習スクリプト
#
# 目的:
# - ロードマップ フェーズ2「2.2. 知識蒸留の本格導入」に対応。
# - 大規模ANN（教師）の知識をSNN（生徒）に転移させ、学習を効率化・高性能化する。
#
# 実行方法:
# torchrun --nproc_per_node=<NUM_GPUS> distillation_train.py <DATA_PATH> [OPTIONS]
# 例: torchrun --nproc_per_node=2 distillation_train.py data/wikitext-103_train.jsonl --epochs 5

import os
import argparse
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM

from main import Vocabulary, set_seed
from snn_core import BreakthroughSNN
from knowledge_distillation import DistillationLoss, DistillationTrainer

def setup_distributed(rank: int, world_size: int):
    """分散学習環境をセットアップする。"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# --- データセットとCollate Function ---
class DistillationDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line)['text'] for line in f if line.strip()]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def distillation_collate_fn(batch, student_vocab, teacher_tokenizer, device):
    """教師モデルと生徒モデルで異なるトークン化を行うCollate Function。"""
    raw_texts = batch
    
    # 1. 生徒モデル用のデータ
    student_inputs, student_targets = [], []
    for text in raw_texts:
        encoded = student_vocab.encode(text)
        student_inputs.append(torch.tensor(encoded[:-1]))
        student_targets.append(torch.tensor(encoded[1:], dtype=torch.long))
    
    student_padded_inputs = pad_sequence(student_inputs, batch_first=True, padding_value=student_vocab.pad_id)
    student_padded_targets = pad_sequence(student_targets, batch_first=True, padding_value=student_vocab.pad_id)

    # 2. 教師モデル用のデータ
    teacher_tokenized = teacher_tokenizer(raw_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    return (
        student_padded_inputs,
        student_padded_targets,
        teacher_tokenized.input_ids,
        teacher_tokenized.attention_mask
    )

def main_worker(rank: int, world_size: int, args: argparse.Namespace):
    """各GPUプロセスで実行されるメインワーカー関数。"""
    print(f"Running Distillation Training on rank {rank}.")
    setup_distributed(rank, world_size)
    set_seed(args.seed)

    # --- 教師モデルとTokenizerの準備 ---
    if rank == 0:
        print(f"Loading teacher model: {args.teacher_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model)
    print(f"✅ [Rank {rank}] Teacher model loaded.")

    # --- 生徒モデル用の語彙準備 ---
    vocab_path = "vocab_distill.pth"
    if rank == 0:
        vocab = Vocabulary()
        print("📖 [Rank 0] Building student vocabulary...")
        with open(args.data_path, 'r', encoding='utf-8') as f:
            texts = (json.loads(line)['text'] for line in f)
            vocab.build_vocab(texts)
        torch.save(vocab, vocab_path)
        print(f"✅ [Rank 0] Student vocabulary built. Size: {vocab.vocab_size}")
    
    dist.barrier()
    vocab = torch.load(vocab_path, map_location='cpu')
    
    # --- データローダーの準備 ---
    dataset = DistillationDataset(args.data_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    custom_collate = lambda batch: distillation_collate_fn(batch, vocab, tokenizer, f"cuda:{rank}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=custom_collate, num_workers=2)

    # --- 生徒モデルの準備 ---
    student_config = {'d_model': args.d_model, 'd_state': args.d_state, 'num_layers': args.num_layers, 'time_steps': args.time_steps}
    student_model = BreakthroughSNN(vocab_size=vocab.vocab_size, **student_config).to(rank)
    ddp_student_model = DDP(student_model, device_ids=[rank])

    optimizer = torch.optim.AdamW(ddp_student_model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.use_scheduler else None
    criterion = DistillationLoss(student_pad_id=vocab.pad_id)
    
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=ddp_student_model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=f"cuda:{rank}",
        rank=rank
    )

    # --- 学習ループ ---
    if rank == 0:
        print("\n🔥 Knowledge Distillation Training Started...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        metrics = trainer.train_epoch(dataloader)
        if rank == 0:
            lr = scheduler.get_last_lr()[0] if scheduler else args.learning_rate
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"Epoch {epoch+1: >3}/{args.epochs}: {metrics_str}, lr: {lr:.6f}")
            if (epoch + 1) % args.log_interval == 0:
                trainer.save_checkpoint(args.model_path, epoch, vocab=vocab, config=student_config)

    if rank == 0 and os.path.exists(vocab_path):
        os.remove(vocab_path)
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNNモデルの知識蒸留")
    parser.add_argument("data_path", type=str, help="学習データ (.jsonl, simple_text形式)")
    parser.add_argument("--teacher_model", type=str, default="gpt2", help="Hugging Faceの教師モデル名")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8, help="各GPUあたりのバッチサイズ")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--model_path", type=str, default="snn_distilled_model.pth")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_state", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--time_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_scheduler", action='store_true')
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    main_worker(rank, world_size, args)
