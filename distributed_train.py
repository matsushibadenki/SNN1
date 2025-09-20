# /path/to/your/project/distributed_train.py
# SNNモデルの分散学習を実行するためのスクリプト
#
# 目的:
# - ロードマップ フェーズ2「2.1. 大規模分散学習環境の構築」に対応。
# - 複数GPUを利用してモデルの大規模化と学習の高速化を実現する。
#
# 実行方法:
# torchrun --nproc_per_node=<NUM_GPUS> distributed_train.py <DATA_PATH> [OPTIONS]
# 例: torchrun --nproc_per_node=2 distributed_train.py data/wikitext-103_train.jsonl --epochs 5 --batch_size 16

import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from main import (
    Vocabulary,
    DataFormat,
    get_text_extractor,
    create_dataset,
    collate_fn,
    set_seed
)
from snn_core import BreakthroughSNN, CombinedLoss, BreakthroughTrainer

def setup_distributed(rank: int, world_size: int):
    """分散学習環境をセットアップする。"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # NCCLバックエンドを初期化
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """分散学習環境をクリーンアップする。"""
    dist.destroy_process_group()

def main_worker(rank: int, world_size: int, args: argparse.Namespace):
    """各GPUプロセスで実行されるメインワーカー関数。"""
    print(f"Running DDP on rank {rank}.")
    setup_distributed(rank, world_size)
    set_seed(args.seed)

    # --- データセットと語彙の準備 ---
    # rank 0 のプロセスのみが語彙を構築し、他プロセスはそれをロードする
    vocab_path = "vocab.pth"
    if rank == 0:
        vocab = Vocabulary()
        print("📖 [Rank 0] 語彙を構築中...")
        text_extractor = get_text_extractor(args.data_format)
        vocab.build_vocab(text_extractor(args.data_path))
        torch.save(vocab, vocab_path)
        print(f"✅ [Rank 0] 語彙を構築し、保存しました。語彙数: {vocab.vocab_size}")
    
    dist.barrier()  # rank 0 が語彙を保存し終わるまで全プロセスが待機
    
    vocab = torch.load(vocab_path, map_location='cpu')
    if rank != 0:
        print(f"✅ [Rank {rank}] 語彙をロードしました。")

    dataset = create_dataset(args.data_format, args.data_path, vocab)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    custom_collate_fn = lambda batch: collate_fn(batch, vocab.pad_id)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,
        collate_fn=custom_collate_fn,
        num_workers=2,
        pin_memory=True
    )

    # --- モデル、損失、オプティマイザの準備 ---
    config = {'d_model': args.d_model, 'd_state': args.d_state, 'num_layers': args.num_layers, 'time_steps': args.time_steps}
    model = BreakthroughSNN(vocab_size=vocab.vocab_size, **config).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.use_scheduler else None
    criterion = CombinedLoss(pad_id=vocab.pad_id)
    
    # --- トレーナーの準備 ---
    trainer = BreakthroughTrainer(
        ddp_model, optimizer, criterion, scheduler, device=f"cuda:{rank}", grad_clip_norm=1.0, rank=rank
    )

    # --- 学習ループ ---
    if rank == 0:
        print("\n🔥 分散学習を開始します...")
        
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        train_metrics = trainer.train_epoch(dataloader)
        
        if rank == 0:
            lr = scheduler.get_last_lr()[0] if scheduler else args.learning_rate
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
            print(f"Epoch {epoch+1: >3}/{args.epochs}: {metrics_str}, lr: {lr:.6f}")

            if (epoch + 1) % args.log_interval == 0:
                trainer.save_checkpoint(
                    args.model_path, 
                    epoch, 
                    vocab=vocab, 
                    config=config
                )

    if rank == 0 and os.path.exists(vocab_path):
        os.remove(vocab_path)
        
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNNモデルの分散学習")
    # main.pyから引数をコピーし、大規模化用にデフォルト値を変更
    parser.add_argument("data_path", type=str, help="学習データのファイルパス (.jsonl)")
    parser.add_argument("--data_format", type=DataFormat, default=DataFormat.SIMPLE_TEXT, choices=list(DataFormat), help="学習データの形式")
    parser.add_argument("--epochs", type=int, default=10, help="学習エポック数")
    parser.add_argument("--batch_size", type=int, default=16, help="各GPUあたりのバッチサイズ")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="学習率")
    parser.add_argument("--log_interval", type=int, default=1, help="ログ・モデル保存のエポック間隔")
    parser.add_argument("--model_path", type=str, default="breakthrough_snn_distributed.pth", help="学習済みモデルの保存パス")
    parser.add_argument("--d_model", type=int, default=256, help="モデルの次元数 (大規模化)")
    parser.add_argument("--d_state", type=int, default=128, help="状態空間モデルの状態次元数")
    parser.add_argument("--num_layers", type=int, default=8, help="レイヤー数")
    parser.add_argument("--time_steps", type=int, default=20, help="シミュレーションのタイムステップ数")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--use_scheduler", action='store_true', help="学習率スケジューラを有効にする")
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        main_worker(rank, world_size, args)
    else:
        print("単一GPUモードで実行します。分散学習には `torchrun` を使用してください。")
        # 簡易的にrank 0として実行
        main_worker(0, 1, args)
