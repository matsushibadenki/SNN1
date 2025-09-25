# matsushibadenki/snn/snn_research/training/trainers.py
# SNNモデルの学習と評価ループを管理するTrainerクラス (AMP・チェックポイント復元対応)
# 
# 機能:
# - 自動混合精度(AMP)学習に対応。`torch.cuda.amp.GradScaler` を使用して学習を高速化。
# - チェックポイントからの学習再開機能 (`load_checkpoint` メソッド) を追加。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import collections
from tqdm import tqdm  # type: ignore
from typing import Tuple, Dict, Any, Optional

from snn_research.training.losses import CombinedLoss, DistillationLoss

class BreakthroughTrainer:
    """汎用性と拡張性を高めたSNNの統合トレーニングシステム。"""
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler], device: str,
                 grad_clip_norm: float, rank: int, use_amp: bool):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.grad_clip_norm = grad_clip_norm
        self.rank = rank
        self.use_amp = use_amp and torch.cuda.is_available()
        # AMPが有効な場合、GradScalerを初期化
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        self.model.train(is_train)
        
        input_ids, target_ids = [t.to(self.device) for t in batch[:2]]

        # AMPを有効にした状態でフォワードパスを実行
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                logits, spike_data = self.model(input_ids, return_spikes=True)
                loss_dict = self.criterion(logits, target_ids, spike_data)
        
        if is_train:
            self.optimizer.zero_grad()
            # GradScalerを使って損失をスケールし、バックワードパスを実行
            self.scaler.scale(loss_dict['total']).backward()
            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer) # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            # GradScalerを使ってオプティマイザのステップを実行
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            if isinstance(self.criterion, CombinedLoss):
                ignore_idx = self.criterion.ce_loss_fn.ignore_index
                mask = target_ids != ignore_idx
                accuracy = (preds[mask] == target_ids[mask]).sum().float() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0)
                loss_dict['accuracy'] = accuracy

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        total_metrics: Dict[str, float] = collections.defaultdict(float)
        num_batches = len(dataloader)
        progress_bar = tqdm(dataloader, desc="Training", disable=(self.rank not in [-1, 0]))
        
        for batch in progress_bar:
            metrics = self._run_step(batch, is_train=True)
            for key, value in metrics.items(): total_metrics[key] += value

        if self.scheduler: self.scheduler.step()
        return {key: value / num_batches for key, value in total_metrics.items()}

    def save_checkpoint(self, path: str, epoch: int, **kwargs: Any):
        if self.rank in [-1, 0]:
            model_state = self.model.module.state_dict() if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model.state_dict()
            state = {
                'epoch': epoch, 
                'model_state_dict': model_state, 
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler_state_dict': self.scaler.state_dict()
            }
            if self.scheduler: state['scheduler_state_dict'] = self.scheduler.state_dict()
            state.update(kwargs)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state, path)
            print(f"✅ チェックポイントを '{path}' に保存しました (Epoch: {epoch})。")

# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def load_checkpoint(self, path: str) -> int:
        """チェックポイントをロードし、学習を再開するためのエポック番号を返す。"""
        if not os.path.exists(path):
            print(f"⚠️ チェックポイントファイルが見つかりません: {path}。最初から学習を開始します。")
            return 0
            
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank} if self.rank != -1 else self.device
        checkpoint = torch.load(path, map_location=map_location)
        
        model_to_load = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint.get('epoch', -1) + 1
        print(f"✅ チェックポイント '{path}' を正常にロードしました。Epoch {start_epoch} から学習を再開します。")
        return start_epoch
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️


class DistillationTrainer(BreakthroughTrainer):
    """知識蒸留に特化したトレーナー（事前計算ロジット使用版）。"""
    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        self.model.train(is_train)
        student_input, student_target, teacher_logits = [t.to(self.device) for t in batch]

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                student_logits, spike_data = self.model(student_input, return_spikes=True)
                assert isinstance(self.criterion, DistillationLoss)
                loss_dict = self.criterion(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    targets=student_target,
                    spikes=spike_data
                )
        
        if is_train:
            self.optimizer.zero_grad()
            self.scaler.scale(loss_dict['total']).backward()
            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        return {k: v.cpu().item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
