# matsushibadenki/snn/snn_research/training/trainers.py
# SNNモデルの学習と評価ループを管理するTrainerクラス
# 
# 機能:
# - main.pyとknowledge_distillation.pyからTrainerクラスを移動・集約。
# - 分散学習に対応したチェックポイント機能などを整備。
# - mypyエラー解消のため、型ヒントを追加・修正。
# - DistillationTrainerを事前計算ロジットを使用するように簡素化。

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
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler], device: str,
                 grad_clip_norm: float, rank: int):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.grad_clip_norm = grad_clip_norm
        self.rank = rank

    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        self.model.train(is_train)
        
        input_ids, target_ids = [t.to(self.device) for t in batch]

        with torch.set_grad_enabled(is_train):
            logits, spike_data = self.model(input_ids, return_spikes=True)
            loss_dict = self.criterion(logits, target_ids, spike_data)
        
        if is_train:
            self.optimizer.zero_grad()
            loss_dict['total'].backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
        
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            if isinstance(self.criterion, CombinedLoss):
                ignore_idx = self.criterion.ce_loss_fn.ignore_index
                mask = target_ids != ignore_idx
                accuracy = (preds[mask] == target_ids[mask]).sum().float() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0)
                loss_dict['accuracy'] = accuracy

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

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
            state = {'epoch': epoch, 'model_state_dict': model_state, 'optimizer_state_dict': self.optimizer.state_dict()}
            if self.scheduler: state['scheduler_state_dict'] = self.scheduler.state_dict()
            state.update(kwargs)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state, path)
            print(f"✅ チェックポイントを '{path}' に保存しました。")

# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
class DistillationTrainer(BreakthroughTrainer):
    """知識蒸留に特化したトレーナー（事前計算ロジット使用版）。"""
    def __init__(self, *args: Any, **kwargs: Any):
        # teacher_modelの依存を削除
        if 'teacher_model' in kwargs:
            del kwargs['teacher_model']
        super().__init__(*args, **kwargs)

    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        self.model.train(is_train)
        # データローダーから事前計算済みロジットを受け取る
        student_input, student_target, teacher_logits = [t.to(self.device) for t in batch]

        with torch.set_grad_enabled(is_train):
            student_logits, spike_data = self.model(student_input, return_spikes=True)
            
            # 損失関数に渡す
            assert isinstance(self.criterion, DistillationLoss)
            loss_dict = self.criterion(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                targets=student_target,
                spikes=spike_data
            )
        
        if is_train:
            self.optimizer.zero_grad()
            loss_dict['total'].backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
        
        return {k: v.cpu().item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
