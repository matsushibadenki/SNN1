# matsushibadenki/snn/snn_research/training/trainers.py
# SNNモデルの学習と評価ループを管理するTrainerクラス
# 
# 機能:
# - main.pyとknowledge_distillation.pyからTrainerクラスを移動・集約。
# - 分散学習に対応したチェックポイント機能などを整備。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import collections
from tqdm import tqdm
from typing import Tuple, Dict, Any, Optional

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
            mask = target_ids != self.criterion.ce_loss_fn.ignore_index
            accuracy = (preds[mask] == target_ids[mask]).sum() / mask.sum() if mask.sum() > 0 else 0.0
            loss_dict['accuracy'] = accuracy

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        total_metrics = collections.defaultdict(float)
        num_batches = len(dataloader)
        progress_bar = tqdm(dataloader, desc="Training", disable=(self.rank not in [-1, 0]))
        
        for batch in progress_bar:
            metrics = self._run_step(batch, is_train=True)
            for key, value in metrics.items(): total_metrics[key] += value

        if self.scheduler: self.scheduler.step()
        return {key: value / num_batches for key, value in total_metrics.items()}

    def save_checkpoint(self, path: str, epoch: int, **kwargs):
        if self.rank in [-1, 0]:
            model_state = self.model.module.state_dict() if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model.state_dict()
            state = {'epoch': epoch, 'model_state_dict': model_state, 'optimizer_state_dict': self.optimizer.state_dict()}
            if self.scheduler: state['scheduler_state_dict'] = self.scheduler.state_dict()
            state.update(kwargs)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state, path)
            print(f"✅ チェックポイントを '{path}' に保存しました。")


class DistillationTrainer(BreakthroughTrainer):
    """知識蒸留に特化したトレーナー。"""
    def __init__(self, teacher_model: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.to(self.device) if self.rank in [-1, 0] else None
        if self.teacher_model: self.teacher_model.eval()

    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        self.model.train(is_train)
        student_input, student_target, teacher_input, teacher_mask = [t.to(self.device) for t in batch]
        
        with torch.no_grad():
            teacher_logits = self.teacher_model(input_ids=teacher_input, attention_mask=teacher_mask).logits

        with torch.set_grad_enabled(is_train):
            student_logits, spike_data = self.model(student_input, return_spikes=True)
            loss_dict = self.criterion(student_logits, teacher_logits, student_target, spike_data)
        
        if is_train:
            self.optimizer.zero_grad()
            loss_dict['total'].backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
        
        return {k: v.cpu().item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}