# matsushibadenki/snn/snn_research/training/losses.py
# SNN学習で使用する損失関数
# 
# 機能:
# - snn_coreとknowledge_distillationから損失関数クラスを移動・集約。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class CombinedLoss(nn.Module):
    """クロスエントロピー損失とスパイク発火率の正則化を組み合わせた損失関数。"""
    def __init__(self, ce_weight: float, spike_reg_weight: float, pad_id: int, target_spike_rate: float = 0.02):
        super().__init__()
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
        self.weights = {'ce': ce_weight, 'spike_reg': spike_reg_weight}
        self.target_spike_rate = target_spike_rate

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, spikes: torch.Tensor) -> dict:
        ce_loss = self.ce_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        spike_rate = spikes.mean()
        spike_reg_loss = F.mse_loss(spike_rate, torch.tensor(self.target_spike_rate, device=spike_rate.device))
        total_loss = self.weights['ce'] * ce_loss + self.weights['spike_reg'] * spike_reg_loss
        
        return {
            'total': total_loss, 'ce_loss': ce_loss,
            'spike_reg_loss': spike_reg_loss, 'spike_rate': spike_rate
        }

class DistillationLoss(nn.Module):
    """知識蒸留のための損失関数。"""
    def __init__(self, student_pad_id: int, ce_weight: float, distill_weight: float,
                 spike_reg_weight: float, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.weights = {'ce': ce_weight, 'distill': distill_weight, 'spike_reg': spike_reg_weight}
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=student_pad_id)
        self.distill_loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=False)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                targets: torch.Tensor, spikes: torch.Tensor) -> Dict[str, torch.Tensor]:

        ce_loss = self.ce_loss_fn(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))

        if student_logits.size(-1) != teacher_logits.size(-1):
            if student_logits.size(-1) < teacher_logits.size(-1):
                teacher_logits = teacher_logits[:, :, :student_logits.size(-1)]
            else:
                vocab_diff = student_logits.size(-1) - teacher_logits.size(-1)
                padding = torch.zeros(teacher_logits.size(0), teacher_logits.size(1), vocab_diff, device=teacher_logits.device)
                teacher_logits = torch.cat([teacher_logits, padding], dim=-1)

        if student_logits.shape[1] != teacher_logits.shape[1]:
            teacher_logits = F.interpolate(teacher_logits.transpose(1, 2), size=student_logits.shape[1],
                                           mode='linear', align_corners=False).transpose(1, 2)
        
        soft_student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        distill_loss = self.distill_loss_fn(soft_student_log_probs, soft_teacher_probs) * (self.temperature ** 2)

        spike_rate = spikes.mean()
        target_spike_rate = torch.tensor(0.02, device=spikes.device)
        spike_reg_loss = F.mse_loss(spike_rate, target_spike_rate)

        total_loss = (self.weights['ce'] * ce_loss +
                      self.weights['distill'] * distill_loss +
                      self.weights['spike_reg'] * spike_reg_loss)

        return {
            'total': total_loss, 'ce_loss': ce_loss,
            'distill_loss': distill_loss, 'spike_reg_loss': spike_reg_loss
        }