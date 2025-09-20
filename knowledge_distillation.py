# /path/to/your/project/knowledge_distillation.py
# 知識蒸留のための学習システム
#
# 目的:
# - ロードマップ フェーズ2「2.2. 知識蒸留の本格導入」に対応。
# - 大規模ANN（教師）からSNN（生徒）へ知識を転移させるためのコンポーネントを定義する。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from snn_core import BreakthroughTrainer, CombinedLoss

class DistillationLoss(nn.Module):
    """
    知識蒸留のための損失関数。
    通常のクロスエントロピー損失に加え、教師モデルのソフトターゲットとの
    KLダイバージェンス損失を計算する。
    """
    def __init__(self,
                 student_pad_id: int,
                 ce_weight: float = 0.3,
                 distill_weight: float = 0.7,
                 spike_reg_weight: float = 0.01,
                 temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature
        self.weights = {
            'ce': ce_weight,
            'distill': distill_weight,
            'spike_reg': spike_reg_weight
        }
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=student_pad_id)
        self.distill_loss_fn = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                targets: torch.Tensor, spikes: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # 1. クロスエントロピー損失 (Hard Loss) - 生徒モデルが正解ラベルを学習
        ce_loss = self.ce_loss_fn(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))

        # 2. 蒸留損失 (Soft Loss) - 生徒モデルが教師モデルの出力分布を模倣
        
        #◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # [KNOWLEDGE] 生徒と教師で語彙サイズが異なる場合への対応
        # 教師モデル(例:GPT-2)と生徒モデル(SNN)では語彙が異なるため、
        # KLダイバージェンスを計算する前に、教師の出力テンソルの語彙次元を生徒に合わせる必要がある。
        #◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        if student_logits.size(-1) != teacher_logits.size(-1):
            vocab_size_diff = student_logits.size(-1) - teacher_logits.size(-1)
            if vocab_size_diff > 0:
                # 生徒の語彙 > 教師の語彙 の場合、教師のlogitにゼロパディングを追加
                padding = torch.zeros(teacher_logits.size(0), teacher_logits.size(1), vocab_size_diff, device=teacher_logits.device)
                teacher_logits = torch.cat([teacher_logits, padding], dim=-1)
            else:
                # 生徒の語彙 < 教師の語彙 の場合、教師のlogitを生徒の語彙サイズまでスライス
                teacher_logits = teacher_logits[:, :, :student_logits.size(-1)]
        
        # 温度付きソフトマックスで確率分布を平滑化
        soft_student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        distill_loss = self.distill_loss_fn(soft_student_log_probs, soft_teacher_probs) * (self.temperature ** 2)

        # 3. スパイク正則化損失 - SNNの発火率を抑制し、エネルギー効率を向上
        spike_rate = spikes.mean()
        spike_reg_loss = F.mse_loss(spike_rate, torch.tensor(0.02, device=spikes.device))

        # 4. 各損失の重み付け加算
        total_loss = (self.weights['ce'] * ce_loss +
                      self.weights['distill'] * distill_loss +
                      self.weights['spike_reg'] * spike_reg_loss)
        
        return {
            'total': total_loss,
            'ce_loss': ce_loss,
            'distill_loss': distill_loss,
            'spike_reg_loss': spike_reg_loss
        }

class DistillationTrainer(BreakthroughTrainer):
    """
    知識蒸留に特化したトレーナー。
    学習ステップ内で教師モデルによる推論を追加する。
    """
    def __init__(self, teacher_model: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.to(self.device)
        self.teacher_model.eval()

    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        self.model.train(is_train)
        
        student_input_ids, student_target_ids, teacher_input_ids, teacher_attention_mask = [t.to(self.device) for t in batch]
        
        # 1. 教師モデルの推論 (勾配計算は行わない)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask
            )
            teacher_logits = teacher_outputs.logits

        # 2. 生徒モデル（SNN）の推論
        with torch.set_grad_enabled(is_train):
            student_logits, spike_data = self.model(student_input_ids, return_spikes=True)
            
            #◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            # [KNOWLEDGE] 生徒と教師でシーケンス長が異なる場合への対応
            # Tokenizerの違いにより、同じテキストでもトークン数が異なる場合がある。
            # ここでは、短い方のシーケンス長に合わせる形で教師の出力テンソルの形状を調整する。
            #◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            seq_len_diff = student_logits.shape[1] - teacher_logits.shape[1]
            if seq_len_diff > 0:
                # 生徒 > 教師 の場合、教師のシーケンスにパディングを追加
                padding = torch.zeros(teacher_logits.size(0), seq_len_diff, teacher_logits.size(2), device=self.device)
                teacher_logits = torch.cat([teacher_logits, padding], dim=1)
            elif seq_len_diff < 0:
                # 生徒 < 教師 の場合、教師のシーケンスを生徒の長さにスライス
                teacher_logits = teacher_logits[:, :student_logits.shape[1], :]

            loss_dict = self.criterion(student_logits, teacher_logits, student_target_ids, spike_data)
        
        # 3. バックプロパゲーション (学習時のみ)
        if is_train:
            self.optimizer.zero_grad()
            loss_dict['total'].backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
