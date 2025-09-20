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

from snn_core import BreakthroughTrainer

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
        # log_target=Trueはlog_softmaxとsoftmaxの組み合わせで使うため、Falseに変更
        self.distill_loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=False)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                targets: torch.Tensor, spikes: torch.Tensor) -> Dict[str, torch.Tensor]:

        # 1. クロスエントロピー損失 (Hard Loss)
        ce_loss = self.ce_loss_fn(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))

        # 2. 蒸留損失 (Soft Loss)
        
        #◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # [改善案] 生徒と教師で語彙サイズが異なる場合への対応
        # 教師モデル(例:GPT-2)と生徒モデル(SNN)では語彙が異なるため、
        # KLダイバージェンスを計算する前に、教師の出力テンソルの語彙次元を生徒に合わせる。
        # 共通の語彙空間に射影する、あるいは単純に次元を合わせるなどの方法があるが、
        # ここではよりシンプルな次元調整を行う。
        if student_logits.size(-1) != teacher_logits.size(-1):
            # 生徒の語彙 < 教師の語彙 の場合、教師のlogitを生徒の語彙サイズまでスライス
            if student_logits.size(-1) < teacher_logits.size(-1):
                teacher_logits = teacher_logits[:, :, :student_logits.size(-1)]
            # 生徒の語彙 > 教師の語彙 の場合、教師のlogitにゼロパディングを追加
            else:
                vocab_size_diff = student_logits.size(-1) - teacher_logits.size(-1)
                padding = torch.zeros(
                    teacher_logits.size(0),
                    teacher_logits.size(1),
                    vocab_size_diff,
                    device=teacher_logits.device
                )
                teacher_logits = torch.cat([teacher_logits, padding], dim=-1)

        # [改善案] 生徒と教師でシーケンス長が異なる場合への対応
        # Tokenizerの違いにより、同じテキストでもトークン数が異なる場合がある。
        # 単純なスライスやパディングではなく、Interpolation（補間）を用いてシーケンス長を揃える。
        # これにより、情報の欠落や不自然なパディングを防ぎ、シーケンス全体の分布を維持する。
        if student_logits.shape[1] != teacher_logits.shape[1]:
            teacher_logits = F.interpolate(
                teacher_logits.transpose(1, 2), # (batch, vocab, seq_len)
                size=student_logits.shape[1],
                mode='linear',
                align_corners=False
            ).transpose(1, 2) # (batch, seq_len, vocab)
        
        # 温度付きソフトマックスで確率分布を平滑化
        soft_student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # distill_loss_fnのlog_target=Falseに合わせて、教師側はlogをかけない
        distill_loss = self.distill_loss_fn(soft_student_log_probs, soft_teacher_probs) * (self.temperature ** 2)
        #◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

        # 3. スパイク正則化損失
        spike_rate = spikes.mean()
        # 目標発火率をパラメータ化し、柔軟性を向上
        target_spike_rate = torch.tensor(0.02, device=spikes.device)
        spike_reg_loss = F.mse_loss(spike_rate, target_spike_rate)

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
        if self.rank in [-1, 0]: # メインプロセスでのみ実行
            self.teacher_model = teacher_model.to(self.device)
            self.teacher_model.eval()
        else:
            self.teacher_model = None # 他のプロセスでは教師モデルを保持しない

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
            loss_dict = self.criterion(student_logits, teacher_logits, student_target_ids, spike_data)
        
        # 3. バックプロパゲーション (学習時のみ)
        if is_train:
            self.optimizer.zero_grad()
            loss_dict['total'].backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
        
        # .item()呼び出しの前にテンソルがCPU上にあることを確認
        return {k: v.cpu().item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
