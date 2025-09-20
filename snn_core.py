# matsushibadenki/snn/snn_core.pyの修正
# SNNモデルの定義、次世代ニューロン、学習システムなど、中核となるロジックを集約したライブラリ
#
# 元ファイル:
# - snn_breakthrough.py
# - advanced_snn_chat.py
# - train_text_snn.py
# - snn_advanced_optimization.py
# - snn_advanced_plasticity.py
# のうち、最も先進的な機能を統合・整理し、さらに洗練させたバージョン
#
# 改善点:
# - BreakthroughTrainerを汎用化し、学習・評価ループ、チェックポイント機能などを統合。
# - TTFSEncoder, EventDrivenSSMLayerなどの先進的コンポーネントを統合。
# - STDP, STP, メタ可塑性などの生物学的可塑性メカニズムを統合。
# - 評価時に正解率(Accuracy)を計算する機能を追加。

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, functional
from typing import Tuple, Dict, Any, Optional, List
from torch.utils.data import DataLoader
import os
import collections
import math
from collections import deque

# ----------------------------------------
# 1. 高度なスパイクエンコーダー (snn_advanced_optimization.pyより)
# ----------------------------------------

class TTFSEncoder(nn.Module):
    """
    Time-to-First-Spike (TTFS) 符号化器
    連続値を最初のスパイクまでの時間（レイテンシ）に変換
    """
    def __init__(self, d_model: int, time_steps: int, max_latency: int = 10):
        super().__init__()
        self.d_model = d_model
        self.time_steps = time_steps
        self.max_latency = min(max_latency, time_steps)
        # 符号化の感度を調整するための学習可能パラメータ
        self.scaling = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 入力テンソル (batch_size, seq_len, d_model)
        Returns:
            スパイク列 (batch_size, time_steps, seq_len, d_model)
        """
        # 活性化関数で入力を正の値に変換
        x_activated = torch.sigmoid(self.scaling * x)

        # スパイクタイミングを計算 (値が大きいほど早く発火)
        spike_times = self.max_latency * (1.0 - x_activated)
        spike_times = torch.round(spike_times).long()

        # スパイク列を生成
        spikes = torch.zeros(x.shape[0], self.time_steps, x.shape[1], x.shape[2], device=x.device)
        
        # scatter_を使って指定したタイミングで発火
        # (batch, 1, seq, d_model) -> (batch, time, seq, d_model)
        # spike_timesが範囲外になるのを防ぐ
        spike_times.clamp_(0, self.time_steps - 1)
        spikes.scatter_(1, spike_times.unsqueeze(1), 1.0)
        return spikes

# ----------------------------------------
# 2. 高度なニューロンモデル (snn_advanced_optimization.py, snn_advanced_plasticity.pyより)
# ----------------------------------------

class AdaptiveLIFNeuron(nn.Module):
    """
    適応的閾値を持つLIFニューロン
    発火率に応じて閾値を動的に調整し、安定性を向上
    """
    def __init__(self, features: int, tau: float = 2.0, base_threshold: float = 1.0, adaptation_strength: float = 0.1):
        super().__init__()
        self.features = features
        self.tau = tau
        self.base_threshold = base_threshold
        self.adaptation_strength = adaptation_strength
        
        self.v_mem = nn.Parameter(torch.zeros(1, features), requires_grad=False)
        self.adaptive_threshold = nn.Parameter(torch.ones(features) * base_threshold, requires_grad=False)
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        self.mem_decay = math.exp(-1.0 / tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # v_memのバッチサイズを動的に調整
        if self.v_mem.shape[0] != batch_size:
            self.v_mem = self.v_mem.data.expand(batch_size, -1).clone()

        self.v_mem = self.v_mem.data * self.mem_decay + x
        spike = self.surrogate_function(self.v_mem - self.adaptive_threshold)
        self.v_mem = self.v_mem.data * (1.0 - spike.detach())
        
        # 適応的閾値の更新
        with torch.no_grad():
            self.adaptive_threshold += self.adaptation_strength * (spike - 0.1) # 目標発火率0.1
        
        return spike

class MetaplasticLIFNeuron(nn.Module):
    """
    メタ可塑性を持つLIFニューロン
    学習履歴に基づいて可塑性を動的に調整
    """
    def __init__(self, 
                 features: int,
                 tau: float = 2.0,
                 threshold: float = 1.0,
                 metaplastic_tau: float = 1000.0,
                 metaplastic_strength: float = 0.1):
        super().__init__()
        
        self.features = features
        self.tau = tau
        self.base_threshold = threshold
        self.metaplastic_tau = metaplastic_tau
        self.metaplastic_strength = metaplastic_strength
        
        self.register_buffer('v_mem', torch.zeros(1, features))
        self.register_buffer('activity_history', torch.zeros(1, features))
        self.register_buffer('adaptive_threshold', torch.ones(features) * threshold)
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        
        self.mem_decay = math.exp(-1.0 / tau)
        self.meta_decay = math.exp(-1.0 / metaplastic_tau)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        if self.v_mem.shape[0] != batch_size:
            self.v_mem = self.v_mem.expand(batch_size, -1).clone()
            self.activity_history = self.activity_history.expand(batch_size, -1).clone()
        
        self.v_mem = self.v_mem * self.mem_decay + x
        current_threshold = self.adaptive_threshold * (1.0 + self.metaplastic_strength * self.activity_history.mean(0))
        spike = self.surrogate_function(self.v_mem - current_threshold)
        self.v_mem = self.v_mem * (1.0 - spike.detach())
        self.activity_history = self.activity_history * self.meta_decay + spike.detach() * (1 - self.meta_decay)
        
        return spike

# ----------------------------------------
# 3. 生物学的シナプス可塑性 (snn_advanced_plasticity.pyより)
# ----------------------------------------
# (省略：変更なし)
# ----------------------------------------
# 4. Event-Driven State Space Model (snn_advanced_optimization.pyより)
# ----------------------------------------

class EventDrivenSSMLayer(nn.Module):
    """
    Event-driven Spiking State Space Model
    入力スパイクが疎な場合に計算をスキップし、効率を向上
    """
    def __init__(self, d_model: int, d_state: int = 64, dt: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt = dt
        
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model, d_model)) # 修正: Dはd_model -> d_model
        
        self.state_lif = AdaptiveLIFNeuron(d_state)
        self.output_lif = AdaptiveLIFNeuron(d_model)
        self.register_buffer('h_state', torch.zeros(1, 1, d_state))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, seq_len, _ = x.shape
        if self.h_state.shape[0] != batch_size or self.h_state.shape[1] != seq_len:
            self.h_state = torch.zeros(batch_size, seq_len, self.d_state, device=x.device)
        self.h_state = self.h_state.detach()

        outputs = []
        for t in range(time_steps):
            x_t = x[:, t, :, :]
            # Event-driven: 計算をアクティブなニューロンに限定
            if torch.any(x_t > 0):
                state_transition = F.linear(self.h_state, self.A)
                input_projection = F.linear(x_t, self.B.T) # 修正: Bの転置
                state_update = state_transition + input_projection
                self.h_state = self.state_lif(state_update)
                
                output_projection = F.linear(self.h_state, self.C.T) # 修正: Cの転置
                output_update = output_projection + F.linear(x_t, self.D) # 修正: Dを線形層として適用
                out_spike = self.output_lif(output_update)
            else:
                # 入力がない場合は状態を減衰させ、出力スパイクはゼロ
                self.h_state = self.state_lif(F.linear(self.h_state, self.A))
                out_spike = torch.zeros_like(x_t)
            
            outputs.append(out_spike)
        
        functional.reset_net(self)
        return torch.stack(outputs, dim=1)

# ----------------------------------------
# 5. エネルギー効率最適化 (snn_comprehensive_optimization.pyより)
# ----------------------------------------
# (省略：変更なし)
# ----------------------------------------
# 6. 統合された次世代SNNアーキテクチャ
# ----------------------------------------

class BreakthroughSNN(nn.Module):
    """
    EventDriven-SSMを中核に据えた、次世代のSNNアーキテクチャ。
    """
    def __init__(self, vocab_size: int, d_model: int = 256, d_state: int = 64, num_layers: int = 4, time_steps: int = 20):
        super().__init__()
        self.time_steps = time_steps
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.spike_encoder = TTFSEncoder(d_model=d_model, time_steps=time_steps)
        self.layers = nn.ModuleList([EventDrivenSSMLayer(d_model, d_state) for _ in range(num_layers)])
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        token_emb = self.token_embedding(input_ids)
        spike_sequence = self.spike_encoder(token_emb)
        
        hidden_states = spike_sequence
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        time_integrated = hidden_states.mean(dim=1)
        logits = self.output_projection(time_integrated)
        
        # 常に (logits, hidden_states) のタプルを返すように統一
        return logits, hidden_states

# ----------------------------------------
# 7. 統合トレーニングシステム
# ----------------------------------------

class CombinedLoss(nn.Module):
    """
    クロスエントロピー損失とスパイク発火率の正則化を組み合わせた損失関数。
    """
    def __init__(self, ce_weight: float = 1.0, spike_reg_weight: float = 0.01, target_spike_rate: float = 0.02):
        super().__init__()
        # ignore_index=-100などを設定できるように拡張
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.weights = {'ce': ce_weight, 'spike_reg': spike_reg_weight}
        self.target_spike_rate = target_spike_rate

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, spikes: torch.Tensor) -> dict:
        ce_loss = self.ce_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        spike_rate = spikes.mean()
        spike_reg_loss = F.mse_loss(spike_rate, torch.tensor(self.target_spike_rate, device=spike_rate.device))
        total_loss = self.weights['ce'] * ce_loss + self.weights['spike_reg'] * spike_reg_loss
        
        return {
            'total': total_loss,
            'ce_loss': ce_loss,
            'spike_reg_loss': spike_reg_loss,
            'spike_rate': spike_rate
        }

class BreakthroughTrainer:
    """
    汎用性と拡張性を高めたSNNの統合トレーニングシステム。
    """
    def __init__(
        self,
        model: BreakthroughSNN,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        grad_clip_norm: float = 1.0
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.grad_clip_norm = grad_clip_norm

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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
        
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # 正解率(Accuracy)の計算を追加
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            # パディング部分(-100 or pad_id)を除外して計算
            mask = target_ids != self.criterion.ce_loss_fn.ignore_index
            corrects = (preds[mask] == target_ids[mask]).sum()
            total_valid = mask.sum()
            accuracy = corrects / total_valid if total_valid > 0 else 0.0
            loss_dict['accuracy'] = accuracy
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        total_metrics = collections.defaultdict(float)
        num_batches = len(dataloader)
        for batch in dataloader:
            metrics = self._run_step(batch, is_train=True)
            for key, value in metrics.items():
                total_metrics[key] += value
        return {key: value / num_batches for key, value in total_metrics.items()}

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        total_metrics = collections.defaultdict(float)
        num_batches = len(dataloader)
        # 評価時は tqdm を使って進捗を表示
        from tqdm import tqdm
        for batch in tqdm(dataloader, desc="Evaluating"):
            metrics = self._run_step(batch, is_train=False)
            for key, value in metrics.items():
                total_metrics[key] += value
        return {key: value / num_batches for key, value in total_metrics.items()}
    
    # (save_checkpoint, load_checkpoint は変更なし)
