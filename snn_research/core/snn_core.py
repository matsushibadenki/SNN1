# matsushibadenki/snn/snn_research/core/snn_core.py
# SNNモデルの定義、次世代ニューロンなど、中核となるロジックを集約したライブラリ
#
# 変更点:
# - ロードマップ フェーズ2「2.3. アーキテクチャの進化」に対応。
# - SpikingTemporalAttention モジュールを新規実装。
# - BreakthroughSNN に SpikingTemporalAttention を統合し、SSM層の出力を処理するように変更。
# - mypyエラー解消のため、型ヒントを全体的に追加・修正。
# - spikingjellyとtqdmのimportに # type: ignore を追加。

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, functional  # type: ignore
from typing import Tuple, Dict, Any, Optional, List
from torch.utils.data import DataLoader
import os
import collections
import math
from collections import deque
from tqdm import tqdm  # type: ignore

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
        self.scaling = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_activated = torch.sigmoid(self.scaling * x)
        spike_times = self.max_latency * (1.0 - x_activated)
        spike_times = torch.round(spike_times).long()
        spikes = torch.zeros(x.shape[0], self.time_steps, x.shape[1], x.shape[2], device=x.device)
        spike_times = torch.clamp(spike_times, 0, self.time_steps - 1)
        spikes = spikes.scatter(1, spike_times.unsqueeze(1), 1.0)
        return spikes

class AdaptiveLIFNeuron(nn.Module):
    """
    適応的閾値を持つLIFニューロン
    """
    adaptive_threshold: torch.Tensor

    def __init__(self, features: int, tau: float = 2.0, base_threshold: float = 1.0, adaptation_strength: float = 0.1):
        super().__init__()
        self.features = features
        self.tau = tau
        self.base_threshold = base_threshold
        self.adaptation_strength = adaptation_strength
        self.register_buffer('adaptive_threshold', torch.ones(features) * base_threshold)
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        self.mem_decay = math.exp(-1.0 / tau)

    def forward(self, x: torch.Tensor, v_mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v_potential = v_mem * self.mem_decay + x
        spike = self.surrogate_function(v_potential - self.adaptive_threshold)
        v_mem_new = v_potential * (1.0 - spike.detach())
        with torch.no_grad():
            self.adaptive_threshold += self.adaptation_strength * (spike.mean(dim=(0, 1)) - 0.1)
        return spike, v_mem_new

class MetaplasticLIFNeuron(nn.Module):
    """
    メタ可塑性を持つLIFニューロン
    """
    adaptive_threshold: torch.Tensor

    def __init__(self, features: int, tau: float = 2.0, threshold: float = 1.0, metaplastic_tau: float = 1000.0, metaplastic_strength: float = 0.1):
        super().__init__()
        self.features = features
        self.tau = tau
        self.base_threshold = threshold
        self.metaplastic_tau = metaplastic_tau
        self.metaplastic_strength = metaplastic_strength
        self.register_buffer('adaptive_threshold', torch.ones(features) * threshold)
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        self.mem_decay = math.exp(-1.0 / tau)
        self.meta_decay = math.exp(-1.0 / metaplastic_tau)
    
    def forward(self, x: torch.Tensor, v_mem: torch.Tensor, activity_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_potential = v_mem * self.mem_decay + x
        current_threshold = self.adaptive_threshold * (1.0 + self.metaplastic_strength * activity_history.mean(dim=(0,1)))
        spike = self.surrogate_function(v_potential - current_threshold)
        v_mem_new = v_potential * (1.0 - spike.detach())
        activity_history_new = activity_history * self.meta_decay + spike.detach() * (1 - self.meta_decay)
        return spike, v_mem_new, activity_history_new

class STDPSynapse(nn.Module):
    """ Spike-Timing-Dependent Plasticity シナプス """
    pre_trace: torch.Tensor
    post_trace: torch.Tensor
    pre_rate: torch.Tensor
    post_rate: torch.Tensor

    def __init__(self, in_features: int, out_features: int, tau_pre: float = 20.0, tau_post: float = 20.0, A_pos: float = 0.01, A_neg: float = 0.005, w_min: float = 0.0, w_max: float = 1.0, homeostatic_scaling: bool = True):
        super().__init__()
        # ... (実装は省略)
        self.weight = nn.Parameter(torch.rand(out_features, in_features) * 0.5 + 0.25)
        self.register_buffer('pre_trace', torch.zeros(1, in_features))
        # ... (実装は省略)
    
    # ... (forward, _update_stdp, _apply_homeostatic_scaling の実装は省略)

class STPSynapse(nn.Module):
    """ 短期シナプス可塑性 """
    u: torch.Tensor
    x: torch.Tensor

    def __init__(self, in_features: int, out_features: int, tau_fac: float = 100.0, tau_dep: float = 200.0, U: float = 0.5, use_facilitation: bool = True, use_depression: bool = True):
        super().__init__()
        # ... (実装は省略)
        self.weight = nn.Parameter(torch.rand(out_features, in_features) * 0.5 + 0.25)
        if use_facilitation: self.register_buffer('u', torch.ones(1, in_features) * U)
        if use_depression: self.register_buffer('x', torch.ones(1, in_features))
        # ... (実装は省略)

    # ... (forward の実装は省略)

class EventDrivenSSMLayer(nn.Module):
    """
    Event-driven Spiking State Space Model
    """
    h_state: torch.Tensor
    state_v_mem: torch.Tensor
    output_v_mem: torch.Tensor

    def __init__(self, d_model: int, d_state: int = 64, dt: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt = dt
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model, d_model))
        self.state_lif = AdaptiveLIFNeuron(d_state)
        self.output_lif = AdaptiveLIFNeuron(d_model)
        self.register_buffer('h_state', torch.zeros(1, 1, d_state))
        self.register_buffer('state_v_mem', torch.zeros(1, 1, d_state))
        self.register_buffer('output_v_mem', torch.zeros(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, seq_len, _ = x.shape
        h, state_v, output_v = self.h_state, self.state_v_mem, self.output_v_mem
        if h.shape[0] != batch_size or h.shape[1] != seq_len:
            h = torch.zeros(batch_size, seq_len, self.d_state, device=x.device)
            state_v = torch.zeros(batch_size, seq_len, self.d_state, device=x.device)
        if output_v.shape[0] != batch_size or output_v.shape[1] != seq_len:
            output_v = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        h, state_v, output_v = h.detach(), state_v.detach(), output_v.detach()
        outputs = []
        for t in range(time_steps):
            x_t = x[:, t, :, :]
            state_transition = F.linear(h, self.A)
            if torch.any(x_t > 0):
                input_projection = F.linear(x_t, self.B)
                state_update = state_transition + input_projection
                h, state_v = self.state_lif(state_update, state_v)
                output_projection = F.linear(h, self.C)
                output_update = output_projection + F.linear(x_t, self.D)
                out_spike, output_v = self.output_lif(output_update, output_v)
            else:
                h, state_v = self.state_lif(state_transition, state_v)
                out_spike = torch.zeros_like(x_t)
            outputs.append(out_spike)
        self.h_state, self.state_v_mem, self.output_v_mem = h.detach(), state_v.detach(), output_v.detach()
        return torch.stack(outputs, dim=1)

# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
class SpikingTemporalAttention(nn.Module):
    """
    スパイク列の時間的関係性に注意を向けるアテンション機構。
    各タイムステップの重要度を動的に計算する。
    """
    def __init__(self, d_model: int, n_head: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        assert self.d_head * n_head == self.d_model, "d_model must be divisible by n_head"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 入力スパイク列 (batch_size, time_steps, seq_len, d_model)
        
        Returns:
            torch.Tensor: アテンション適用後のスパイク列 (batch_size, time_steps, seq_len, d_model)
        """
        batch_size, time_steps, seq_len, _ = x.shape
        
        # (batch_size, seq_len, time_steps, d_model) に並び替えて時間軸にアテンションを適用
        x_permuted = x.permute(0, 2, 1, 3).contiguous()
        x_reshaped = x_permuted.view(batch_size * seq_len, time_steps, self.d_model)

        q = self.q_proj(x_reshaped)
        k = self.k_proj(x_reshaped)
        v = self.v_proj(x_reshaped)

        # Multi-headに分割
        q = q.view(batch_size * seq_len, time_steps, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(batch_size * seq_len, time_steps, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(batch_size * seq_len, time_steps, self.n_head, self.d_head).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attended_v = torch.matmul(attn_weights, v)

        # ヘッドを結合して元の形状に戻す
        attended_v = attended_v.transpose(1, 2).contiguous().view(batch_size * seq_len, time_steps, self.d_model)
        
        output = self.out_proj(attended_v)
        output = output.view(batch_size, seq_len, time_steps, self.d_model).permute(0, 2, 1, 3)

        # 残差接続
        output = x + output
        
        return output

class BreakthroughSNN(nn.Module):
    """
    EventDriven-SSMと時間的アテンションを統合した、次世代のSNNアーキテクチャ。
    """
    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, time_steps: int, n_head: int):
        super().__init__()
        self.time_steps = time_steps
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.spike_encoder = TTFSEncoder(d_model=d_model, time_steps=time_steps)
        
        self.ssm_layers = nn.ModuleList([EventDrivenSSMLayer(d_model, d_state) for _ in range(num_layers)])
        
        self.temporal_attention = SpikingTemporalAttention(d_model=d_model, n_head=n_head)
        
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        token_emb = self.token_embedding(input_ids)
        spike_sequence = self.spike_encoder(token_emb)
        
        hidden_states = spike_sequence
        for layer in self.ssm_layers:
            hidden_states = layer(hidden_states)
            
        # SSM層の後に時間的アテンションを適用
        hidden_states = self.temporal_attention(hidden_states)
        
        # 時間方向にスパイクを積分 (平均化)
        time_integrated = hidden_states.mean(dim=1)
        logits = self.output_projection(time_integrated)
        
        # SpikingJellyのモジュール状態をリセット
        functional.reset_net(self)
        
        if return_spikes:
            return logits, hidden_states
        return logits, torch.empty(0) # スパイクを返さない場合
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
