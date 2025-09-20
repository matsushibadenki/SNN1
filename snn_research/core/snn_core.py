# matsushibadenki/snn/snn_research/core/snn_core.py
# SNNモデルの定義、次世代ニューロンなど、中核となるロジックを集約したライブラリ
# 
# 変更点:
# - CombinedLossの定義を削除（losses.pyに移動したため）

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
from tqdm import tqdm

# (TTFSEncoder, AdaptiveLIFNeuron, MetaplasticLIFNeuron, STDPSynapse, STPSynapse, EventDrivenSSMLayer, SpikingTemporalAttention, BreakthroughSNN のコードは変更なし)
# ... (既存のコードをここにペースト) ...
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
        
        self.register_buffer('v_mem', torch.zeros(1, features))
        self.register_buffer('adaptive_threshold', torch.ones(features) * base_threshold)
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        self.mem_decay = math.exp(-1.0 / tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # v_memのバッチサイズを動的に調整
        if self.v_mem.shape[0] != batch_size:
            self.v_mem = self.v_mem[0].expand(batch_size, -1).clone()

        self.v_mem = self.v_mem * self.mem_decay + x
        spike = self.surrogate_function(self.v_mem - self.adaptive_threshold)
        self.v_mem = self.v_mem * (1.0 - spike.detach())
        
        # 適応的閾値の更新
        with torch.no_grad():
            self.adaptive_threshold += self.adaptation_strength * (spike.mean(0) - 0.1) # 目標発火率0.1
        
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
class STDPSynapse(nn.Module):
    """ Spike-Timing-Dependent Plasticity シナプス """
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 tau_pre: float = 20.0, tau_post: float = 20.0,
                 A_pos: float = 0.01, A_neg: float = 0.005,
                 w_min: float = 0.0, w_max: float = 1.0,
                 homeostatic_scaling: bool = True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.tau_pre, self.tau_post = tau_pre, tau_post
        self.A_pos, self.A_neg = A_pos, A_neg
        self.w_min, self.w_max = w_min, w_max
        self.homeostatic_scaling = homeostatic_scaling
        
        self.weight = nn.Parameter(torch.rand(out_features, in_features) * 0.5 + 0.25)
        self.register_buffer('pre_trace', torch.zeros(1, in_features))
        self.register_buffer('post_trace', torch.zeros(1, out_features))
        
        if homeostatic_scaling:
            self.register_buffer('pre_rate', torch.ones(in_features) * 0.02)
            self.register_buffer('post_rate', torch.ones(out_features) * 0.02)
            self.target_rate = 0.02
            self.homeostatic_alpha = 0.001
        
        self.pre_decay = math.exp(-1.0 / tau_pre)
        self.post_decay = math.exp(-1.0 / tau_post)
        
    def forward(self, pre_spike: torch.Tensor, post_spike: torch.Tensor, 
                learning: bool = True) -> torch.Tensor:
        batch_size = pre_spike.shape[0]
        if self.pre_trace.shape[0] != batch_size:
            self.pre_trace = torch.zeros(batch_size, self.in_features, device=pre_spike.device)
            self.post_trace = torch.zeros(batch_size, self.out_features, device=post_spike.device)
        
        output = F.linear(pre_spike, self.weight)
        if learning: self._update_stdp(pre_spike, post_spike)
        
        self.pre_trace = self.pre_trace * self.pre_decay + pre_spike
        self.post_trace = self.post_trace * self.post_decay + post_spike
        return output
    
    def _update_stdp(self, pre_spike: torch.Tensor, post_spike: torch.Tensor):
        ltp_update = torch.outer(post_spike.mean(0), self.pre_trace.mean(0)) * self.A_pos
        ltd_update = torch.outer(self.post_trace.mean(0), pre_spike.mean(0)) * self.A_neg
        delta_w = ltp_update - ltd_update
        if self.homeostatic_scaling: delta_w = self._apply_homeostatic_scaling(delta_w, pre_spike, post_spike)
        
        with torch.no_grad():
            self.weight.data += delta_w
            self.weight.data.clamp_(self.w_min, self.w_max)
    
    def _apply_homeostatic_scaling(self, delta_w, pre_spike, post_spike):
        current_pre_rate = pre_spike.mean(0)
        current_post_rate = post_spike.mean(0)
        self.pre_rate = (1 - self.homeostatic_alpha) * self.pre_rate + self.homeostatic_alpha * current_pre_rate
        self.post_rate = (1 - self.homeostatic_alpha) * self.post_rate + self.homeostatic_alpha * current_post_rate
        pre_scaling = self.target_rate / (self.pre_rate + 1e-6)
        post_scaling = self.target_rate / (self.post_rate + 1e-6)
        return delta_w * torch.outer(post_scaling, pre_scaling).sqrt()

class STPSynapse(nn.Module):
    """ 短期シナプス可塑性 """
    def __init__(self, 
                 in_features: int, out_features: int,
                 tau_fac: float = 100.0, tau_dep: float = 200.0, U: float = 0.5,
                 use_facilitation: bool = True, use_depression: bool = True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.tau_fac, self.tau_dep, self.U = tau_fac, tau_dep, U
        self.use_facilitation, self.use_depression = use_facilitation, use_depression
        
        self.weight = nn.Parameter(torch.rand(out_features, in_features) * 0.5 + 0.25)
        if use_facilitation: self.register_buffer('u', torch.ones(1, in_features) * U)
        if use_depression: self.register_buffer('x', torch.ones(1, in_features))
        if use_facilitation: self.fac_decay = math.exp(-1.0 / tau_fac)
        if use_depression: self.dep_decay = math.exp(-1.0 / tau_dep)
    
    def forward(self, pre_spike: torch.Tensor) -> torch.Tensor:
        batch_size = pre_spike.shape[0]
        if self.use_facilitation and self.u.shape[0] != batch_size:
            self.u = torch.ones(batch_size, self.in_features, device=pre_spike.device) * self.U
        if self.use_depression and self.x.shape[0] != batch_size:
            self.x = torch.ones(batch_size, self.in_features, device=pre_spike.device)
        
        effective_weight = self.weight.clone()
        if self.use_facilitation and self.use_depression:
            release_prob = self.u * self.x
            effective_weight = effective_weight * release_prob.unsqueeze(0)
            self.u = self.u * self.fac_decay + self.U * (1 - self.u * self.fac_decay) * pre_spike
            self.x = self.x * self.dep_decay + (1 - self.x) * self.dep_decay * (1 - pre_spike * self.u)
        return F.linear(pre_spike, effective_weight)


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
        self.D = nn.Parameter(torch.randn(d_model, d_model))
        
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
            if torch.any(x_t > 0):
                state_transition = F.linear(self.h_state, self.A)
                input_projection = F.linear(x_t, self.B.T)
                state_update = state_transition + input_projection
                self.h_state = self.state_lif(state_update)
                
                output_projection = F.linear(self.h_state, self.C.T)
                output_update = output_projection + F.linear(x_t, self.D)
                out_spike = self.output_lif(output_update)
            else:
                self.h_state = self.state_lif(F.linear(self.h_state, self.A))
                out_spike = torch.zeros_like(x_t)
            
            outputs.append(out_spike)
        
        functional.reset_net(self)
        return torch.stack(outputs, dim=1)

# ----------------------------------------
# 5. 時間的アテンション機構
# ----------------------------------------

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

# ----------------------------------------
# 6. 統合された次世代SNNアーキテクチャ
# ----------------------------------------
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
            
        hidden_states = self.temporal_attention(hidden_states)
        
        time_integrated = hidden_states.mean(dim=1)
        logits = self.output_projection(time_integrated)
        
        return logits, hidden_states