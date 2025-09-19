# /path/to/your/project/snn_advanced_plasticity.py
# 生物学的シナプス可塑性を統合した次世代SNNシステム
# 
# 最新研究に基づく実装:
# - STDP (Spike-Timing-Dependent Plasticity) 
# - 短期シナプス可塑性 (STP)
# - メタ可塑性 (Metaplasticity)
# - ホメオスタシス機構
# - リザバーコンピューティング的自己組織化

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from spikingjelly.activation_based import neuron, surrogate, functional
import math
from collections import deque

# ----------------------------------------
# 1. STDP可塑性モジュール
# ----------------------------------------

class STDPSynapse(nn.Module):
    """
    Spike-Timing-Dependent Plasticity シナプス
    2024年の最新研究に基づく高効率STDP実装
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 tau_pre: float = 20.0,      # プレシナプスの時定数
                 tau_post: float = 20.0,     # ポストシナプスの時定数
                 A_pos: float = 0.01,        # LTP強度
                 A_neg: float = 0.005,       # LTD強度
                 w_min: float = 0.0,         # 最小重み
                 w_max: float = 1.0,         # 最大重み
                 homeostatic_scaling: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.A_pos = A_pos
        self.A_neg = A_neg
        self.w_min = w_min
        self.w_max = w_max
        self.homeostatic_scaling = homeostatic_scaling
        
        # シナプス重み
        self.weight = nn.Parameter(torch.rand(out_features, in_features) * 0.5 + 0.25)
        
        # STDPトレース（プレ・ポストシナプスの活動履歴）
        self.register_buffer('pre_trace', torch.zeros(1, in_features))
        self.register_buffer('post_trace', torch.zeros(1, out_features))
        
        # ホメオスタシス用の活動レート追跡
        if homeostatic_scaling:
            self.register_buffer('pre_rate', torch.ones(in_features) * 0.02)
            self.register_buffer('post_rate', torch.ones(out_features) * 0.02)
            self.target_rate = 0.02  # 目標発火率
            self.homeostatic_alpha = 0.001
        
        # 減衰係数（事前計算で効率化）
        self.pre_decay = math.exp(-1.0 / tau_pre)
        self.post_decay = math.exp(-1.0 / tau_post)
        
    def forward(self, pre_spike: torch.Tensor, post_spike: torch.Tensor, 
                learning: bool = True) -> torch.Tensor:
        """
        Args:
            pre_spike: プレシナプススパイク (batch_size, in_features)
            post_spike: ポストシナプススパイク (batch_size, out_features)
            learning: STDP学習を実行するかどうか
        Returns:
            更新されたシナプス出力
        """
        batch_size = pre_spike.shape[0]
        
        # バッチサイズに合わせてトレースを調整
        if self.pre_trace.shape[0] != batch_size:
            self.pre_trace = torch.zeros(batch_size, self.in_features, device=pre_spike.device)
            self.post_trace = torch.zeros(batch_size, self.out_features, device=post_spike.device)
        
        # シナプス出力計算
        output = F.linear(pre_spike, self.weight)
        
        if learning:
            self._update_stdp(pre_spike, post_spike)
        
        # トレース更新（指数減衰 + 新しいスパイク）
        self.pre_trace = self.pre_trace * self.pre_decay + pre_spike
        self.post_trace = self.post_trace * self.post_decay + post_spike
        
        return output
    
    def _update_stdp(self, pre_spike: torch.Tensor, post_spike: torch.Tensor):
        """STDP重み更新"""
        # LTP (Long-Term Potentiation): ポストスパイクがプレトレースと相関
        # pre_trace は過去のプレスパイクの履歴、post_spike は現在のポストスパイク
        ltp_update = torch.outer(post_spike.mean(0), self.pre_trace.mean(0)) * self.A_pos
        
        # LTD (Long-Term Depression): プレスパイクがポストトレースと相関  
        # pre_spike は現在のプレスパイク、post_trace は過去のポストスパイクの履歴
        ltd_update = torch.outer(self.post_trace.mean(0), pre_spike.mean(0)) * self.A_neg
        
        # 重み更新
        delta_w = ltp_update - ltd_update
        
        # ホメオスタシススケーリング
        if self.homeostatic_scaling:
            delta_w = self._apply_homeostatic_scaling(delta_w, pre_spike, post_spike)
        
        # 重み更新と制約
        with torch.no_grad():
            self.weight.data += delta_w
            self.weight.data.clamp_(self.w_min, self.w_max)
    
    def _apply_homeostatic_scaling(self, delta_w: torch.Tensor, 
                                 pre_spike: torch.Tensor, post_spike: torch.Tensor) -> torch.Tensor:
        """ホメオスタシススケーリング：発火率を目標値に維持"""
        # 活動レート更新（指数移動平均）
        current_pre_rate = pre_spike.mean(0)
        current_post_rate = post_spike.mean(0)
        
        self.pre_rate = (1 - self.homeostatic_alpha) * self.pre_rate + self.homeostatic_alpha * current_pre_rate
        self.post_rate = (1 - self.homeostatic_alpha) * self.post_rate + self.homeostatic_alpha * current_post_rate
        
        # スケーリングファクター計算
        pre_scaling = self.target_rate / (self.pre_rate + 1e-6)
        post_scaling = self.target_rate / (self.post_rate + 1e-6)
        
        # 重み更新にスケーリング適用
        scaling_matrix = torch.outer(post_scaling, pre_scaling)
        return delta_w * scaling_matrix.sqrt()

# ----------------------------------------
# 2. 短期シナプス可塑性 (STP)
# ----------------------------------------

class STPSynapse(nn.Module):
    """
    短期シナプス可塑性 - 促進と抑圧の動的制御
    時間スケールの異なる適応を実現
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 tau_fac: float = 100.0,     # 促進の時定数
                 tau_dep: float = 200.0,     # 抑圧の時定数
                 U: float = 0.5,             # 使用率
                 use_facilitation: bool = True,
                 use_depression: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.tau_fac = tau_fac
        self.tau_dep = tau_dep
        self.U = U
        self.use_facilitation = use_facilitation
        self.use_depression = use_depression
        
        # 基本重み
        self.weight = nn.Parameter(torch.rand(out_features, in_features) * 0.5 + 0.25)
        
        # STP状態変数
        if use_facilitation:
            self.register_buffer('u', torch.ones(1, in_features) * U)  # 促進変数
        if use_depression:
            self.register_buffer('x', torch.ones(1, in_features))      # リソース変数
        
        # 減衰係数
        if use_facilitation:
            self.fac_decay = math.exp(-1.0 / tau_fac)
        if use_depression:
            self.dep_decay = math.exp(-1.0 / tau_dep)
    
    def forward(self, pre_spike: torch.Tensor) -> torch.Tensor:
        batch_size = pre_spike.shape[0]
        
        # バッチサイズ調整
        if self.use_facilitation and self.u.shape[0] != batch_size:
            self.u = torch.ones(batch_size, self.in_features, device=pre_spike.device) * self.U
        if self.use_depression and self.x.shape[0] != batch_size:
            self.x = torch.ones(batch_size, self.in_features, device=pre_spike.device)
        
        # 現在の有効重み計算
        effective_weight = self.weight.clone()
        
        if self.use_facilitation and self.use_depression:
            # 促進と抑圧の両方
            release_prob = self.u * self.x
            effective_weight = effective_weight * release_prob.unsqueeze(0)
            
            # 状態更新
            self.u = self.u * self.fac_decay + self.U * (1 - self.u * self.fac_decay) * pre_spike
            self.x = self.x * self.dep_decay + (1 - self.x) * self.dep_decay * (1 - pre_spike * self.u)
            
        elif self.use_facilitation:
            # 促進のみ
            effective_weight = effective_weight * self.u.unsqueeze(0)
            self.u = self.u * self.fac_decay + self.U * (1 - self.u * self.fac_decay) * pre_spike
            
        elif self.use_depression:
            # 抑圧のみ
            effective_weight = effective_weight * self.x.unsqueeze(0)
            self.x = self.x * self.dep_decay + (1 - self.x) * (1 - pre_spike)
        
        return F.linear(pre_spike, effective_weight)

# ----------------------------------------
# 3. メタ可塑性ニューロン
# ----------------------------------------

class MetaplasticLIFNeuron(nn.Module):
    """
    メタ可塑性を持つLIFニューロン
    学習履歴に基づいて可塑性を動的に調整
    """
    def __init__(self, 
                 features: int,
                 tau: float = 2.0,
                 threshold: float = 1.0,
                 metaplastic_tau: float = 1000.0,  # メタ可塑性の時定数
                 metaplastic_strength: float = 0.1):
        super().__init__()
        
        self.features = features
        self.tau = tau
        self.base_threshold = threshold
        self.metaplastic_tau = metaplastic_tau
        self.metaplastic_strength = metaplastic_strength
        
        # ニューロンの状態
        self.register_buffer('v_mem', torch.zeros(1, features))
        self.register_buffer('activity_history', torch.zeros(1, features))  # メタ可塑性用
        
        # 適応的閾値（学習に基づいて変化）
        self.register_buffer('adaptive_threshold', torch.ones(features) * threshold)
        
        # 代理勾配
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        
        # 減衰係数
        self.mem_decay = math.exp(-1.0 / tau)
        self.meta_decay = math.exp(-1.0 / metaplastic_tau)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # バッチサイズ調整
        if self.v_mem.shape[0] != batch_size:
            self.v_mem = torch.zeros(batch_size, self.features, device=x.device)
            self.activity_history = torch.zeros(batch_size, self.features, device=x.device)
        
        # 膜電位更新
        self.v_mem = self.v_mem * self.mem_decay + x
        
        # メタ可塑性に基づく適応的閾値
        current_threshold = self.adaptive_threshold * (1.0 + self.metaplastic_strength * self.activity_history.mean(0))
        
        # スパイク発生
        spike = self.surrogate_function(self.v_mem - current_threshold)
        
        # 発火後リセット
        self.v_mem = self.v_mem * (1.0 - spike.detach())
        
        # 活動履歴更新（メタ可塑性用）
        self.activity_history = self.activity_history * self.meta_decay + spike.detach() * (1 - self.meta_decay)
        
        return spike

# ----------------------------------------
# 4. 生物学的可塑性統合SNNアーキテクチャ
# ----------------------------------------

class BioplasticSNN(nn.Module):
    """
    生物学的シナプス可塑性を統合したSNNアーキテクチャ
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 256,
                 num_layers: int = 3,
                 time_steps: int = 20,
                 use_stdp: bool = True,
                 use_stp: bool = True,
                 use_metaplasticity: bool = True):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.time_steps = time_steps
        self.use_stdp = use_stdp
        self.use_stp = use_stp
        self.use_metaplasticity = use_metaplasticity
        
        # 埋め込み層
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # スパイクエンコーダー（TTFSベース）
        from snn_advanced_optimization import TTFSEncoder
        self.spike_encoder = TTFSEncoder(d_model, time_steps)
        
        # 可塑性シナプス層
        self.plastic_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_dict = nn.ModuleDict()
            
            if use_stdp:
                layer_dict['stdp_synapse'] = STDPSynapse(d_model, d_model)
            if use_stp:
                layer_dict['stp_synapse'] = STPSynapse(d_model, d_model)
            if use_metaplasticity:
                layer_dict['metaplastic_neuron'] = MetaplasticLIFNeuron(d_model)
            else:
                # 標準LIF
                layer_dict['lif_neuron'] = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
            
            self.plastic_layers.append(layer_dict)
        
        # 出力層
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # リザバー状態（自己組織化用）
        self.register_buffer('reservoir_state', torch.zeros(1, d_model))
        
    def forward(self, input_ids: torch.Tensor, learning_mode: bool = True) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # 埋め込み
        token_emb = self.token_embedding(input_ids)
        
        # スパイク符号化
        spike_sequence = self.spike_encoder(token_emb)
        
        # 可塑性層での処理
        hidden_states = spike_sequence
        
        for t in range(self.time_steps):
            current_spikes = hidden_states[:, t, :, :]  # (batch, seq_len, d_model)
            
            layer_output = current_spikes
            for layer in self.plastic_layers:
                if 'stdp_synapse' in layer:
                    # STDP学習（前の層の出力を使用）
                    pre_spikes = layer_output
                    post_spikes = layer['metaplastic_neuron'](layer_output) if 'metaplastic_neuron' in layer else layer_output
                    layer_output = layer['stdp_synapse'](pre_spikes.view(-1, self.d_model), 
                                                       post_spikes.view(-1, self.d_model), 
                                                       learning=learning_mode).view(batch_size, seq_len, self.d_model)
                
                if 'stp_synapse' in layer:
                    # 短期可塑性
                    layer_output = layer['stp_synapse'](layer_output.view(-1, self.d_model)).view(batch_size, seq_len, self.d_model)
                
                if 'metaplastic_neuron' in layer:
                    # メタ可塑性ニューロン
                    layer_output = layer['metaplastic_neuron'](layer_output.view(-1, self.d_model)).view(batch_size, seq_len, self.d_model)
            
            hidden_states[:, t, :, :] = layer_output
        
        # 時間統合
        time_integrated = hidden_states.mean(dim=1)  # (batch, seq_len, d_model)
        
        # 出力投影
        logits = self.output_projection(time_integrated)
        
        return logits

# ----------------------------------------
# 5. 適応的学習システム
# ----------------------------------------

class AdaptivePlasticityTrainer:
    """
    生物学的可塑性を持つSNNの適応的学習システム
    """
    def __init__(self, 
                 model: BioplasticSNN,
                 base_lr: float = 1e-4,
                 plasticity_lr: float = 1e-3,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.model = model.to(device)
        self.device = device
        
        # 異なる学習率でオプティマイザーを分離
        # 埋め込みと出力層：低学習率
        backbone_params = list(self.model.token_embedding.parameters()) + list(self.model.output_projection.parameters())
        self.backbone_optimizer = torch.optim.AdamW(backbone_params, lr=base_lr)
        
        # 可塑性層：高学習率（生物学的学習の高速性を模倣）
        plasticity_params = []
        for layer in self.model.plastic_layers:
            plasticity_params.extend(layer.parameters())
        self.plasticity_optimizer = torch.optim.AdamW(plasticity_params, lr=plasticity_lr)
        
        # 損失関数（可塑性を考慮）
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 適応的学習率スケジューラー
        self.backbone_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.backbone_optimizer, T_0=50, T_mult=2)
        self.plasticity_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.plasticity_optimizer, T_0=20, T_mult=1.5)
        
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """生物学的可塑性を考慮した学習ステップ"""
        input_ids, target_ids = [t.to(self.device) for t in batch]
        
        self.model.train()
        
        # Forward pass（可塑性学習モード）
        logits = self.model(input_ids, learning_mode=True)
        
        # 損失計算
        loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        # Backward pass
        self.backbone_optimizer.zero_grad()
        self.plasticity_optimizer.zero_grad()
        
        loss.backward()
        
        # 勾配クリッピング（可塑性の安定化）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # オプティマイザー更新
        self.backbone_optimizer.step()
        self.plasticity_optimizer.step()
        
        # スケジューラー更新
        self.backbone_scheduler.step()
        self.plasticity_scheduler.step()
        
        return {'loss': loss.item()}

# ----------------------------------------
# 6. 使用例とベンチマーク
# ----------------------------------------

def test_bioplastic_snn():
    """生物学的可塑性SNNのテスト"""
    print("🧠 生物学的可塑性SNNのテストを開始...")
    
    # パラメータ
    vocab_size = 1000
    d_model = 128
    batch_size = 4
    seq_len = 16
    time_steps = 12
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデル作成
    model = BioplasticSNN(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=3,
        time_steps=time_steps,
        use_stdp=True,
        use_stp=True,
        use_metaplasticity=True
    )
    
    # テストデータ
    test_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    test_targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 学習前の推論
    model.eval()
    with torch.no_grad():
        logits_before = model(test_input, learning_mode=False)
        initial_acc = (logits_before.argmax(-1) == test_targets).float().mean()
    
    print(f"初期精度: {initial_acc:.3f}")
    
    # 適応的学習
    trainer = AdaptivePlasticityTrainer(model)
    
    print("🔄 生物学的可塑性による適応学習...")
    for step in range(50):
        metrics = trainer.train_step((test_input, test_targets))
        if (step + 1) % 10 == 0:
            print(f"Step {step+1}: Loss = {metrics['loss']:.4f}")
    
    # 学習後の推論
    model.eval()
    with torch.no_grad():
        logits_after = model(test_input, learning_mode=False)
        final_acc = (logits_after.argmax(-1) == test_targets).float().mean()
    
    print(f"最終精度: {final_acc:.3f}")
    print(f"精度向上: {final_acc - initial_acc:.3f}")
    
    # 可塑性の分析
    print("\n📊 シナプス可塑性分析:")
    for i, layer in enumerate(model.plastic_layers):
        if 'stdp_synapse' in layer:
            stdp_weights = layer['stdp_synapse'].weight.data
            print(f"Layer {i} STDP重み範囲: [{stdp_weights.min():.3f}, {stdp_weights.max():.3f}]")
            print(f"Layer {i} STDP重み平均: {stdp_weights.mean():.3f}")
    
    print("✅ 生物学的可塑性SNNテスト完了")

if __name__ == "__main__":
    test_bioplastic_snn()