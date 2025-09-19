# /path/to/your/project/snn_comprehensive_optimization.py
# SNNの総合的最適化システム：全ての先進技術を統合
# 
# 統合される最適化技術:
# - Time-to-First-Spike (TTFS) 符号化
# - 生物学的シナプス可塑性 (STDP, STP, メタ可塑性)
# - ニューロモーフィック最適化
# - マルチモーダル学習
# - エネルギー効率最大化
# - 動的時空間プルーニング
# - リアルタイム適応学習

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
import time
import math
from dataclasses import dataclass
from enum import Enum
import threading
from collections import deque, defaultdict

# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
# 先行モジュールからのインポートパスを更新
from snn_core import TTFSEncoder, AdaptiveLIFNeuron, EventDrivenSSMLayer, STDPSynapse, STPSynapse, MetaplasticLIFNeuron
from deployment import NeuromorphicProfile, NeuromorphicDeploymentManager, NeuromorphicChip
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

# ----------------------------------------
# 1. エネルギー効率最大化システム
# ----------------------------------------

class EnergyEfficiencyOptimizer:
    """
    0.3スパイク/ニューロンの超効率を目指すエネルギー最適化システム
    """
    def __init__(self, target_spike_rate: float = 0.3, 
                 energy_budget_pj: float = 1000.0):  # pJ per inference
        self.target_spike_rate = target_spike_rate
        self.energy_budget = energy_budget_pj
        self.current_efficiency = {}
        
        # Energy cost models (based on neuromorphic hardware research)
        self.energy_costs = {
            'spike_generation': 0.1,     # pJ per spike
            'spike_transmission': 0.05,  # pJ per synaptic transmission
            'weight_update': 1.0,        # pJ per STDP update
            'memory_access': 0.5,        # pJ per weight access
            'computation': 0.2           # pJ per MAC operation
        }
        
    def calculate_layer_energy(self, layer_name: str, spikes: torch.Tensor, 
                              weights: torch.Tensor, operations: Dict[str, int]) -> float:
        """レイヤーごとのエネルギー消費計算"""
        
        # Spike-related energy
        num_spikes = spikes.sum().item()
        spike_energy = num_spikes * self.energy_costs['spike_generation']
        
        # Synaptic transmission energy (proportional to active synapses)
        active_synapses = (spikes.unsqueeze(-1) * weights.abs() > 0.01).sum().item()
        transmission_energy = active_synapses * self.energy_costs['spike_transmission']
        
        # Memory access energy
        memory_accesses = operations.get('memory_accesses', 0)
        memory_energy = memory_accesses * self.energy_costs['memory_access']
        
        # Computational energy
        mac_operations = operations.get('mac_operations', 0)
        compute_energy = mac_operations * self.energy_costs['computation']
        
        total_energy = spike_energy + transmission_energy + memory_energy + compute_energy
        
        # Energy efficiency metric (information per picojoule)
        spike_rate = num_spikes / spikes.numel()
        efficiency = spike_rate / max(total_energy, 1e-6)  # spikes per pJ
        
        self.current_efficiency[layer_name] = {
            'total_energy_pj': total_energy,
            'spike_rate': spike_rate,
            'efficiency_spikes_per_pj': efficiency,
            'breakdown': {
                'spikes': spike_energy,
                'transmission': transmission_energy,
                'memory': memory_energy,
                'compute': compute_energy
            }
        }
        
        return total_energy
    
    def optimize_spike_patterns(self, spike_sequence: torch.Tensor) -> torch.Tensor:
        """スパイクパターンの最適化（エネルギー効率向上）"""
        batch_size, time_steps, seq_len, d_model = spike_sequence.shape
        
        # Temporal compression: reduce redundant spikes in consecutive time steps
        compressed_spikes = torch.zeros_like(spike_sequence)
        
        for t in range(time_steps):
            current_spikes = spike_sequence[:, t, :, :]
            
            if t == 0:
                # First time step: keep all significant spikes
                compressed_spikes[:, t, :, :] = current_spikes
            else:
                # Subsequent time steps: only keep new information
                prev_activity = compressed_spikes[:, t-1, :, :]
                
                # Only spike if significantly different from previous
                spike_threshold = 0.1
                new_info_mask = torch.abs(current_spikes - prev_activity) > spike_threshold
                compressed_spikes[:, t, :, :] = current_spikes * new_info_mask.float()
        
        # Spatial sparsification: keep only top-k% most important spikes per time step
        sparsity_ratio = 1.0 - self.target_spike_rate
        for t in range(time_steps):
            time_slice = compressed_spikes[:, t, :, :]
            flat_spikes = time_slice.view(-1)
            
            if flat_spikes.sum() > 0:
                # Keep top spikes based on magnitude
                k = int(len(flat_spikes) * (1.0 - sparsity_ratio))
                if k > 0:
                    topk_values, topk_indices = torch.topk(torch.abs(flat_spikes), k)
                    sparse_mask = torch.zeros_like(flat_spikes)
                    sparse_mask[topk_indices] = 1.0
                    
                    compressed_spikes[:, t, :, :] = time_slice * sparse_mask.view(time_slice.shape)
        
        return compressed_spikes

# ----------------------------------------
# 2. マルチモーダル統合SNNアーキテクチャ
# ----------------------------------------

class MultimodalSNN(nn.Module):
    """
    テキスト・視覚・音声を統合処理するマルチモーダルSNN
    """
    def __init__(self, 
                 vocab_size: int = 1000,
                 d_model: int = 256,
                 image_channels: int = 3,
                 audio_features: int = 128,
                 time_steps: int = 20,
                 fusion_strategy: str = "cross_attention"):
        super().__init__()
        
        self.d_model = d_model
        self.time_steps = time_steps
        self.fusion_strategy = fusion_strategy
        
        # Modality-specific encoders
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.vision_encoder = MultimodalVisionEncoder(image_channels, d_model, time_steps)
        self.audio_encoder = MultimodalAudioEncoder(audio_features, d_model, time_steps)
        
        # Unified spike encoder (TTFS for efficiency)
        self.spike_encoder = TTFSEncoder(d_model, time_steps)
        
        # Cross-modal fusion layers
        if fusion_strategy == "cross_attention":
            self.fusion_layer = CrossModalAttentionFusion(d_model, num_heads=8)
        else:
            self.fusion_layer = LinearModalityFusion(d_model * 3, d_model)
        
        # Shared SNN processing layers
        self.snn_layers = nn.ModuleList([
            EventDrivenSSMLayer(d_model, d_state=64) for _ in range(3)
        ])
        
        # Energy optimizer
        self.energy_optimizer = EnergyEfficiencyOptimizer()
        
        # Output projections for different tasks
        self.text_output = nn.Linear(d_model, vocab_size)
        self.vision_output = nn.Linear(d_model, 1000)  # ImageNet classes
        self.audio_output = nn.Linear(d_model, 256)    # Audio classes
        
    def forward(self, 
                text_input: Optional[torch.Tensor] = None,
                image_input: Optional[torch.Tensor] = None, 
                audio_input: Optional[torch.Tensor] = None,
                task: str = "text") -> torch.Tensor:
        
        modality_features = []
        modality_spikes = []
        
        # Process each available modality
        if text_input is not None:
            text_emb = self.text_embedding(text_input)
            text_spikes = self.spike_encoder(text_emb)
            modality_spikes.append(text_spikes)
            
        if image_input is not None:
            vision_spikes = self.vision_encoder(image_input)
            modality_spikes.append(vision_spikes)
            
        if audio_input is not None:
            audio_spikes = self.audio_encoder(audio_input)
            modality_spikes.append(audio_spikes)
        
        if not modality_spikes:
            raise ValueError("At least one modality input must be provided")
        
        # Cross-modal fusion
        if len(modality_spikes) > 1:
            fused_spikes = self.fusion_layer(modality_spikes)
        else:
            fused_spikes = modality_spikes[0]
        
        # Energy-optimized spike processing
        optimized_spikes = self.energy_optimizer.optimize_spike_patterns(fused_spikes)
        
        # SNN processing
        hidden_states = optimized_spikes
        for i, layer in enumerate(self.snn_layers):
            hidden_states = layer(hidden_states)
            
            # Calculate energy consumption
            layer_energy = self.energy_optimizer.calculate_layer_energy(
                f"snn_layer_{i}", 
                hidden_states,
                layer.C.data,  # Use output weights as representative
                {'memory_accesses': hidden_states.numel(), 
                 'mac_operations': hidden_states.sum().item()}
            )
        
        # Time integration
        integrated_features = hidden_states.mean(dim=1)  # (batch, seq_len, d_model)
        
        # Task-specific output
        if task == "text":
            return self.text_output(integrated_features)
        elif task == "vision":
            return self.vision_output(integrated_features.mean(dim=1))  # Global pooling
        elif task == "audio":
            return self.audio_output(integrated_features.mean(dim=1))
        else:
            return integrated_features

class MultimodalVisionEncoder(nn.Module):
    """視覚情報のSNN符号化"""
    def __init__(self, in_channels: int, d_model: int, time_steps: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.feature_projection = nn.Linear(256 * 64, d_model)
        self.spike_encoder = TTFSEncoder(d_model, time_steps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        features = self.conv_layers(x)  # (batch, 256, 8, 8)
        features = features.view(features.size(0), -1)  # (batch, 256*64)
        
        # Project to model dimension
        projected = self.feature_projection(features).unsqueeze(1)  # (batch, 1, d_model)
        
        # Convert to spikes
        spikes = self.spike_encoder(projected)
        return spikes

class MultimodalAudioEncoder(nn.Module):
    """音声情報のSNN符号化"""
    def __init__(self, in_features: int, d_model: int, time_steps: int):
        super().__init__()
        self.feature_projection = nn.Sequential(
            nn.Linear(in_features, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.spike_encoder = TTFSEncoder(d_model, time_steps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Audio feature projection
        projected = self.feature_projection(x)  # (batch, seq_len, d_model)
        
        # Convert to spikes
        spikes = self.spike_encoder(projected)
        return spikes

class CrossModalAttentionFusion(nn.Module):
    """クロスモーダル注意機構による融合"""
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, modality_spikes: List[torch.Tensor]) -> torch.Tensor:
        # Concatenate all modalities along sequence dimension
        batch_size, time_steps = modality_spikes[0].shape[:2]
        
        # Flatten time and sequence dimensions for attention
        flattened_spikes = []
        for spikes in modality_spikes:
            # (batch, time, seq, d_model) -> (batch, time*seq, d_model)
            flat_spikes = spikes.view(batch_size, -1, self.d_model)
            flattened_spikes.append(flat_spikes)
        
        # Concatenate modalities
        combined = torch.cat(flattened_spikes, dim=1)  # (batch, total_seq, d_model)
        
        # Multi-head attention
        Q = self.q_proj(combined)
        K = self.k_proj(combined) 
        V = self.v_proj(combined)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, V)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        fused = self.out_proj(attended)
        
        # Reshape back to spike format (batch, time, seq, d_model)
        # Take the first modality's shape as reference
        ref_shape = modality_spikes[0].shape
        seq_len = ref_shape[2]
        fused_spikes = fused[:, :time_steps*seq_len].view(batch_size, time_steps, seq_len, self.d_model)
        
        return fused_spikes

class LinearModalityFusion(nn.Module):
    """線形融合（シンプルな結合）"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fusion_layer = nn.Linear(input_dim, output_dim)
        
    def forward(self, modality_spikes: List[torch.Tensor]) -> torch.Tensor:
        # Simple concatenation and linear projection
        concatenated = torch.cat(modality_spikes, dim=-1)
        fused = self.fusion_layer(concatenated)
        return fused

# ----------------------------------------
# 3. 適応的リアルタイム学習システム
# ----------------------------------------

class AdaptiveRealtimeLearner:
    """
    リアルタイム環境での適応学習システム
    オンライン学習と継続学習を統合
    """
    def __init__(self, 
                 model: nn.Module,
                 base_lr: float = 1e-4,
                 adaptation_speed: str = "fast",
                 memory_size: int = 10000):
        
        self.model = model
        self.base_lr = base_lr
        self.adaptation_speed = adaptation_speed
        self.memory_size = memory_size
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=memory_size)
        
        # Adaptive learning rate
        self.adaptive_lr = base_lr
        self.lr_adaptation_factor = 0.1 if adaptation_speed == "slow" else 0.5
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.learning_efficiency = 1.0
        
        # Multi-objective optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.adaptive_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=1.2
        )
        
        # Loss components
        self.criterion = nn.CrossEntropyLoss()
        self.consistency_weight = 0.1
        self.energy_weight = 0.05
        
    def online_learning_step(self, 
                           new_data: Dict[str, torch.Tensor], 
                           targets: torch.Tensor) -> Dict[str, float]:
        """オンライン学習ステップ"""
        
        # Store experience for replay
        experience = {'data': new_data, 'targets': targets, 'timestamp': time.time()}
        self.experience_buffer.append(experience)
        
        # Forward pass
        self.model.train()
        logits = self.model(**new_data)
        
        # Multi-objective loss
        ce_loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Consistency loss (if we have history)
        consistency_loss = torch.tensor(0.0)
        if len(self.experience_buffer) > 1:
            prev_experience = self.experience_buffer[-2]
            with torch.no_grad():
                prev_logits = self.model(**prev_experience['data'])
            consistency_loss = F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.softmax(prev_logits, dim=-1),
                reduction='batchmean'
            )
        
        # Energy efficiency loss
        energy_loss = torch.tensor(0.0)
        if hasattr(self.model, 'energy_optimizer'):
            total_energy = sum(eff['total_energy_pj'] 
                             for eff in self.model.energy_optimizer.current_efficiency.values())
            energy_loss = torch.tensor(total_energy / 1000.0)  # Normalize
        
        # Combined loss
        total_loss = (ce_loss + 
                     self.consistency_weight * consistency_loss +
                     self.energy_weight * energy_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Adaptive gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update adaptive learning rate based on performance
        current_loss = total_loss.item()
        self.performance_history.append(current_loss)
        
        if len(self.performance_history) > 10:
            recent_trend = np.mean(list(self.performance_history)[-5:]) - np.mean(list(self.performance_history)[-10:-5])
            if recent_trend > 0:  # Performance degrading
                self.adaptive_lr *= (1 - self.lr_adaptation_factor)
            else:  # Performance improving
                self.adaptive_lr *= (1 + self.lr_adaptation_factor * 0.5)
            
            # Update optimizer learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.adaptive_lr
        
        return {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'energy_loss': energy_loss.item(),
            'learning_rate': self.adaptive_lr,
            'grad_norm': grad_norm.item()
        }
    
    def experience_replay_step(self, batch_size: int = 32) -> Dict[str, float]:
        """経験リプレイ学習"""
        if len(self.experience_buffer) < batch_size:
            return {}
        
        # Sample random batch from experience buffer
        import random
        batch = random.sample(list(self.experience_buffer), batch_size)
        
        # Reconstruct batch data
        batch_data = defaultdict(list)
        batch_targets = []
        
        for exp in batch:
            for key, value in exp['data'].items():
                batch_data[key].append(value)
            batch_targets.append(exp['targets'])
        
        # Convert to tensors
        for key in batch_data:
            batch_data[key] = torch.cat(batch_data[key], dim=0)
        batch_targets = torch.cat(batch_targets, dim=0)
        
        # Learning step
        return self.online_learning_step(batch_data, batch_targets)

# ----------------------------------------
# 4. 統合最適化SNNシステム
# ----------------------------------------

class ComprehensiveOptimizedSNN:
    """
    全ての最適化技術を統合したSNNシステム
    """
    def __init__(self, 
                 vocab_size: int = 1000,
                 d_model: int = 256,
                 optimization_level: str = "maximum_efficiency",
                 hardware_profile: Optional[NeuromorphicProfile] = None):
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.optimization_level = optimization_level
        
        # Create optimized multimodal model
        self.model = MultimodalSNN(
            vocab_size=vocab_size,
            d_model=d_model,
            time_steps=16 if optimization_level == "maximum_efficiency" else 20
        )
        
        # Neuromorphic deployment if hardware profile provided
        self.neuromorphic_manager = None
        if hardware_profile:
            self.neuromorphic_manager = NeuromorphicDeploymentManager(hardware_profile)
        
        # Adaptive learning system
        self.adaptive_learner = AdaptiveRealtimeLearner(
            self.model, 
            adaptation_speed="fast" if optimization_level == "maximum_efficiency" else "balanced"
        )
        
        # Performance tracking
        self.performance_metrics = {
            'inference_times': deque(maxlen=1000),
            'energy_consumption': deque(maxlen=1000),
            'spike_rates': deque(maxlen=1000),
            'accuracy_samples': deque(maxlen=1000)
        }
        
    def deploy(self, deployment_name: str = "comprehensive_snn"):
        """統合最適化システムのデプロイ"""
        print(f"🚀 統合最適化SNNシステム デプロイ開始: {deployment_name}")
        
        if self.neuromorphic_manager:
            # Neuromorphic hardware deployment
            self.neuromorphic_manager.deploy_model(
                self.model, deployment_name, 
                optimization_target="ultra_low_power" if self.optimization_level == "maximum_efficiency" else "balanced"
            )
            print("✅ ニューロモーフィックデプロイメント完了")
        else:
            print("✅ 標準デプロイメント完了")
        
    def comprehensive_inference(self, 
                              text_input: Optional[torch.Tensor] = None,
                              image_input: Optional[torch.Tensor] = None,
                              audio_input: Optional[torch.Tensor] = None,
                              task: str = "text") -> Dict[str, Any]:
        """包括的推論（全最適化適用）"""
        
        start_time = time.time()
        
        # Inference
        if self.neuromorphic_manager:
            # Neuromorphic optimized inference
            result = self.neuromorphic_manager.inference(
                "comprehensive_snn", text_input
            )
            perf_stats = {'latency_ms': (time.time() - start_time) * 1000}

        else:
            # Standard optimized inference
            self.model.eval()
            with torch.no_grad():
                result = self.model(
                    text_input=text_input,
                    image_input=image_input, 
                    audio_input=audio_input,
                    task=task
                )
            
            latency = (time.time() - start_time) * 1000
            perf_stats = {
                'latency_ms': latency,
                'throughput_infer_sec': 1000.0 / latency if latency > 0 else 0
            }
        
        # Collect energy statistics
        energy_stats = {}
        if hasattr(self.model, 'energy_optimizer'):
            total_energy = sum(eff['total_energy_pj'] 
                             for eff in self.model.energy_optimizer.current_efficiency.values())
            avg_spike_rate = np.mean([eff['spike_rate'] 
                                    for eff in self.model.energy_optimizer.current_efficiency.values()])
            
            energy_stats = {
                'total_energy_pj': total_energy,
                'avg_spike_rate': avg_spike_rate,
                'energy_per_spike_pj': total_energy / max(avg_spike_rate * self.d_model, 1),
                'efficiency_target_met': avg_spike_rate <= 0.35  # Close to 0.3 target
            }
        
        # Update performance tracking
        self.performance_metrics['inference_times'].append(perf_stats['latency_ms'])
        if energy_stats:
            self.performance_metrics['energy_consumption'].append(energy_stats['total_energy_pj'])
            self.performance_metrics['spike_rates'].append(energy_stats['avg_spike_rate'])
        
        return {
            'result': result,
            'performance': perf_stats,
            'energy': energy_stats,
            'optimization_level': self.optimization_level
        }
    
    def adaptive_learning_update(self, 
                               new_data: Dict[str, torch.Tensor],
                               targets: torch.Tensor) -> Dict[str, float]:
        """適応学習アップデート"""
        return self.adaptive_learner.online_learning_step(new_data, targets)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """包括的統計情報"""
        stats = {
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'optimization_level': self.optimization_level
        }
        
        # Performance statistics
        if self.performance_metrics['inference_times']:
            stats['performance'] = {
                'avg_latency_ms': np.mean(self.performance_metrics['inference_times']),
                'p95_latency_ms': np.percentile(self.performance_metrics['inference_times'], 95),
            }
        
        # Energy statistics
        if self.performance_metrics['energy_consumption']:
            stats['energy'] = {
                'avg_energy_pj': np.mean(self.performance_metrics['energy_consumption']),
                'avg_spike_rate': np.mean(self.performance_metrics['spike_rates']) if self.performance_metrics['spike_rates'] else 0,
            }
        
        return stats

# ----------------------------------------
# 5. 包括的ベンチマークシステム
# ----------------------------------------

def comprehensive_snn_benchmark():
    """包括的SNNシステムの性能ベンチマーク"""
    print("🌟 包括的SNNシステム ベンチマーク開始")
    
    neuromorphic_profile = NeuromorphicProfile(
        chip_type=NeuromorphicChip.INTEL_LOIHI,
        num_cores=64,
        memory_hierarchy={"L1": 32768, "L2": 262144, "DRAM": 4294967296},
        power_budget_mw=50.0
    )
    
    comprehensive_system = ComprehensiveOptimizedSNN(
        vocab_size=5000,
        d_model=256,
        optimization_level="maximum_efficiency",
        hardware_profile=neuromorphic_profile
    )
    
    comprehensive_system.deploy("ultra_efficient_snn")
    
    text_data = torch.randint(0, 5000, (8, 32))
    
    print(f"\n📝 テキスト単一モダリティテスト:")
    for i in range(5): # Reduced for brevity
        comprehensive_system.comprehensive_inference(text_input=text_data, task="text")
    
    print(f"\n📈 総合性能統計:")
    final_stats = comprehensive_system.get_comprehensive_stats()
    print(final_stats)

if __name__ == "__main__":
    comprehensive_snn_benchmark()
