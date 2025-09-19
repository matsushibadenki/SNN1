# /path/to/your/project/deployment.py
# SNNの実用デプロイメントのための最適化、監視、継続学習システム
#
# 元ファイル: snn_deployment_optimization.py (全機能統合)
# 改善点:
# - 手動のプルーニングと量子化を、より堅牢で高性能なPyTorch公式ユーティリティに置き換え。
# - 継続学習の知識蒸留損失をより厳密な計算に変更。

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
import time
import copy
from dataclasses import dataclass
from enum import Enum
# PyTorchの高度な最適化ツールをインポート
import torch.quantization
from torch.nn.utils import prune

# ----------------------------------------
# 1. 動的最適化エンジン
# ----------------------------------------

class OptimizationLevel(Enum):
    """最適化レベル"""
    ULTRA_LOW_POWER = "ultra_low_power"
    BALANCED = "balanced"
    HIGH_PERFORMANCE = "high_performance"

@dataclass
class HardwareProfile:
    """ハードウェアプロファイル"""
    device_type: str
    memory_limit_gb: float
    power_budget_w: float
    supports_neuromorphic: bool = False

class DynamicOptimizer:
    """動的最適化エンジン (PyTorchユーティリティ使用)"""
    def __init__(self, model: nn.Module, hardware_profile: HardwareProfile):
        self.model = model
        self.hardware = hardware_profile

    def optimize_for_deployment(self, target_level: OptimizationLevel) -> nn.Module:
        """デプロイメント向け最適化"""
        print(f"🔧 {target_level.value} モードで最適化開始...")
        config = self._get_config(target_level)
        
        # モデルをCPUに移動して最適化処理を実行
        optimized_model = copy.deepcopy(self.model).cpu()
        optimized_model.eval()

        print("  ⚡ プルーニングを適用中...")
        self._apply_pruning(optimized_model, config['pruning_ratio'])
        
        print("  ⚡ 量子化を適用中...")
        optimized_model = self._apply_quantization(optimized_model, config['quantization_bits'])
        
        print("✅ 最適化完了")
        return optimized_model

    def _get_config(self, level: OptimizationLevel) -> Dict[str, Any]:
        if level == OptimizationLevel.ULTRA_LOW_POWER:
            return {'pruning_ratio': 0.8, 'quantization_bits': 8} # INT8
        elif level == OptimizationLevel.BALANCED:
            return {'pruning_ratio': 0.5, 'quantization_bits': 16} # FP16
        else: # HIGH_PERFORMANCE
            return {'pruning_ratio': 0.2, 'quantization_bits': 32} # FP32

    def _apply_pruning(self, model: nn.Module, pruning_ratio: float):
        """
        PyTorchの prune ユーティリティを使用した構造化プルーニング。
        L1ノルム（重みの絶対値）が小さいものを重要度が低いと見なし、除去します。
        """
        if pruning_ratio <= 0: return
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=pruning_ratio)
                # プルーニングを恒久的に適用（マスクを削除し、重みを直接0にする）
                prune.remove(module, 'weight')
    
    def _apply_quantization(self, model: nn.Module, bits: int) -> nn.Module:
        """
        PyTorchの quantization ユーティリティを使用した動的量子化。
        モデルのサイズを削減し、推論を高速化します。
        """
        if bits >= 32: return model # FP32 (量子化なし)
        
        if bits == 8:
            # INT8動的量子化
            # 重みをINT8、活性化関数をfloatで計算
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            return quantized_model
        elif bits == 16:
             # FP16への変換（GPUでの推論が高速化）
             return model.half()
        
        print(f"警告: {bits}ビットの量子化は現在サポートされていません。モデルは変更されません。")
        return model

# ----------------------------------------
# 2. リアルタイム性能監視システム（変更なし）
# ----------------------------------------

class RealtimePerformanceTracker:
    """リアルタイムパフォーマンストラッカー"""
    def __init__(self, monitoring_interval: float = 1.0):
        self.metrics_history = []
        self.current_metrics = {
            'inference_latency_ms': 0.0,
            'throughput_qps': 0.0,
            'spike_rate': 0.0
        }
    def start_monitoring(self): print("📊 リアルタイム監視開始")
    def stop_monitoring(self): print("⏹️ リアルタイム監視停止")
    def record_inference(self, latency_ms: float, output: torch.Tensor):
        self.current_metrics['inference_latency_ms'] = latency_ms
        self.current_metrics['throughput_qps'] = 1000.0 / latency_ms if latency_ms > 0 else 0
        if hasattr(output, 'mean'):
             self.current_metrics['spike_rate'] = output.mean().item()
        self.metrics_history.append(self.current_metrics.copy())
        if len(self.metrics_history) > 100: self.metrics_history.pop(0)
    def get_current_performance(self) -> Dict[str, float]: return self.current_metrics.copy()

# ----------------------------------------
# 3. 継続学習システム
# ----------------------------------------

class ContinualLearningEngine:
    """継続学習エンジン (知識蒸留を改善)"""
    def __init__(self, model: nn.Module, learning_rate: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        # 過去のモデルの状態を「教師」として保持
        self.teacher_model = copy.deepcopy(self.model).eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def online_learning_step(self, new_data: torch.Tensor, new_targets: torch.Tensor):
        """オンライン学習ステップ"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 新データでの損失
        outputs = self.model(new_data)
        ce_loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), new_targets.view(-1))
        
        # 知識蒸留: 過去の自分（教師モデル）の出力を再現するように学習
        # これにより、新しい知識を学びつつ、古い知識を忘れにくくなる
        with torch.no_grad():
            teacher_outputs = self.teacher_model(new_data)
        
        distillation_loss = F.kl_div(
            F.log_softmax(outputs / 2.0, dim=-1),       # 生徒の出力
            F.log_softmax(teacher_outputs / 2.0, dim=-1), # 教師の出力
            reduction='batchmean',
            log_target=True # 教師の出力もlog-softmax
        )
        
        # クロスエントロピー損失と知識蒸留損失を組み合わせて学習
        total_loss = ce_loss + 0.7 * distillation_loss
        total_loss.backward()
        self.optimizer.step()
        
        return {'total_loss': total_loss.item(), 'ce_loss': ce_loss.item(), 'distillation_loss': distillation_loss.item()}

# ----------------------------------------
# 4. 実用デプロイメントマネージャー（変更なし）
# ----------------------------------------

class SNNDeploymentManager:
    """SNN実用デプロイメントマネージャー"""
    def __init__(self): self.deployed_models = {}
    def deploy_model(self, model: nn.Module, name: str, profile: HardwareProfile, level: OptimizationLevel):
        print(f"🚀 モデル '{name}' をデプロイ中...")
        optimizer = DynamicOptimizer(model, profile)
        opt_model = optimizer.optimize_for_deployment(level)
        self.deployed_models[name] = {
            'model': opt_model, 'hardware_profile': profile,
            'performance_tracker': RealtimePerformanceTracker(),
            'continual_learner': ContinualLearningEngine(opt_model)
        }
        self.deployed_models[name]['performance_tracker'].start_monitoring()
        print(f"✅ デプロイメント '{name}' 完了")
    def inference(self, name: str, data: torch.Tensor) -> torch.Tensor:
        deployment = self.deployed_models[name]
        start_time = time.time()
        deployment['model'].eval()
        with torch.no_grad(): output = deployment['model'](data)
        latency = (time.time() - start_time) * 1000
        deployment['performance_tracker'].record_inference(latency, output)
        return output
    def online_adaptation(self, name: str, data: torch.Tensor, targets: torch.Tensor):
        loss = self.deployed_models[name]['continual_learner'].online_learning_step(data, targets)
        print(f"📚 オンライン学習完了: {loss}")
    def get_deployment_status(self, name: str) -> Dict[str, Any]:
        d = self.deployed_models[name]
        return {"status": "active", "hardware": d['hardware_profile'], "performance": d['performance_tracker'].get_current_performance()}
    def shutdown_deployment(self, name: str):
        self.deployed_models[name]['performance_tracker'].stop_monitoring()
        del self.deployed_models[name]
        print(f"🛑 デプロイメント '{name}' を終了しました")

# ----------------------------------------
# 5. 使用例
# ----------------------------------------

def main_deployment_example():
    """実用デプロイメントの例"""
    from snn_core import BreakthroughSNN # 例のためにインポート
    print("🌟 SNNの実用デプロイメント例を開始")
    
    # ダミーモデルとハードウェアプロファイル
    dummy_model = BreakthroughSNN(vocab_size=100, d_model=32, d_state=16, num_layers=1, time_steps=8)
    hardware = HardwareProfile(device_type="edge_gpu", memory_limit_gb=4.0, power_budget_w=15.0)
    
    manager = SNNDeploymentManager()
    deployment_name = "edge_ai_deployment"
    manager.deploy_model(dummy_model, deployment_name, hardware, OptimizationLevel.BALANCED)
    
    # 推論テスト
    print("\n📊 推論テスト実行中...")
    test_input = torch.randint(0, 100, (1, 10))
    for _ in range(5):
        _ = manager.inference(deployment_name, test_input)
        time.sleep(0.05)
    print(f"最新の性能: {manager.get_deployment_status(deployment_name)['performance']}")
    
    # 継続学習テスト
    print("\n🧠 継続学習テスト...")
    new_data = torch.randint(0, 100, (4, 10))
    new_targets = torch.randint(0, 100, (4, 10)) # 次トークン予測タスクなのでターゲットもシーケンス
    manager.online_adaptation(deployment_name, new_data, new_targets)
    
    manager.shutdown_deployment(deployment_name)
    print("\n✅ 実用デプロイメント例完了")

if __name__ == "__main__":
    main_deployment_example()
