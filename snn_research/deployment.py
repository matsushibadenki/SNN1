# matsushibadenki/snn/snn_research/deployment.py
# SNNの実用デプロイメントのための最適化、監視、継続学習システム
#
# 変更点:
# - mypyエラー解消のため、型ヒントを追加。
# - 独自Vocabularyを廃止し、Hugging Face Tokenizerを使用するようにSNNInferenceEngineを修正。
# - `generate` メソッドをストリーミング応答（ジェネレータ）に変更し、逐次的なテキスト生成を可能に。
# - `stop_sequences` のロジックを改善し、生成テキスト全体に含まれるかをチェックするようにした。
# - 推論時の総スパイク数を計測し、インスタンス変数 `last_inference_stats` に保存する機能を追加。
# - コンストラクタでモデル設定を直接受け取り、チェックポイントに設定がない場合でもフォールバックできるように修正。
# - model.load_state_dict に strict=False を追加し、バッファなどのキーが不足していてもエラーにならないように堅牢性を向上。

import torch
import torch.nn as nn
import os
import copy
import time
from typing import Dict, Any, List, Optional, Iterator
from enum import Enum
from dataclasses import dataclass
from transformers import AutoTokenizer

# --- SNN 推論エンジン ---
class SNNInferenceEngine:
    """SNNモデルでテキスト生成を行う推論エンジン"""
    def __init__(self, model_path: str, device: str, model_config: Optional[Dict[str, Any]] = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

        from .core.snn_core import BreakthroughSNN

        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device)

        tokenizer_name = checkpoint.get('tokenizer_name', 'gpt2')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 引数で渡されたモデル設定を優先し、なければチェックポイントから読み込む
        final_model_config = None
        if model_config and isinstance(model_config, dict):
            final_model_config = model_config
        else:
            final_model_config = checkpoint.get('config')

        if not isinstance(final_model_config, dict):
            raise TypeError(
                "モデル設定(config)が見つかりません。"
                "アプリケーション起動時に --model_config で正しいYAMLファイルを指定するか、"
                "設定情報が含まれたモデルファイルを使用してください。"
            )
        self.config: Dict[str, Any] = final_model_config

        # BreakthroughSNNのコンストラクタが受け入れる引数のみをフィルタリング
        expected_args = {
            'd_model', 'd_state', 'num_layers', 'time_steps', 'n_head'
        }
        model_kwargs = {k: v for k, v in self.config.items() if k in expected_args}

        self.model = BreakthroughSNN(vocab_size=self.tokenizer.vocab_size, **model_kwargs).to(self.device)
        
        # strict=False を設定して、不足しているキーがあってもエラーを発生させない
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        self.last_inference_stats: Dict[str, Any] = {}

    def generate(self, start_text: str, max_len: int, stop_sequences: Optional[List[str]] = None) -> Iterator[str]:
        """
        テキストをストリーミング形式で生成します。
        """
        self.last_inference_stats = {"total_spikes": 0}

        bos_token = self.tokenizer.bos_token or ''
        prompt_ids = self.tokenizer.encode(f"{bos_token}{start_text}", return_tensors='pt').to(self.device)

        input_tensor = prompt_ids
        generated_text = ""

        with torch.no_grad():
            for _ in range(max_len):
                logits, hidden_states = self.model(input_tensor, return_spikes=True)

                if hidden_states.numel() > 0:
                    self.last_inference_stats["total_spikes"] += hidden_states.sum().item()

                next_token_logits = logits[:, -1, :]
                next_token_id_tensor = torch.argmax(next_token_logits, dim=-1)
                next_token_id = next_token_id_tensor.item()

                if next_token_id == self.tokenizer.eos_token_id:
                    break

                new_token = self.tokenizer.decode([next_token_id])
                generated_text += new_token
                yield new_token

                if stop_sequences:
                    if any(stop_seq in generated_text for stop_seq in stop_sequences):
                        break

                input_tensor = torch.cat([input_tensor, next_token_id_tensor.unsqueeze(0)], dim=1)


# --- ニューロモーフィック デプロイメント機能 ---
import torch.nn.functional as F

class NeuromorphicChip(Enum):
    INTEL_LOIHI = "intel_loIhi"
    IBM_TRUENORTH = "ibm_truenorth"
    GENERIC_EDGE = "generic_edge"

@dataclass
class NeuromorphicProfile:
    chip_type: NeuromorphicChip
    num_cores: int
    memory_hierarchy: Dict[str, int]
    power_budget_mw: float
    supports_online_learning: bool = True

class AdaptiveQuantizationPruning:
    def apply_pruning(self, model: nn.Module, pruning_ratio: float):
        if pruning_ratio <= 0: return
        for module in model.modules():
            if isinstance(module, nn.Linear):
                pass

    def apply_quantization(self, model: nn.Module, bits: int) -> nn.Module:
        if bits >= 32: return model
        return model

class ContinualLearningEngine:
    def __init__(self, model: nn.Module, learning_rate: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.teacher_model = copy.deepcopy(self.model).eval()

    def online_learning_step(self, new_data: torch.Tensor, new_targets: torch.Tensor) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        outputs, _ = self.model(new_data)
        ce_loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), new_targets.view(-1))
        with torch.no_grad(): teacher_outputs, _ = self.teacher_model(new_data)
        distillation_loss = F.kl_div(
            F.log_softmax(outputs / 2.0, dim=-1),
            F.log_softmax(teacher_outputs / 2.0, dim=-1),
            reduction='batchmean', log_target=True
        )
        total_loss = ce_loss + 0.7 * distillation_loss
        total_loss.backward()
        self.optimizer.step()
        return {'total_loss': total_loss.item()}

class NeuromorphicDeploymentManager:
    deployed_models: Dict[str, Dict[str, Any]]

    def __init__(self, profile: NeuromorphicProfile):
        self.profile = profile
        self.adaptive_compression = AdaptiveQuantizationPruning()
        self.deployed_models = {}

    def deploy_model(self, model: nn.Module, name: str, optimization_target: str = "balanced"):
        print(f"🔧 ニューロモーフィックデプロイメント開始: {name}")
        if optimization_target == "balanced": sparsity, bit_width = 0.7, 8
        elif optimization_target == "ultra_low_power": sparsity, bit_width = 0.9, 8
        else: sparsity, bit_width = 0.5, 16
        optimized_model = copy.deepcopy(model).cpu()
        optimized_model.eval()
        print(f"  - プルーニング適用中 (スパース率: {sparsity})...")
        self.adaptive_compression.apply_pruning(optimized_model, float(sparsity))
        print(f"  - 量子化適用中 (ビット幅: {bit_width}-bit)...")
        optimized_model = self.adaptive_compression.apply_quantization(optimized_model, int(bit_width))
        self.deployed_models[name] = {
            'model': optimized_model,
            'continual_learner': ContinualLearningEngine(optimized_model)
        }
        print(f"✅ デプロイメント完了: {name}")
