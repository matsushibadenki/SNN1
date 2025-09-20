# matsushibadenki/snn/snn_research/deployment.py
# SNNの実用デプロイメントのための最適化、監視、継続学習システム
# 
# 変更点:
# - main.pyからSNNInferenceEngineをこのモジュールに移動。
# - 循環参照を避けるため、snn_coreのインポートをクラスメソッド内で行うように変更。

import torch
import torch.nn as nn
import os
from typing import Dict, Any

# --- (NeuromorphicChip, NeuromorphicProfile, ... NeuromorphicDeploymentManager のコードは変更なし) ---
# ... (既存のコードをここにペースト) ...

class SNNInferenceEngine:
    """SNNモデルでテキスト生成や分析を行う推論エンジン"""
    def __init__(self, model_path: str, device: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
        
        # 循環参照を避けるための遅延インポート
        from .core.snn_core import BreakthroughSNN
        from .data.datasets import Vocabulary

        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.vocab: Vocabulary = checkpoint['vocab']
        config = checkpoint['config']
        
        self.model = BreakthroughSNN(vocab_size=self.vocab.vocab_size, **config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def generate(self, start_text: str, max_len: int) -> str:
        input_ids = self.vocab.encode(start_text, add_start_end=True)[:-1]
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        generated_ids = list(input_ids)
        
        with torch.no_grad():
            for _ in range(max_len):
                logits, _ = self.model(input_tensor, return_spikes=True)
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                
                if next_token_id == self.vocab.special_tokens["<END>"]:
                    break
                
                generated_ids.append(next_token_id)
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=self.device)], dim=1)
        
        return self.vocab.decode(generated_ids)