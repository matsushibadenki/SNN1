# matsushibadenki/snn/benchmark/ann_baseline.py
#
# SNNモデルとの性能比較を行うためのANNベースラインモデル
#
# 変更点:
# - __init__ に num_classes 引数を追加し、分類先のクラス数を可変にした。
#   これにより、SST-2 (2クラス分類) 以外のタスクにも対応可能になった。

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ANNBaselineModel(nn.Module):
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    """
    シンプルなTransformerベースのテキスト分類モデル。
    BreakthroughSNNとの比較用。
    """
    def __init__(self, vocab_size: int, d_model: int, nhead: int, d_hid: int, nlayers: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformerエンコーダ層を定義
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        # 分類ヘッド
        self.classifier = nn.Linear(d_model, num_classes)

        self.init_weights()
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): 入力シーケンス (batch_size, seq_len)
            src_padding_mask (torch.Tensor): パディングマスク (batch_size, seq_len)

        Returns:
            torch.Tensor: 分類ロジット (batch_size, num_classes)
        """
        embedded = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        
        # Transformerエンコーダに入力
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=src_padding_mask)
        
        # パディングを考慮した平均プーリング
        mask = ~src_padding_mask.unsqueeze(-1).expand_as(encoded)
        masked_encoded = encoded * mask.float()
        pooled = masked_encoded.sum(dim=1) / mask.float().sum(dim=1)
        
        logits = self.classifier(pooled)
        return logits
