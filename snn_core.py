# /path/to/your/project/snn_core.py
# SNNモデルの定義、次世代ニューロン、学習システムなど、中核となるロジックを集約したライブラリ
#
# 元ファイル:
# - snn_breakthrough.py
# - advanced_snn_chat.py
# - train_text_snn.py
# のうち、最も先進的な機能を統合・整理し、さらに洗練させたバージョン

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, functional
from typing import Tuple

# ----------------------------------------
# 1. 高度なスパイクエンコーダー
# ----------------------------------------

class TemporalEncoder(nn.Module):
    """
    埋め込みベクトルを時間的なスパイクパターンに変換するエンコーダー。
    ベクトルの各要素値をスパイクの発火タイミングにマッピングします。
    """
    def __init__(self, time_steps: int):
        super().__init__()
        self.time_steps = time_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 埋め込みベクトル。形状は (batch_size, seq_len, d_model)。
        Returns:
            torch.Tensor: スパイク列。形状は (batch_size, time_steps, seq_len, d_model)。
        """
        # ベクトル値をシグモイド関数で 0-1 の範囲に正規化し、発火タイミングを決定
        spike_times = (torch.sigmoid(x) * (self.time_steps - 1)).long()
        
        # (batch_size, seq_len, d_model) -> (batch_size, 1, seq_len, d_model)
        spike_times = spike_times.unsqueeze(1)

        # 出力用のスパイク列テンソルを初期化
        spikes = torch.zeros(x.shape[0], self.time_steps, x.shape[1], x.shape[2], device=x.device)
        
        # scatter_ を使って、計算された発火タイミングの位置にスパイク (1.0) を配置
        # これにより、各ニューロンが指定されたタイムステップで一度だけ発火するパターンが生成される
        spikes.scatter_(1, spike_times, 1.0)
        
        return spikes

# ----------------------------------------
# 2. Spiking State Space Model (革新的アーキテクチャ)
# ----------------------------------------

class SpikingSSMLayer(nn.Module):
    """
    スパイキング状態空間モデル。線形計算量で長期依存関係を効率的に処理します。
    LIFニューロンを組み込み、状態遷移をスパイクベースで行います。
    """
    def __init__(self, d_model: int, d_state: int = 64, dt: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt = dt

        # 状態遷移行列 A, B, C とバイアス項 D を定義
        # A: 状態h_tを次の状態h_{t+1}にどう遷移させるかを決定
        # B: 入力x_tが状態h_tにどう影響するかを決定
        # C: 状態h_tが出力y_tにどう影響するかを決定
        # D: 入力x_tが出力y_tに直接どう影響するかを決定 (スキップ接続)
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        
        # 状態更新と出力計算のためのLIFニューロン
        self.state_lif = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        self.output_lif = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        
        # 内部状態を保持するバッファ
        self.register_buffer('h_state', torch.zeros(1, 1, d_state))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 入力スパイク列。形状は (batch_size, time_steps, seq_len, d_model)。
        Returns:
            torch.Tensor: 出力スパイク列。形状は (batch_size, time_steps, seq_len, d_model)。
        """
        batch_size, time_steps, seq_len, _ = x.shape
        
        # バッチサイズとシーケンス長に合わせて内部状態の形状を調整
        if self.h_state.shape[0] != batch_size or self.h_state.shape[1] != seq_len:
            self.h_state = torch.zeros(batch_size, seq_len, self.d_state, device=x.device)

        outputs = []
        for t in range(time_steps):
            x_t = x[:, t, :, :]  # (batch_size, seq_len, d_model)
            
            # 状態更新: h_t = A * h_{t-1} + B * x_t
            # h_state: (batch_size, seq_len, d_state)
            # x_t: (batch_size, seq_len, d_model)
            state_update = F.linear(self.h_state, self.A) + F.linear(x_t, self.B.T)
            self.h_state = self.state_lif(state_update)
            
            # 出力計算: y_t = C * h_t + D * x_t
            output_update = F.linear(self.h_state, self.C.T) + F.linear(x_t, self.D)
            out_spike = self.output_lif(output_update)
            
            outputs.append(out_spike)

        return torch.stack(outputs, dim=1)


# ----------------------------------------
# 3. 統合された次世代SNNアーキテクチャ
# ----------------------------------------

class BreakthroughSNN(nn.Module):
    """
    Spiking-SSMを中核に据えた、次世代のSNNアーキテクチャ。
    ANNを超える性能と効率を目指します。
    """
    def __init__(self, vocab_size: int, d_model: int = 256, d_state: int = 64, num_layers: int = 4, time_steps: int = 20):
        super().__init__()
        self.time_steps = time_steps
        
        # 1. 単語IDをベクトルに変換する埋め込み層
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. 埋め込みベクトルをスパイク列に変換するエンコーダー
        self.spike_encoder = TemporalEncoder(time_steps)
        
        # 3. Spiking-SSM層を複数重ねて、文脈の深い理解を可能にする
        self.layers = nn.ModuleList([SpikingSSMLayer(d_model, d_state) for _ in range(num_layers)])
        
        # 4. SNNからの出力（スパイク）を次の単語の予測確率（ロジット）に変換
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        # (batch, seq_len) -> (batch, seq_len, d_model)
        token_emb = self.token_embedding(input_ids)
        
        # (batch, seq_len, d_model) -> (batch, time, seq_len, d_model)
        spike_sequence = self.spike_encoder(token_emb)
        
        hidden_states = spike_sequence
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # 時間次元でスパイクを平均化し、情報を集約
        # (batch, time, seq_len, d_model) -> (batch, seq_len, d_model)
        time_integrated = hidden_states.mean(dim=1)
        
        # 次の単語の確率分布を計算
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        logits = self.output_projection(time_integrated)
        
        return (logits, hidden_states) if return_spikes else logits

# ----------------------------------------
# 4. 統合トレーニングシステム
# ----------------------------------------

class CombinedLoss(nn.Module):
    """
    クロスエントロピー損失とスパイク発火率の正則化を組み合わせた損失関数。
    モデルの精度向上と、エネルギー効率（低発火率）の維持を両立させます。
    """
    def __init__(self, ce_weight: float = 1.0, spike_reg_weight: float = 0.01, target_spike_rate: float = 0.02):
        super().__init__()
        self.ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.weights = {'ce': ce_weight, 'spike_reg': spike_reg_weight}
        self.target_spike_rate = target_spike_rate

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, spikes: torch.Tensor) -> dict:
        # 1. クロスエントロピー損失（精度向上）
        # logits: (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
        # targets: (batch, seq_len) -> (batch * seq_len)
        ce_loss = self.ce_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # 2. スパイク正則化損失（省エネ化）
        spike_rate = spikes.mean()
        # 発火率が目標値から離れるほどペナルティを与える
        spike_reg_loss = F.mse_loss(spike_rate, torch.tensor(self.target_spike_rate, device=spike_rate.device))
        
        # 3. 2つの損失を重み付けして合計
        total_loss = self.weights['ce'] * ce_loss + self.weights['spike_reg'] * spike_reg_loss
        
        return {
            'total': total_loss,
            'ce_loss': ce_loss,
            'spike_reg_loss': spike_reg_loss,
            'spike_rate': spike_rate
        }

class BreakthroughTrainer:
    """革新的SNNの訓練システム"""
    def __init__(self, model: BreakthroughSNN, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10000)
        self.criterion = CombinedLoss()

    def train_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> dict:
        self.model.train()
        input_ids, target_ids = input_ids.to(self.device), target_ids.to(self.device)
        
        self.optimizer.zero_grad()
        logits, spike_data = self.model(input_ids, return_spikes=True)
        
        loss_dict = self.criterion(logits, target_ids, spike_data)
        total_loss = loss_dict['total']
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
