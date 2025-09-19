# /path/to/your/project/advanced_snn_chat.py
# 高性能SNNベースチャットシステム - ANN系AIに対抗するための改良版
#
# 主な改善点:
# 1. SpikeGPT/SRWKVアーキテクチャの実装
# 2. 複数のスパイクエンコーディング手法の組み合わせ
# 3. 多層化とアテンション機構の導入
# 4. 代理勾配法の最適化

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from spikingjelly.activation_based import neuron, surrogate, functional
from collections import Counter
import itertools
import math
from typing import List, Tuple, Optional

# ----------------------------------------
# 1. 改良版語彙管理システム
# ----------------------------------------

class AdvancedVocabulary:
    """BPE（Byte Pair Encoding）対応の高性能語彙システム"""
    def __init__(self, corpus_texts: List[str], vocab_size: int = 10000):
        self.special_tokens = {
            "<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3, "<MASK>": 4
        }
        self.word2idx = self.special_tokens.copy()
        self.idx2word = {v: k for k, v in self.special_tokens.items()}
        self.vocab_size_target = vocab_size
        
        if corpus_texts:
            self._build_bpe_vocab(corpus_texts)

    def _build_bpe_vocab(self, texts: List[str]):
        """BPE（簡易版）を使用した語彙構築"""
        # 文字レベルでの初期化
        all_chars = set(''.join(texts))
        for char in sorted(all_chars):
            if char not in self.word2idx:
                self.word2idx[char] = len(self.word2idx)
        
        # 単語頻度の計算
        word_freq = Counter()
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freq[' '.join(word) + ' </w>'] += 1
        
        # BPE処理（簡略版）
        vocab = word_freq.copy()
        num_merges = min(1000, self.vocab_size_target - len(self.word2idx))
        
        for i in range(num_merges):
            pairs = self._get_pairs(vocab)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)
            
            if ' '.join(best_pair) not in self.word2idx:
                self.word2idx[' '.join(best_pair)] = len(self.word2idx)
        
        # idx2wordの更新
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def _get_pairs(self, vocab):
        """隣接するトークンペアの頻度を計算"""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair, v_in):
        """指定されたペアをマージ"""
        v_out = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in v_in:
            new_word = word.replace(bigram, replacement)
            v_out[new_word] = v_in[word]
        return v_out

    def encode(self, text: str) -> List[int]:
        """テキストをトークンIDのリストに変換"""
        tokens = []
        for word in text.lower().split():
            word_tokens = self._encode_word(word)
            tokens.extend(word_tokens)
        return tokens

    def _encode_word(self, word: str) -> List[int]:
        """単語を分割してトークンIDに変換"""
        if word in self.word2idx:
            return [self.word2idx[word]]
        
        # 文字レベルにフォールバック
        char_tokens = []
        for char in word:
            char_tokens.append(self.word2idx.get(char, self.word2idx["<UNK>"]))
        return char_tokens

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

# ----------------------------------------
# 2. 高性能スパイクエンコーディング
# ----------------------------------------

class MultiModalSpikeEncoder:
    """複数のエンコーディング手法を組み合わせた高性能スパイクエンコーダ"""
    
    def __init__(self, embed_dim: int, time_steps: int):
        self.embed_dim = embed_dim
        self.time_steps = time_steps
        
    def rate_coding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """レートコーディング（改良版）"""
        # 正規化とノイズ追加でロバスト性向上
        normalized = torch.sigmoid(embeddings) * 0.8 + 0.1  # [0.1, 0.9]の範囲
        noise = torch.randn_like(normalized) * 0.05
        firing_rates = torch.clamp(normalized + noise, 0, 1)
        
        spikes = torch.rand(self.time_steps, *firing_rates.shape) < firing_rates
        return spikes.float()
    
    def temporal_coding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """テンポラルコーディング - 値によってスパイクタイミングを制御"""
        spikes = torch.zeros(self.time_steps, *embeddings.shape)
        
        # 埋め込み値を[0, time_steps-1]の範囲にマッピング
        normalized = torch.sigmoid(embeddings)
        spike_times = (normalized * (self.time_steps - 1)).long()
        
        for i, timing in enumerate(spike_times.flatten()):
            if timing < self.time_steps:
                flat_idx = i % embeddings.numel()
                batch_idx = flat_idx // embeddings.shape[-1]
                feat_idx = flat_idx % embeddings.shape[-1]
                spikes[timing, batch_idx, feat_idx] = 1.0
                
        return spikes
    
    def phase_coding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """位相コーディング - 正弦波の位相でエンコード"""
        phases = torch.sigmoid(embeddings) * 2 * math.pi
        time_points = torch.linspace(0, 2*math.pi, self.time_steps).unsqueeze(-1).unsqueeze(-1)
        
        # 正弦波生成
        waves = torch.sin(time_points + phases.unsqueeze(0))
        spikes = (waves > 0.5).float()
        
        return spikes
    
    def encode(self, embeddings: torch.Tensor, method: str = "hybrid") -> torch.Tensor:
        """指定された手法でエンコーディング実行"""
        if method == "rate":
            return self.rate_coding(embeddings)
        elif method == "temporal":
            return self.temporal_coding(embeddings)
        elif method == "phase":
            return self.phase_coding(embeddings)
        elif method == "hybrid":
            # 複数手法の組み合わせ
            rate_spikes = self.rate_coding(embeddings)
            temporal_spikes = self.temporal_coding(embeddings)
            
            # 重み付き結合
            hybrid_spikes = 0.7 * rate_spikes + 0.3 * temporal_spikes
            return (hybrid_spikes > 0.5).float()
        else:
            raise ValueError(f"不明なエンコーディング手法: {method}")

# ----------------------------------------
# 3. SRWKV（Spiking RWKV）アーキテクチャの実装
# ----------------------------------------

class SpikingRWKVBlock(nn.Module):
    """SpikeGPTで使用されるSRWKVブロック"""
    
    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        
        # RWKV key components
        self.receptance = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Spiking components
        self.lif_r = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        self.lif_k = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        self.lif_v = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        
        # Feed-forward network (spiking)
        self.ffn = SpikingFFN(embed_dim, ffn_dim)
        
        # Layer normalization (adapted for spikes)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, time_steps, embed_dim)
        batch_size, time_steps, embed_dim = x.shape
        
        # RWKV attention mechanism with spikes
        residual = x
        h = torch.zeros(batch_size, embed_dim, device=x.device)
        outputs = []
        
        for t in range(time_steps):
            x_t = x[:, t, :]
            
            # RWKV computation
            r_t = self.lif_r(self.receptance(x_t))
            k_t = self.lif_k(self.key(x_t))
            v_t = self.lif_v(self.value(x_t))
            
            # Update hidden state
            h = h * 0.9 + k_t * v_t  # Simplified RWKV update
            y_t = r_t * h
            
            outputs.append(self.output(y_t))
        
        attn_out = torch.stack(outputs, dim=1)
        
        # Residual connection and normalization
        x = self.norm1(residual + attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class SpikingFFN(nn.Module):
    """スパイキング版フィードフォワードネットワーク"""
    
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, time_steps, embed_dim)
        batch_size, time_steps, embed_dim = x.shape
        outputs = []
        
        for t in range(time_steps):
            x_t = x[:, t, :]
            
            # First layer
            y = self.fc1(x_t)
            y = self.lif1(y)
            y = self.dropout(y)
            
            # Second layer
            y = self.fc2(y)
            y = self.lif2(y)
            
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)

# ----------------------------------------
# 4. 高性能SNNチャットモデル
# ----------------------------------------

class AdvancedSpikingChatModel(nn.Module):
    """SpikeGPT/SRWKVベースの高性能チャットモデル"""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_layers: int = 6,
        ffn_dim: int = 1024,
        max_seq_len: int = 512,
        time_steps: int = 50
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.time_steps = time_steps
        
        # 埋め込み層
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(max_seq_len, embed_dim))
        
        # スパイクエンコーダ
        self.spike_encoder = MultiModalSpikeEncoder(embed_dim, time_steps)
        
        # SRWKV transformer layers
        self.layers = nn.ModuleList([
            SpikingRWKVBlock(embed_dim, ffn_dim)
            for _ in range(num_layers)
        ])
        
        # 出力層
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        self.final_lif = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Token + Position embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        embeddings = token_emb + pos_emb
        
        # スパイクエンコーディング
        spike_inputs = self.spike_encoder.encode(embeddings, method="hybrid")
        # spike_inputs shape: (time_steps, batch, seq_len, embed_dim)
        spike_inputs = spike_inputs.permute(1, 0, 2, 3)  # (batch, time_steps, seq_len, embed_dim)
        
        # Transformer layers
        hidden_states = spike_inputs.view(batch_size, self.time_steps, -1)  # Flatten sequence
        
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # 出力の生成
        output_spikes = torch.zeros(batch_size, seq_len, self.token_embedding.num_embeddings)
        
        for t in range(self.time_steps):
            h_t = hidden_states[:, t, :].view(batch_size, seq_len, self.embed_dim)
            logits_t = self.output_projection(h_t)
            output_spikes += self.final_lif(logits_t)
        
        return output_spikes

# ----------------------------------------
# 5. 高性能データローダーとトレーナー
# ----------------------------------------

class ChatDataset(Dataset):
    """対話データセット"""
    
    def __init__(self, conversations: List[Tuple[str, str]], vocab: AdvancedVocabulary, max_len: int = 256):
        self.conversations = conversations
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        input_text, target_text = self.conversations[idx]
        
        # エンコーディング
        input_ids = self.vocab.encode(input_text)[:self.max_len-1]
        target_ids = self.vocab.encode(target_text)[:self.max_len-1]
        
        # パディング
        input_ids += [self.vocab.word2idx["<PAD>"]] * (self.max_len - len(input_ids))
        target_ids += [self.vocab.word2idx["<PAD>"]] * (self.max_len - len(target_ids))
        
        return torch.tensor(input_ids[:self.max_len]), torch.tensor(target_ids[:self.max_len])

class AdvancedSNNTrainer:
    """高性能SNNトレーナー"""
    
    def __init__(self, model, vocab, device="cuda"):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        
        # 最適化設定
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        # スケジューラー
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2
        )
        
        # 損失関数（ラベルスムージング付き）
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(input_ids)
            
            # 損失計算
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)),
                target_ids.view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches

# ----------------------------------------
# 6. 実行例とサンプルデータ
# ----------------------------------------

def create_sample_conversations():
    """サンプル対話データの作成"""
    return [
        ("hello how are you", "i am doing well thank you"),
        ("what is the weather like", "it is sunny and warm today"),
        ("can you help me with programming", "of course i would be happy to help"),
        ("what is machine learning", "machine learning is a subset of artificial intelligence"),
        ("how do neural networks work", "neural networks process information through interconnected nodes"),
        ("explain deep learning", "deep learning uses multiple layers to learn complex patterns"),
        ("what are spiking neural networks", "spiking neural networks process information using discrete spikes"),
        ("why are snns energy efficient", "snns only compute when spikes occur making them very efficient"),
        ("what is the future of ai", "ai will continue advancing with better algorithms and hardware"),
        ("how can ai help society", "ai can improve healthcare education and scientific research"),
    ]

def main():
    """メイン実行関数"""
    print("=== 高性能SNNチャットシステムの訓練開始 ===")
    
    # サンプルデータの準備
    conversations = create_sample_conversations()
    all_texts = [text for conv in conversations for text in conv]
    
    # 語彙構築
    vocab = AdvancedVocabulary(all_texts, vocab_size=5000)
    print(f"語彙サイズ: {vocab.vocab_size}")
    
    # モデル初期化
    model = AdvancedSpikingChatModel(
        vocab_size=vocab.vocab_size,
        embed_dim=256,
        num_layers=4,
        ffn_dim=1024,
        time_steps=30
    )
    
    print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # データセットとローダー
    dataset = ChatDataset(conversations, vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # トレーナー初期化
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = AdvancedSNNTrainer(model, vocab, device)
    
    # 訓練実行
    num_epochs = 20
    for epoch in range(num_epochs):
        avg_loss = trainer.train_epoch(dataloader, epoch)
        print(f"Epoch {epoch} 完了 - 平均損失: {avg_loss:.4f}")
    
    # モデル保存
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'config': {
            'vocab_size': vocab.vocab_size,
            'embed_dim': 256,
            'num_layers': 4,
            'ffn_dim': 1024,
            'time_steps': 30
        }
    }, "advanced_snn_chat_model.pth")
    
    print("✅ 高性能SNNチャットモデルの訓練完了!")

if __name__ == "__main__":
    main()