# /path/to/your/project/main.py
# SNNモデルの学習と推論を実行するためのメインスクリプト
#
# 元ファイル:
# - train_text_snn.py (学習部分)
# - inference.py (推論部分)
# - snn_breakthrough.py (実行ブロック)
# を統合し、snn_core.pyのコンポーネントを使用するように変更。
# 学習タスクを「次トークン予測」に修正し、より高度なモデルに対応。

import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import itertools
from typing import List, Tuple
import os
import random

# snn_coreから主要コンポーネントをインポート
from snn_core import BreakthroughSNN, BreakthroughTrainer

# ----------------------------------------
# 1. データ準備と語彙の構築
# ----------------------------------------

class Vocabulary:
    """テキストとIDを相互変換するための語彙クラス"""
    def __init__(self, all_texts: List[str]):
        # 予約トークン
        self.special_tokens = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.word2idx = self.special_tokens.copy()
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        if all_texts:
            self._build_vocab(all_texts)

    def _build_vocab(self, all_texts: List[str]):
        all_words = list(itertools.chain.from_iterable(txt.lower().split() for txt in all_texts))
        word_counts = Counter(all_words)
        for word, _ in word_counts.items():
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def encode(self, text: str, add_start_end: bool = True) -> List[int]:
        tokens = [self.word2idx.get(word.lower(), self.special_tokens["<UNK>"]) for word in text.split()]
        if add_start_end:
            return [self.special_tokens["<START>"]] + tokens + [self.special_tokens["<END>"]]
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        return " ".join([self.idx2word.get(idx, "<UNK>") for idx in token_ids])

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)
    
    @property
    def pad_id(self) -> int:
        return self.special_tokens["<PAD>"]

class NextTokenPredictionDataset(Dataset):
    """
    次トークン予測のためのデータセット。
    文章を受け取り、(入力シーケンス, ターゲットシーケンス)のペアを生成します。
    例: "i love this movie" -> input="<START> i love this", target="i love this <END>"
    """
    def __init__(self, data: List[str], vocab: Vocabulary, max_len: int = 32):
        self.vocab = vocab
        self.max_len = max_len
        self.encoded_data = [self.vocab.encode(text) for text in data]
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        encoded = self.encoded_data[idx]
        
        # 入力とターゲットを作成
        input_seq = encoded[:-1]
        target_seq = encoded[1:]
        
        # パディング
        input_len = len(input_seq)
        pad_len = self.max_len - input_len
        
        padded_input = input_seq[:self.max_len] + [self.vocab.pad_id] * max(0, pad_len)
        padded_target = target_seq[:self.max_len] + [self.vocab.pad_id] * max(0, pad_len)
        
        return torch.tensor(padded_input), torch.tensor(padded_target, dtype=torch.long)

# ----------------------------------------
# 2. 推論エンジン
# ----------------------------------------

class SNNInferenceEngine:
    """SNNモデルでテキスト生成や分析を行う推論エンジン"""
    def __init__(self, model_path: str, device: str = "cpu"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
        
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.vocab = checkpoint['vocab']
        config = checkpoint['config']
        
        self.model = BreakthroughSNN(vocab_size=self.vocab.vocab_size, **config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def generate(self, start_text: str, max_len: int = 20) -> str:
        """与えられたテキストに続く文章を生成する"""
        print(f"\n生成開始: '{start_text}'")
        
        # 初期テキストをエンコード
        input_ids = self.vocab.encode(start_text, add_start_end=False)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        generated_ids = list(input_ids)
        
        with torch.no_grad():
            for _ in range(max_len):
                # モデルに現在のシーケンスを入力
                logits = self.model(input_tensor)
                
                # 最後のトークンの予測確率から次のトークンをサンプリング
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                
                # 生成が終了トークンに達したら終了
                if next_token_id == self.vocab.special_tokens["<END>"]:
                    break
                
                generated_ids.append(next_token_id)
                # 次の入力として、生成されたトークンを追加
                input_tensor = torch.tensor([generated_ids], device=self.device)
        
        return self.vocab.decode(generated_ids)

# ----------------------------------------
# 3. 実行ブロック
# ----------------------------------------

def train():
    """モデルの学習を実行"""
    print("🚀 革新的SNNシステムの訓練開始 (次トークン予測タスク)")
    
    # サンプルデータ（多様な文構造）
    TRAIN_DATA = [
        "this movie was terrible", "i absolutely loved it",
        "a complete disappointment", "one of the best films ever made",
        "the plot was confusing and slow", "a truly heartwarming story",
        "i would not recommend this to anyone", "an unforgettable experience for sure",
        "what a complete mess", "simply fantastic from start to finish"
    ]
    
    vocab = Vocabulary(TRAIN_DATA)
    
    # モデル設定
    config = {'d_model': 64, 'd_state': 32, 'num_layers': 2, 'time_steps': 16}
    model = BreakthroughSNN(vocab_size=vocab.vocab_size, **config)
    trainer = BreakthroughTrainer(model)
    
    dataset = NextTokenPredictionDataset(TRAIN_DATA, vocab)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 学習ループ
    num_epochs = 100
    for epoch in range(num_epochs):
        for input_ids, target_ids in dataloader:
            metrics = trainer.train_step(input_ids, target_ids)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: {metrics}")
            
    # モデル保存
    model_path = "breakthrough_snn_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'config': config
    }, model_path)
    print(f"\n✅ 学習済みモデルを '{model_path}' に保存しました。")

def inference():
    """学習済みモデルで推論（文章生成）を実行"""
    MODEL_FILE_PATH = "breakthrough_snn_model.pth"
    
    try:
        engine = SNNInferenceEngine(model_path=MODEL_FILE_PATH)
        test_sentences = [
            "this movie was",
            "i loved",
            "the story",
        ]
        
        for sentence in test_sentences:
            generated_text = engine.generate(sentence)
            print(f"生成結果: {generated_text}")

    except FileNotFoundError as e:
        print(e)
        print("エラー: 学習済みモデルファイルが必要です。先に 'train' モードで実行してください。")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) > 1 and sys.argv[1] == "inference":
        inference()
    else:
        print("使い方: python main.py [train|inference]")
        # デフォルトで学習を実行
        print("\n--- デフォルトの学習モードを実行します ---")
        train()
