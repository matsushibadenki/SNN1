# /path/to/your/project/main.py
# SNNモデルの学習と推論を実行するためのメインスクリプト
#
# 元ファイル:
# - train_text_snn.py (学習部分)
# - inference.py (推論部分)
# - snn_breakthrough.py (実行ブロック)
# を統合し、snn_core.pyのコンポーネントを使用するように変更

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import itertools
from typing import List, Tuple, Dict
import os

# snn_coreから主要コンポーネントをインポート
from snn_core import BreakthroughSNN, BreakthroughTrainer

# ----------------------------------------
# 1. データ準備と語彙の構築
# ----------------------------------------

class Vocabulary:
    """テキストとIDを相互変換するための語彙クラス"""
    def __init__(self, all_texts: List[Tuple[str, int]]):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        if all_texts:
            self._build_vocab(all_texts)

    def _build_vocab(self, all_texts: List[Tuple[str, int]]):
        all_words = list(itertools.chain.from_iterable(txt.lower().split() for txt, _ in all_texts))
        for word in Counter(all_words).keys():
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
    
    def encode(self, text: str) -> List[int]:
        return [self.word2idx.get(word.lower(), self.word2idx["<UNK>"]) for word in text.split()]
    
    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

class TextDataset(Dataset):
    """テキストデータセット"""
    def __init__(self, data: List[Tuple[str, int]], vocab: Vocabulary, max_len: int = 32):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoded_text = self.vocab.encode(text)[:self.max_len]
        padded_text = encoded_text + [self.vocab.word2idx["<PAD>"]] * (self.max_len - len(encoded_text))
        return torch.tensor(padded_text), torch.tensor(label, dtype=torch.long)

# ----------------------------------------
# 2. 推論エンジン
# ----------------------------------------

class SNNInferenceEngine:
    def __init__(self, model_path: str, device: str = "cpu"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
        
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.vocab = checkpoint['vocab']
        config = checkpoint['config']
        
        self.model = BreakthroughSNN(
            vocab_size=self.vocab.vocab_size,
            d_model=config['d_model'],
            num_layers=config['num_layers'],
            time_steps=config['time_steps']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.class_labels = {0: "ネガティブ", 1: "ポジティブ"}

    def predict(self, text: str) -> str:
        print(f"\n入力文章: '{text}'")
        encoded_text = self.vocab.encode(text)[:32]
        padded_text = encoded_text + [self.vocab.word2idx["<PAD>"]] * (32 - len(encoded_text))
        input_tensor = torch.tensor([padded_text]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        _, predicted_idx = torch.max(outputs.data, 1)
        return self.class_labels[predicted_idx.item()]

# ----------------------------------------
# 3. 実行ブロック
# ----------------------------------------

def train():
    """モデルの学習を実行"""
    print("🚀 革新的SNNシステムの訓練開始")
    
    # サンプルデータ
    TRAIN_DATA = [
        ("this movie was terrible", 0), ("i absolutely loved it", 1),
        ("a complete disappointment", 0), ("one of the best films", 1),
        ("the plot was confusing", 0), ("a heartwarming story", 1),
        ("i would not recommend this", 0), ("an unforgettable experience", 1),
        ("what a mess", 0), ("simply fantastic", 1)
    ]
    
    # 語彙構築
    vocab = Vocabulary(TRAIN_DATA)
    
    # モデル設定
    config = {'d_model': 64, 'num_layers': 2, 'time_steps': 10}
    model = BreakthroughSNN(vocab_size=vocab.vocab_size, **config)
    trainer = BreakthroughTrainer(model)
    
    # データローダー
    dataset = TextDataset(TRAIN_DATA, vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 学習ループ
    for epoch in range(20):
        for input_ids, labels in dataloader:
            # ラベルをダミーのシーケンスに変換（損失計算のため）
            target_ids = labels.unsqueeze(1).repeat(1, 32)
            metrics = trainer.train_step(input_ids, target_ids)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: {metrics}")
            
    # モデル保存
    model_path = "breakthrough_snn_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'config': config
    }, model_path)
    print(f"\n✅ 学習済みモデルを '{model_path}' に保存しました。")

def inference():
    """学習済みモデルで推論を実行"""
    MODEL_FILE_PATH = "breakthrough_snn_model.pth"
    
    try:
        engine = SNNInferenceEngine(model_path=MODEL_FILE_PATH)
        test_sentences = [
            "an unforgettable experience truly a masterpiece",
            "the plot was confusing and the characters were boring",
            "i will watch it again",
            "what a mess"
        ]
        
        for sentence in test_sentences:
            prediction = engine.predict(sentence)
            print(f"推論結果: {prediction}")

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