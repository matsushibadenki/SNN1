# /path/to/your/project/main.py
# SNNモデルの学習と推論を実行するためのメインスクリプト
#
# 元ファイル:
# - train_text_snn.py (学習部分)
# - inference.py (推論部分)
# - snn_breakthrough.py (実行ブロック)
# を統合し、snn_core.pyのコンポーネントを使用するように変更。
# 学習タスクを「次トークン予測」に修正し、より高度なモデルに対応。
#
# 改善点:
# - argparseを導入し、コマンドラインから外部データファイル(JSON/TXT)を読み込めるように修正。
# - 汎用化されたBreakthroughTrainerに対応。

import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import itertools
from typing import List, Tuple
import os
import random
import argparse
import json

# snn_coreから主要コンポーネントをインポート
from snn_core import BreakthroughSNN, BreakthroughTrainer, CombinedLoss

# ----------------------------------------
# 1. データ準備と語彙の構築
# ----------------------------------------

def load_data_from_file(file_path: str, json_key: str = None) -> List[str]:
    """
    外部のJSONまたはTXTファイルからテキストデータを読み込みます。

    Args:
        file_path (str): データファイルのパス。
        json_key (str, optional): JSONファイルの場合、テキストリストが格納されているキー。

    Returns:
        List[str]: テキストデータのリスト。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"データファイルが見つかりません: {file_path}")

    _, ext = os.path.splitext(file_path)
    
    if ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if json_key:
            if json_key not in data:
                raise KeyError(f"指定されたキー '{json_key}' がJSONファイル内に見つかりません。")
            texts = data[json_key]
        else:
            # キーが指定されない場合、JSONデータ自体がリストであることを期待
            texts = data
        
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError("JSONから抽出されたデータは、文字列のリストである必要があります。")
        return texts

    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            # 空行を除外して読み込む
            return [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"サポートされていないファイル形式です: {ext} (.json または .txt を使用してください)")

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

def train(args):
    """モデルの学習を実行"""
    print("🚀 革新的SNNシステムの訓練開始 (次トークン予測タスク)")
    
    try:
        train_data = load_data_from_file(args.data_path, args.json_key)
        print(f"✅ {args.data_path} から {len(train_data)} 件のデータを読み込みました。")
    except (FileNotFoundError, KeyError, TypeError, ValueError) as e:
        print(f"❌ エラー: データファイルの読み込みに失敗しました。\n詳細: {e}")
        return

    vocab = Vocabulary(train_data)
    print(f"📖 語彙を構築しました。語彙数: {vocab.vocab_size}")
    
    # モデル設定
    config = {'d_model': 64, 'd_state': 32, 'num_layers': 2, 'time_steps': 16}
    model = BreakthroughSNN(vocab_size=vocab.vocab_size, **config)
    
    # 汎用Trainerのセットアップ
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = CombinedLoss()
    trainer = BreakthroughTrainer(model, optimizer, criterion)
    
    dataset = NextTokenPredictionDataset(train_data, vocab)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 学習ループ
    print("\n🔥 学習を開始します...")
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(dataloader)
        if (epoch + 1) % args.log_interval == 0:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"Epoch {epoch+1: >3}/{args.epochs}: {metrics_str}")
            
    # モデル保存
    model_path = args.model_path
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'config': config
    }, model_path)
    print(f"\n✅ 学習済みモデルを '{model_path}' に保存しました。")

def inference(args):
    """学習済みモデルで推論（文章生成）を実行"""
    try:
        engine = SNNInferenceEngine(model_path=args.model_path)
        
        # ユーザーからの入力を受け付けるループ
        print("\n💬 テキスト生成を開始します。終了するには 'exit' または 'quit' と入力してください。")
        while True:
            start_text = input("入力テキスト: ")
            if start_text.lower() in ["exit", "quit"]:
                break
            generated_text = engine.generate(start_text, max_len=args.max_len)
            print(f"生成結果: {generated_text}")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print(f"エラー: 学習済みモデルファイル({args.model_path})が必要です。先に 'train' モードで実行してください。")
    except Exception as e:
        print(f"❌ 予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNNベース AIチャットシステム")
    subparsers = parser.add_subparsers(dest="command", required=True, help="実行するコマンド")

    # --- 学習コマンド ---
    parser_train = subparsers.add_parser("train", help="SNNモデルを学習します")
    parser_train.add_argument("data_path", type=str, help="学習データのファイルパス (.json または .txt)")
    parser_train.add_argument("--json_key", type=str, default=None, help="JSONファイル内のテキストリストが格納されているキー")
    parser_train.add_argument("--epochs", type=int, default=100, help="学習エポック数")
    parser_train.add_argument("--batch_size", type=int, default=4, help="バッチサイズ")
    parser_train.add_argument("--learning_rate", type=float, default=5e-4, help="学習率")
    parser_train.add_argument("--log_interval", type=int, default=20, help="ログを表示するエポック間隔")
    parser_train.add_argument("--model_path", type=str, default="breakthrough_snn_model.pth", help="学習済みモデルの保存パス")
    parser_train.set_defaults(func=train)

    # --- 推論コマンド ---
    parser_inference = subparsers.add_parser("inference", help="学習済みモデルで推論（テキスト生成）を実行します")
    parser_inference.add_argument("--model_path", type=str, default="breakthrough_snn_model.pth", help="学習済みモデルのパス")
    parser_inference.add_argument("--max_len", type=int, default=30, help="生成するテキストの最大長")
    parser_inference.set_defaults(func=inference)

    args = parser.parse_args()
    args.func(args)
