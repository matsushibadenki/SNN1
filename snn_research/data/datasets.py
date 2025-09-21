# matsushibadenki/snn/snn_research/data/datasets.py
# 各種データ形式に対応するデータセットクラス
# 
# 機能:
# - main.pyにあったDatasetクラスと語彙クラスをこのモジュールに集約。
# - データ形式に応じたテキスト抽出ロジックを提供。

import torch
from torch.utils.data import Dataset
from collections import Counter
import itertools
from typing import List, Iterator, Dict, Any, Tuple
import os
import json
from enum import Enum

# --- 語彙クラス ---
class Vocabulary:
    """テキストとIDを相互変換するための語彙クラス"""
    def __init__(self):
        self.special_tokens = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.word2idx = self.special_tokens.copy()
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def build_vocab(self, all_texts: Iterator[str], max_size: int = 50000):
        all_words = itertools.chain.from_iterable(txt.lower().split() for txt in all_texts)
        word_counts = Counter(all_words)
        # 頻度上位の単語に絞り込む
        most_common_words = word_counts.most_common(max_size - len(self.special_tokens))
        
        for word, _ in most_common_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    
    def encode(self, text: str, add_start_end: bool = True) -> List[int]:
        tokens = [self.word2idx.get(word.lower(), self.special_tokens["<UNK>"]) for word in text.split()]
        if add_start_end:
            return [self.special_tokens["<START>"]] + tokens + [self.special_tokens["<END>"]]
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        ids_to_decode = [idx for idx in token_ids if idx not in (self.special_tokens["<START>"], self.special_tokens["<END>"])]
        return " ".join([self.idx2word.get(idx, "<UNK>") for idx in ids_to_decode])

    @property
    def vocab_size(self) -> int: return len(self.word2idx)
    
    @property
    def pad_id(self) -> int: return self.special_tokens["<PAD>"]


# --- データローダーとデータ形式 ---
class DataFormat(Enum):
    SIMPLE_TEXT = "simple_text"
    DIALOGUE = "dialogue"
    INSTRUCTION = "instruction"

def load_jsonl_data(file_path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# --- データセットクラス ---
class SNNBaseDataset(Dataset):
    """全てのデータセットクラスの基底クラス"""
    def __init__(self, file_path: str, vocab: Vocabulary):
        self.vocab = vocab
        self.data = list(load_jsonl_data(file_path))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]: raise NotImplementedError
    @staticmethod
    def extract_texts(file_path: str) -> Iterator[str]: raise NotImplementedError

class SimpleTextDataset(SNNBaseDataset):
    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.vocab.encode(item['text'])
        return torch.tensor(encoded[:-1]), torch.tensor(encoded[1:], dtype=torch.long)

    @staticmethod
    def extract_texts(file_path: str) -> Iterator[str]:
        for item in load_jsonl_data(file_path): yield item['text']

class DialogueDataset(SNNBaseDataset):
    def __getitem__(self, idx):
        item = self.data[idx]
        full_conversation = " ".join([turn['value'] for turn in item['conversations']])
        encoded = self.vocab.encode(full_conversation)
        return torch.tensor(encoded[:-1]), torch.tensor(encoded[1:], dtype=torch.long)

    @staticmethod
    def extract_texts(file_path: str) -> Iterator[str]:
        for item in load_jsonl_data(file_path):
            for turn in item['conversations']: yield turn['value']

class InstructionDataset(SNNBaseDataset):
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['instruction']
        if 'input' in item and item['input']: prompt += f"\n{item['input']}"
        full_text = f"{prompt}\n{item['output']}"
        encoded = self.vocab.encode(full_text)
        return torch.tensor(encoded[:-1]), torch.tensor(encoded[1:], dtype=torch.long)

    @staticmethod
    def extract_texts(file_path: str) -> Iterator[str]:
        for item in load_jsonl_data(file_path):
            yield item['instruction']
            if 'input' in item and item['input']: yield item['input']
            yield item['output']

def get_dataset_class(data_format: DataFormat) -> type[SNNBaseDataset]:
    format_map = {
        DataFormat.SIMPLE_TEXT: SimpleTextDataset,
        DataFormat.DIALOGUE: DialogueDataset,
        DataFormat.INSTRUCTION: InstructionDataset
    }
    return format_map[data_format]