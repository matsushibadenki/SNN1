# matsushibadenki/snn/snn_research/data/datasets.py
# 各種データ形式に対応するデータセットクラス
# 
# 機能:
# - Hugging Face Tokenizer を使用するように全面的に刷新。
# - 旧来の独自Vocabularyクラスを廃止し、標準的なNLPパイプラインとの互換性を向上。
# - データ形式に応じたテキスト抽出ロジックを提供。

import torch
from torch.utils.data import Dataset
from typing import Iterator, Dict, Any, Tuple
import os
import json
from enum import Enum
from transformers import PreTrainedTokenizerBase

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
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizerBase, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = list(load_jsonl_data(file_path))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]: raise NotImplementedError
    
    def _encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        # <BOS>トークンを追加し、<EOS>トークンはターゲット側で考慮
        return self.tokenizer(
            f"{self.tokenizer.bos_token or ''}{text}",
            truncation=True,
            max_length=self.max_seq_len,
            padding=False, # collate_fnでパディング
            return_tensors="pt"
        )

    @staticmethod
    def extract_texts(file_path: str) -> Iterator[str]: raise NotImplementedError

class SimpleTextDataset(SNNBaseDataset):
    def __getitem__(self, idx):
        item = self.data[idx]
        tokenized = self._encode_text(item['text'])
        input_ids = tokenized['input_ids'].squeeze(0)
        return input_ids[:-1], input_ids[1:]

    @staticmethod
    def extract_texts(file_path: str) -> Iterator[str]:
        for item in load_jsonl_data(file_path): yield item['text']

class DialogueDataset(SNNBaseDataset):
    def __getitem__(self, idx):
        item = self.data[idx]
        # 各発話の間にEOSトークンを挟み、会話の区切りをモデルに学習させる
        eos_token = self.tokenizer.eos_token or ''
        full_conversation = f" {eos_token} ".join([turn['value'] for turn in item['conversations']])
        tokenized = self._encode_text(full_conversation)
        input_ids = tokenized['input_ids'].squeeze(0)
        return input_ids[:-1], input_ids[1:]

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
        tokenized = self._encode_text(full_text)
        input_ids = tokenized['input_ids'].squeeze(0)
        return input_ids[:-1], input_ids[1:]

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
