# matsushibadenki/snn/scripts/run_benchmark.py
# GLUEベンチマーク (SST-2) を用いたSNN vs ANN 性能評価スクリプト (Tokenizer移行版)
#
# 目的:
# - ロードマップ フェーズ1「1.2」に対応。
# - SNNとANNの感情分析性能を客観的に比較・評価する。
# - SNNのエネルギー消費の代理指標として「合計スパイク数」を計測し、比較項目に追加。
# - 独自Vocabularyを廃止し、Hugging Face Tokenizerを使用するよう変更。

import os
import json
import time
import pandas as pd  # type: ignore
from datasets import load_dataset  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from tqdm import tqdm  # type: ignore
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Any, List, Tuple
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from snn_research.core.snn_core import BreakthroughSNN
from snn_research.benchmark.ann_baseline import ANNBaselineModel

# --- 1. データ準備 ---
def prepare_sst2_data(output_dir: str = "data") -> Dict[str, str]:
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    dataset = load_dataset("glue", "sst2")
    data_paths: Dict[str, str] = {}
    for split in ["train", "validation"]:
        jsonl_path = os.path.join(output_dir, f"sst2_{split}.jsonl")
        data_paths[split] = jsonl_path
        if os.path.exists(jsonl_path): continue
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for ex in tqdm(dataset[split], desc=f"Processing {split}"):
                f.write(json.dumps({"text": ex['sentence'], "label": ex['label']}) + "\n")
    return data_paths


# --- 2. 共通データセット ---
class ClassificationDataset(Dataset):
    def __init__(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
    def __len__(self) -> int: return len(self.data)
    def __getitem__(self, idx: int) -> Tuple[str, int]:
        # collate_fnでまとめてトークン化するため、テキストとラベルをそのまま返す
        return self.data[idx]['text'], self.data[idx]['label']

def create_collate_fn_for_classification(tokenizer: PreTrainedTokenizerBase):
    def collate_fn(batch: List[Tuple[str, int]]):
        texts, targets = zip(*batch)
        tokenized = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        return tokenized['input_ids'], tokenized['attention_mask'], torch.tensor(targets, dtype=torch.long)
    return collate_fn

# --- 3. SNN 分類モデル ---
class SNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, time_steps: int, n_head: int, num_classes: int):
        super().__init__()
        self.snn_backbone = BreakthroughSNN(vocab_size, d_model, d_state, num_layers, time_steps, n_head)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, input_ids: torch.Tensor, src_padding_mask: Any = None) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, spikes = self.snn_backbone(input_ids, return_spikes=True)
        # SNNバックボーンからのスパイク情報を利用して分類
        if spikes.numel() > 0:
            pooled_features = spikes.mean(dim=1).mean(dim=1)
        else: # スパイクがない場合（フォールバック）
            pooled_features = torch.zeros(input_ids.shape[0], self.snn_backbone.output_projection.in_features, device=input_ids.device)
        
        logits = self.classifier(pooled_features)
        return logits, spikes

# --- 4. 実行関数 ---
def run_benchmark_for_model(model_type: str, data_paths: dict, tokenizer: PreTrainedTokenizerBase, model_params: dict) -> Dict[str, Any]:
    print("\n" + "="*20 + f" 🚀 Starting {model_type} Benchmark " + "="*20)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset_train = ClassificationDataset(data_paths['train'])
    dataset_val = ClassificationDataset(data_paths['validation'])
    
    collate_fn = create_collate_fn_for_classification(tokenizer)
    loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
    loader_val = DataLoader(dataset_val, batch_size=32, shuffle=False, collate_fn=collate_fn)

    vocab_size = tokenizer.vocab_size
    model: nn.Module
    if model_type == 'SNN':
        model = SNNClassifier(vocab_size=vocab_size, **model_params, num_classes=2).to(device)
    else: # ANN
        model = ANNBaselineModel(vocab_size=vocab_size, **model_params).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    
    print(f"{model_type} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(3):
        model.train()
        for inputs, attention_mask, targets in tqdm(loader_train, desc=f"{model_type} Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            padding_mask = (attention_mask == 0) # 0がパディング
            optimizer.zero_grad()
            
            outputs, _ = model(inputs, src_padding_mask=padding_mask) if model_type == 'SNN' else (model(inputs, src_padding_mask=padding_mask), None)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    true_labels: List[int] = []
    pred_labels: List[int] = []
    latencies: List[float] = []
    total_spikes: float = 0.0
    with torch.no_grad():
        for inputs, attention_mask, targets in tqdm(loader_val, desc=f"{model_type} Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            padding_mask = (attention_mask == 0)
            start_time = time.time()
            if model_type == 'SNN':
                outputs, spikes = model(inputs, src_padding_mask=padding_mask)
                total_spikes += spikes.sum().item()
            else:
                outputs = model(inputs, src_padding_mask=padding_mask)

            latencies.append((time.time() - start_time) * 1000)
            preds = torch.argmax(outputs, dim=1)
            pred_labels.extend(preds.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
            
    accuracy = accuracy_score(true_labels, pred_labels)
    avg_latency = sum(latencies) / len(latencies)
    avg_spikes = total_spikes / len(dataset_val) if model_type == 'SNN' else 'N/A'

    print(f"  {model_type} Validation Accuracy: {accuracy:.4f}")
    print(f"  {model_type} Average Inference Time (per batch): {avg_latency:.2f} ms")
    if model_type == 'SNN':
        print(f"  {model_type} Average Spikes per Sample: {avg_spikes:,.2f}")
        
    return {"model": model_type, "accuracy": accuracy, "avg_latency_ms": avg_latency, "avg_spikes_per_sample": avg_spikes}

# --- 5. メイン実行ブロック ---
if __name__ == "__main__":
    pd.set_option('display.precision', 4)
    data_paths = prepare_sst2_data()
    
    # 蒸留でも使うGPT-2のTokenizerをベンチマークでも使用し、条件を統一
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    snn_params = {'d_model': 64, 'd_state': 32, 'num_layers': 2, 'time_steps': 64, 'n_head': 2}
    snn_results = run_benchmark_for_model('SNN', data_paths, tokenizer, snn_params)

    ann_params = {'d_model': 64, 'd_hid': 128, 'nlayers': 2, 'nhead': 2}
    ann_results = run_benchmark_for_model('ANN', data_paths, tokenizer, ann_params)
    
    print("\n\n" + "="*25 + " 🏆 Final Benchmark Results " + "="*25)
    results_df = pd.DataFrame([snn_results, ann_results])
    print(results_df.to_string(index=False))
    print("="*75)
