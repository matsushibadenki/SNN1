# matsushibadenki/snn/scripts/run_benchmark.py
# GLUEベンチマーク (SST-2) を用いたSNN vs ANN 性能評価スクリプト (リファクタリング版)
#
# 目的:
# - ロードマップ フェーズ1「1.1, 1.2」に対応。
# - 新しいプロジェクト構造の下で、SNNとANNの感情分析性能を客観的に比較・評価する。
#
# 改善点:
# - SNNモデルに専用の分類ヘッドを追加し、生成タスクとしてではなく分類タスクとして直接評価。
# - ANN/SNN双方のDataset, Collate Functionを共通化し、条件を統一。
# - 学習・評価ループを簡潔に記述。

import os
import json
import time
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# 新しいプロジェクト構造に対応したインポート
from snn_research.data.datasets import Vocabulary
from snn_research.core.snn_core import BreakthroughSNN
from snn_research.benchmark.ann_baseline import ANNBaselineModel

# --- 1. データ準備 ---
def prepare_sst2_data(output_dir: str = "data"):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    dataset = load_dataset("glue", "sst2")
    data_paths = {}
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
    def __init__(self, file_path, vocab):
        self.vocab = vocab
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        encoded = self.vocab.encode(self.data[idx]['text'], add_start_end=False)
        return torch.tensor(encoded, dtype=torch.long), self.data[idx]['label']

def collate_fn_for_classification(batch, pad_id):
    inputs, targets = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    return padded_inputs, torch.tensor(targets, dtype=torch.long)

# --- 3. SNN 分類モデル ---
class SNNClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, d_state, num_layers, time_steps, n_head, num_classes):
        super().__init__()
        self.snn_backbone = BreakthroughSNN(vocab_size, d_model, d_state, num_layers, time_steps, n_head)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, input_ids, src_padding_mask=None): # ANNとI/Fを合わせる
        _, spikes = self.snn_backbone(input_ids, return_spikes=True)
        # 時間積分された特徴量のシーケンス平均を取得
        pooled_features = spikes.mean(dim=1).mean(dim=1)
        return self.classifier(pooled_features)

# --- 4. 実行関数 ---
def run_benchmark_for_model(model_type: str, data_paths: dict, vocab: Vocabulary, model_params: dict):
    print("\n" + "="*20 + f" 🚀 Starting {model_type} Benchmark " + "="*20)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset_train = ClassificationDataset(data_paths['train'], vocab)
    dataset_val = ClassificationDataset(data_paths['validation'], vocab)
    
    collate_fn = lambda batch: collate_fn_for_classification(batch, vocab.pad_id)
    loader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, collate_fn=collate_fn)
    loader_val = DataLoader(dataset_val, batch_size=16, shuffle=False, collate_fn=collate_fn)

    if model_type == 'SNN':
        model = SNNClassifier(vocab_size=vocab.vocab_size, **model_params, num_classes=2).to(device)
    else: # ANN
        model = ANNBaselineModel(vocab_size=vocab.vocab_size, **model_params).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    
    print(f"{model_type} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(3):
        model.train()
        for inputs, targets in tqdm(loader_train, desc=f"{model_type} Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            padding_mask = (inputs == vocab.pad_id)
            optimizer.zero_grad()
            outputs = model(inputs, src_padding_mask=padding_mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    true_labels, pred_labels, latencies = [], [], []
    with torch.no_grad():
        for inputs, targets in tqdm(loader_val, desc=f"{model_type} Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            padding_mask = (inputs == vocab.pad_id)
            start_time = time.time()
            outputs = model(inputs, src_padding_mask=padding_mask)
            latencies.append((time.time() - start_time) * 1000)
            preds = torch.argmax(outputs, dim=1)
            pred_labels.extend(preds.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
            
    accuracy = accuracy_score(true_labels, pred_labels)
    avg_latency = sum(latencies) / len(latencies)

    print(f"  {model_type} Validation Accuracy: {accuracy:.4f}")
    print(f"  {model_type} Average Inference Time (per batch): {avg_latency:.2f} ms")
    return {"model": model_type, "accuracy": accuracy, "avg_latency_ms": avg_latency}

# --- 5. メイン実行ブロック ---
if __name__ == "__main__":
    pd.set_option('display.precision', 4)
    data_paths = prepare_sst2_data()
    
    vocab = Vocabulary()
    print("\n📖 Building shared vocabulary from training data...")
    with open(data_paths['train'], 'r', encoding='utf-8') as f:
        all_texts = (json.loads(line)['text'] for line in f)
        vocab.build_vocab(all_texts)
    print(f"✅ Vocabulary built. Size: {vocab.vocab_size}")

    snn_params = {'d_model': 64, 'd_state': 32, 'num_layers': 2, 'time_steps': 16, 'n_head': 2}
    snn_results = run_benchmark_for_model('SNN', data_paths, vocab, snn_params)

    ann_params = {'d_model': 64, 'd_hid': 128, 'nlayers': 2, 'nhead': 2}
    ann_results = run_benchmark_for_model('ANN', data_paths, vocab, ann_params)
    
    print("\n\n" + "="*25 + " 🏆 Final Benchmark Results " + "="*25)
    results_df = pd.DataFrame([snn_results, ann_results])
    print(results_df.to_string(index=False))
    print("="*75)
