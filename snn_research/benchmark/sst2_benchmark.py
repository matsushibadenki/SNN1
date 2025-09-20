# matsushibadenki/snn/benchmark/sst2_benchmark.pyの修正
#
# GLUEベンチマーク (SST-2タスク) を用いたSNNモデルの性能評価スクリプト
#
# 目的:
# - ロードマップ フェーズ1「1.1. ベンチマーク環境の構築」「1.2. ANNベースラインとの比較」に対応。
# - 標準的なNLPタスクにおけるSNNモデルの性能を、ANNモデルと比較して客観的かつ定量的に評価する。
#
# 機能:
# 1. Hugging Face `datasets`ライブラリからSST-2データセットを自動ダウンロード。
# 2. データセットをSNN/ANNモデルが学習可能な形式に前処理・変換。
# 3. SNNモデルの学習と評価を実行。
# 4. ANNベースラインモデルの学習と評価を実行。
# 5. 両モデルの性能指標（正解率、推論時間）を並べて表示し、比較を容易にする。

import os
import json
import time
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 親ディレクトリをsys.pathに追加して、mainやsnn_coreをインポート可能にする
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import Vocabulary, collate_fn
from snn_core import BreakthroughSNN, BreakthroughTrainer, CombinedLoss
from benchmark.ann_baseline import ANNBaselineModel

# ----------------------------------------
# 1. データ準備
# ----------------------------------------
def prepare_sst2_data(output_dir: str = "data"):
    """
    SST-2データセットをダウンロードし、.jsonl形式に変換して保存する。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Downloading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")

    data_paths = {}
    for split in ["train", "validation"]: # testスプリットはラベルがないため除外
        jsonl_path = os.path.join(output_dir, f"sst2_{split}.jsonl")
        data_paths[split] = jsonl_path
        
        if os.path.exists(jsonl_path):
            print(f"'{split}' split already exists at {jsonl_path}. Skipping preparation.")
            continue
            
        print(f"Processing '{split}' split -> {jsonl_path}")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset[split]):
                record = {
                    "sentence": example['sentence'],
                    "label": example['label'] 
                }
                f.write(json.dumps(record) + "\n")
    
    print("✅ SST-2 data preparation complete.")
    return data_paths

# ----------------------------------------
# 2. SNNモデルのベンチマーク
# ----------------------------------------
def run_snn_benchmark(data_paths: dict, model_path: str, vocab: Vocabulary):
    """準備されたデータでSNNモデルの学習と評価を行う。"""
    print("\n" + "="*20 + " 🚀 Starting SNN Benchmark " + "="*20)

    # --- 1. Training ---
    print("\n🔥 Step 1: Training the SNN model on SST-2 train set...")
    train_args = type('Args', (), {
        'data_path': data_paths['train'],
        'data_format': 'instruction', # データ形式をinstructionに偽装して流用
        'epochs': 3, 
        'batch_size': 16,
        'learning_rate': 1e-4,
        'log_interval': 1,
        'model_path': model_path,
        'd_model': 64, 'd_state': 32, 'num_layers': 2, 'time_steps': 16
    })()
    
    # データをInstruction形式に変換するラッパー
    def convert_to_instruction_format(original_path, new_path):
        with open(original_path, 'r') as fin, open(new_path, 'w') as fout:
            for line in fin:
                item = json.loads(line)
                instruction = "Classify the sentiment of the following sentence."
                output = "positive" if item['label'] == 1 else "negative"
                fout.write(json.dumps({"instruction": instruction, "input": item['sentence'], "output": output}) + "\n")
    
    train_inst_path = data_paths['train'].replace('.jsonl', '_inst.jsonl')
    convert_to_instruction_format(data_paths['train'], train_inst_path)
    train_args.data_path = train_inst_path
    
    run_training(train_args, vocab)
    print("✅ SNN Model training complete.")

    # --- 2. Evaluation ---
    print("\n📊 Step 2: Evaluating the SNN model on SST-2 validation set...")
    engine = SNNInferenceEngine(model_path=model_path)
    
    true_labels, pred_labels, latencies = [], [], []

    with open(data_paths['validation'], 'r', encoding='utf-8') as f:
        validation_data = [json.loads(line) for line in f]

    for item in tqdm(validation_data, desc="SNN Evaluating"):
        prompt = f"Classify the sentiment of the following sentence.\n{item['sentence']}"
        
        start_time = time.time()
        generated_text = engine.generate(prompt, max_len=3)
        latencies.append((time.time() - start_time) * 1000) # ms

        pred_labels.append(1 if "positive" in generated_text else 0)
        true_labels.append(item['label'])
        
    # --- 3. Calculate Metrics ---
    accuracy = accuracy_score(true_labels, pred_labels)
    avg_latency = sum(latencies) / len(latencies)
    
    print(f"  SNN Validation Accuracy: {accuracy:.4f}")
    print(f"  SNN Average Inference Time: {avg_latency:.2f} ms")
    return {"model": "BreakthroughSNN", "accuracy": accuracy, "avg_latency_ms": avg_latency}

# SNN用の分類データセット
class SNNClassificationDataset(Dataset):
    def __init__(self, file_path, vocab):
        self.vocab = vocab
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # SNNモデルの入力は (input, target) のペア
                encoded = self.vocab.encode(item['sentence'])
                # ラベル情報を損失計算のために別途保持
                self.data.append({'encoded': encoded, 'label': item['label']})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = item['encoded']
        # SNNは次トークン予測で事前学習するため、入力とターゲットを生成
        # 分類タスクでは最後の出力のみを使うが、学習形式は合わせる
        return torch.tensor(encoded[:-1]), torch.tensor(encoded[1:], dtype=torch.long)

# SNNの分類ヘッド
class SNNClassifier(nn.Module):
    def __init__(self, snn_backbone, d_model, num_classes):
        super().__init__()
        self.snn_backbone = snn_backbone
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, input_ids):
        # SNNバックボーンから特徴量を取得
        logits, spikes = self.snn_backbone(input_ids, return_spikes=True)
        # シーケンス全体の時間積分された特徴量を使用
        time_integrated_features = spikes.mean(dim=1)
        # シーケンス方向で平均プーリング
        pooled_features = time_integrated_features.mean(dim=1)
        # 分類
        class_logits = self.classifier(pooled_features)
        return class_logits, spikes


def run_snn_benchmark(data_paths: dict, model_path: str, vocab: Vocabulary):
    """準備されたデータでSNNモデルの学習と評価を行う。"""
    print("\n" + "="*20 + " 🚀 Starting SNN Benchmark " + "="*20)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. DataLoaders ---
    # ここでは、SNNもANNと同様のデータセット形式で扱う
    train_dataset = SNNClassificationDataset(data_paths['train'], vocab)
    val_dataset = SNNClassificationDataset(data_paths['validation'], vocab)

    # (DataLoaderの定義はANNと共通化可能)
    # ...

    # --- 2. Model, Optimizer, Loss ---
    snn_backbone = BreakthroughSNN(vocab_size=vocab.vocab_size, d_model=64, d_state=32, num_layers=2, time_steps=16)
    model = SNNClassifier(snn_backbone, d_model=64, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # 分類なのでCrossEntropyLossを直接使う
    criterion = nn.CrossEntropyLoss()
    
    # --- 3. Training Loop (専用のループを実装) ---
    print("\n🔥 Step 1: Training the SNN model for classification...")
    for epoch in range(3):
        model.train()
        # (ここに専用の学習ループを実装)
    
    # --- 4. Evaluation (専用の評価ループを実装) ---
    print("\n📊 Step 2: Evaluating the SNN model...")
    # (ここに専用の評価ループを実装)

    # (結果の計算と返却)
    # ...
    return {"model": "BreakthroughSNN", "accuracy": 0.0, "avg_latency_ms": 0.0} # ダミー

# ----------------------------------------
# 3. ANNベースラインのベンチマーク
# ----------------------------------------
class SST2Dataset(Dataset):
    def __init__(self, file_path, vocab):
        self.vocab = vocab
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        text = self.data[idx]['sentence']
        encoded = self.vocab.encode(text, add_start_end=False)
        return torch.tensor(encoded, dtype=torch.long), self.data[idx]['label']

def run_ann_training_and_eval(data_paths: dict, vocab: Vocabulary, model_params: dict):
    """ANNベースラインモデルの学習と評価を行う。"""
    print("\n" + "="*20 + " 📊 Starting ANN Benchmark " + "="*20)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. DataLoaders ---
    train_dataset = SST2Dataset(data_paths['train'], vocab)
    val_dataset = SST2Dataset(data_paths['validation'], vocab)

    def ann_collate_fn(batch):
        inputs, targets = zip(*batch)
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=vocab.pad_id)
        return padded_inputs, torch.tensor(targets, dtype=torch.long)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=ann_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=ann_collate_fn)

    # --- 2. Model, Optimizer, Loss ---
    model = ANNBaselineModel(vocab_size=vocab.vocab_size, **model_params).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"ANN Model Parameters: {sum(p.numel() for p in model.parameters())}")

    # --- 3. Training Loop ---
    print("\n🔥 Step 1: Training the ANN model...")
    for epoch in range(3): # SNNとエポック数を合わせる
        model.train()
        for inputs, targets in tqdm(train_loader, desc=f"ANN Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            padding_mask = (inputs == vocab.pad_id)
            
            optimizer.zero_grad()
            outputs = model(inputs, padding_mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    print("✅ ANN Model training complete.")
            
    # --- 4. Evaluation ---
    print("\n📊 Step 2: Evaluating the ANN model...")
    model.eval()
    true_labels, pred_labels, latencies = [], [], []
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="ANN Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            padding_mask = (inputs == vocab.pad_id)
            
            start_time = time.time()
            outputs = model(inputs, padding_mask)
            latencies.append((time.time() - start_time) * 1000)
            
            preds = torch.argmax(outputs, dim=1)
            pred_labels.extend(preds.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
    
    # --- 5. Calculate Metrics ---
    accuracy = accuracy_score(true_labels, pred_labels)
    # 1バッチあたりの平均レイテンシ
    avg_latency = sum(latencies) / len(latencies)

    print(f"  ANN Validation Accuracy: {accuracy:.4f}")
    print(f"  ANN Average Inference Time (per batch): {avg_latency:.2f} ms")
    return {"model": "ANN Baseline", "accuracy": accuracy, "avg_latency_ms": avg_latency}

# ----------------------------------------
# 4. メイン実行ブロック
# ----------------------------------------
if __name__ == "__main__":
    # --- 準備 ---
    pd.set_option('display.precision', 4)
    data_paths = prepare_sst2_data()
    snn_model_path = "breakthrough_snn_sst2.pth"
    
    # 共通の語彙を構築
    vocab = Vocabulary()
    print("\n📖 Building shared vocabulary from training data...")
    with open(data_paths['train'], 'r', encoding='utf-8') as f:
        all_texts = (json.loads(line)['sentence'] for line in f)
        vocab.build_vocab(all_texts)
    print(f"✅ Vocabulary built. Size: {vocab.vocab_size}")

    # --- SNNベンチマーク実行 ---
    snn_results = run_snn_benchmark(data_paths, snn_model_path, vocab)

    # --- ANNベンチマーク実行 ---
    ann_model_params = {'d_model': 64, 'nhead': 2, 'd_hid': 128, 'nlayers': 2}
    ann_results = run_ann_training_and_eval(data_paths, vocab, ann_model_params)
    
    # --- 結果の比較 ---
    print("\n\n" + "="*25 + " 🏆 Final Benchmark Results " + "="*25)
    results_df = pd.DataFrame([snn_results, ann_results])
    print(results_df.to_string(index=False))
    print("="*75)
    
    # 注: SNNのレイテンシは1文ごと、ANNは1バッチごとのため直接比較はできない点に注意。
    # ANNの方がベクトル化演算によりバッチ処理で高速になる傾向がある。
