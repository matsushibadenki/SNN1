# matsushibadenki/snn/benchmark/sst2_benchmark.py
#
# GLUEベンチマーク (SST-2タスク) を用いたSNNモデルの性能評価スクリプト
#
# 目的:
# - ロードマップ フェーズ1「1.1. ベンチマーク環境の構築」に対応。
# - 標準的なNLPタスクにおけるモデルの性能を客観的かつ定量的に評価する基盤を確立する。
#
# 機能:
# 1. Hugging Face `datasets`ライブラリからSST-2データセットを自動ダウンロード。
# 2. データセットをSNNモデルが学習可能な.jsonl形式に前処理・変換。
# 3. main.pyの学習・推論ロジックを呼び出し、モデルの訓練と評価を実行。
# 4. scikit-learnを使用し、テストデータに対する正解率(Accuracy)を計算・表示。

import os
import json
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import sys
import torch

# 親ディレクトリをsys.pathに追加して、mainやsnn_coreをインポート可能にする
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import run_training, SNNInferenceEngine
from snn_core import Vocabulary

def prepare_sst2_data(output_dir: str = "data"):
    """
    SST-2データセットをダウンロードし、.jsonl形式に変換して保存する。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Downloading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")

    data_paths = {}
    for split in ["train", "validation", "test"]:
        jsonl_path = os.path.join(output_dir, f"sst2_{split}.jsonl")
        data_paths[split] = jsonl_path
        
        print(f"Processing '{split}' split -> {jsonl_path}")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset[split]):
                # ラベルをテキストに変換（学習時にはテキスト全体で語彙を構築するため）
                # ここでは簡単化のため、テキストとラベルを結合した形式にはせず、
                # instruction形式として扱う
                instruction = "Classify the sentiment of the following sentence."
                output = "positive" if example['label'] == 1 else "negative"
                
                record = {
                    "instruction": instruction,
                    "input": example['sentence'],
                    "output": output
                }
                f.write(json.dumps(record) + "\n")
    
    print("✅ SST-2 data preparation complete.")
    return data_paths

def run_benchmark(data_paths: dict, model_path: str):
    """
    準備されたデータでモデルの学習と評価を行う。
    """
    print("\n🚀 Starting SST-2 Benchmark...")

    # --- 1. Training ---
    print("\n🔥 Step 1: Training the model on SST-2 train set...")
    # main.run_trainingを呼び出すための擬似的な引数オブジェクトを作成
    train_args = type('Args', (), {
        'data_path': data_paths['train'],
        'data_format': 'instruction',
        'epochs': 5, # ベンチマークのためエポック数は少なく設定
        'batch_size': 16,
        'learning_rate': 1e-4,
        'log_interval': 1,
        'model_path': model_path,
        'd_model': 64,
        'd_state': 32,
        'num_layers': 2,
        'time_steps': 16
    })()
    
    vocab = run_training(train_args)
    print("✅ Model training complete.")

    # --- 2. Evaluation ---
    print("\n📊 Step 2: Evaluating the model on SST-2 validation set...")
    engine = SNNInferenceEngine(model_path=model_path)
    
    true_labels = []
    pred_labels = []

    validation_data = []
    with open(data_paths['validation'], 'r', encoding='utf-8') as f:
        for line in f:
            validation_data.append(json.loads(line))

    for item in tqdm(validation_data, desc="Evaluating"):
        prompt = f"{item['instruction']}\n{item['input']}"
        generated_text = engine.generate(prompt, max_len=3) # "positive" or "negative"
        
        # 生成されたテキストから予測ラベルを決定
        if "positive" in generated_text:
            predicted_label = "positive"
        elif "negative" in generated_text:
            predicted_label = "negative"
        else:
            predicted_label = "unknown" # 生成がうまくいかなかった場合

        true_labels.append(item['output'])
        pred_labels.append(predicted_label)
        
    # --- 3. Calculate Metrics ---
    accuracy = accuracy_score(true_labels, pred_labels)
    
    print("\n🎉 Benchmark Results:")
    print("=" * 30)
    print(f"  Validation Accuracy: {accuracy:.4f}")
    print("=" * 30)

if __name__ == "__main__":
    # データ準備
    prepared_data_paths = prepare_sst2_data()
    
    # ベンチマーク実行
    output_model_path = "breakthrough_snn_sst2.pth"
    run_benchmark(prepared_data_paths, output_model_path)