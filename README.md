# **SNNベース AIチャットシステム**

## **1\. 概要**

本プロジェクトは、スパイキングニューラルネットワーク（SNN）技術を基盤とした、次世代のAIチャットシステムです。従来の人工ニューラルネットワーク（ANN）に匹敵する性能を目指しつつ、エネルギー効率とリアルタイム処理能力で圧倒的な優位性を持つことを特徴とします。

### **1.1. 設計思想**

* **エネルギー効率の最大化:** イベント駆動計算により、エッジデバイスでの超低消費電力動作を実現します。  
* **時間情報処理:** スパイクのタイミング情報を活用し、複雑な時系列データや文脈のニュアンスを捉えます。  
* **スケーラビリティ:** ニューロモーフィックハードウェアでの動作を視野に入れた拡張性の高いアーキテクチャを採用します。

### **1.2. ファイル構成**

* **snn\_core.py**: モデル定義など、システムの中核となる機能。  
* **main.py**: 学習と推論を実行するメインスクリプト。  
* **deployment.py**: モデルのデプロイや継続学習に関する機能。  
* **snn\_comprehensive\_optimization.py**: マルチモーダル対応など、先進的な統合システムのテスト。  
* **benchmark/**: SNNの性能を客観的に評価するためのスクリプト群。  
* **data\_preparation/**: 大規模データセットの前処理を行うスクリプト群。  
* **doc/**: 設計書やロードマップなどのドキュメント。

## **2\. 使い方 (How to Use)**

### **ステップ1: 環境設定**

まず、プロジェクトに必要なライブラリをインストールします。

pip install \-r requirements.txt

### **ステップ2: (オプション) 大規模データセットでの事前学習**

より汎用的な言語能力を持つモデルを育成するために、WikiTextデータセットで事前学習を行うことを推奨します。

1. データセットの準備:  
   以下のコマンドを実行すると、data/ ディレクトリに wikitext-103\_train.jsonl が生成されます。  
   python data\_preparation/prepare\_wikitext.py

2. 事前学習の実行:  
   準備したデータセットを使って、モデルを事前学習します。これには時間がかかります。  
   python main.py train data/wikitext-103\_train.jsonl \\  
       \--data\_format simple\_text \\  
       \--epochs 10 \\  
       \--batch\_size 32 \\  
       \--learning\_rate 1e-4 \\  
       \--model\_path snn\_pretrained\_on\_wikitext.pth

### **ステップ3: タスク特化データでの学習 (ファインチューニング)**

main.py を train モードで実行し、特定のタスクデータでSNNモデルを学習させます。学習データは **JSON Lines (.jsonl) 形式** を使用します。学習が完了すると、breakthrough\_snn\_model.pth ファイルが生成されます。

\--data\_format オプションで、データの構造を指定してください。

#### **例1: 基本的なテキスト (simple\_text) で学習**

各行に {"text": "..."} 形式で文章が記述された corpus.jsonl を使って学習します。

python main.py train path/to/corpus.jsonl \--data\_format simple\_text

#### **例2: 対話形式 (dialogue) で学習**

{"conversations": \[{"from": "user", ...}\]} 形式で対話データが記述された dialogues.jsonl を使って学習します。

python main.py train path/to/dialogues.jsonl \--data\_format dialogue \--epochs 200

#### **例3: 指示形式 (instruction) で学習**

{"instruction": "...", "output": "..."} 形式で指示応答データが記述された instructions.jsonl を使って学習します。

python main.py train path/to/instructions.jsonl \--data\_format instruction \--learning\_rate 1e-4

### **ステップ4: 学習済みモデルによる推論**

学習済みのモデルを使って、対話的にテキスト生成を実行します。

python main.py inference \--model\_path path/to/your\_model.pth

プログラムが起動すると、テキストを入力するように求められます。文章の冒頭を入力すると、モデルが続きを生成します。終了するには exit または quit と入力してください。

### **ステップ5: ベンチマークによる性能評価**

SST-2データセットを用いて、SNNモデルとANNベースラインモデルの性能（正解率、推論速度）を比較評価します。

python benchmark/sst2\_benchmark.py

## **3\. システムアーキテクチャ**

### **3.1. データフロー**

テキスト入力 →

入力処理  
→ スパイク列 →

SNNコアエンジン  
→ 出力スパイク →

出力処理  
→ テキスト応答

### **3.2. コアコンポーネント**

* **入力処理:** テキストをトークン化し、単語埋め込みを経て、スパイク列に変換します。  
* **SNNコアエンジン:** 本プロジェクトの核となるBreakthroughSNNモデルが処理を実行します。このモデルは以下の革新的技術を統合しています。  
  * **Spiking State Space Model (Spiking-SSM):** 線形計算量で長期依存関係を扱います。  
* **出力処理:** 出力スパイクをデコードし、次の単語トークンを生成してテキスト応答を構築します。

## **4\. 技術スタック**

| カテゴリ | 技術 | バージョン / 備考 |
| :---- | :---- | :---- |
| プログラミング言語 | Python | 3.10以降 |
| 機械学習バックエンド | PyTorch | 2.0以降 |
| SNNフレームワーク | SpikingJelly | 最新版 |
| 主要ライブラリ | NumPy, pandas, scikit-learn | データ操作、評価用 |
| 推論ターゲット | CPU / (Intel Loihi 2\) | 最終目標はニューロモーフィックハードウェア |

