# **SNNベース AIチャットシステム (v2.0 \- DI/LangChain対応版)**

## **1\. 概要**

本プロジェクトは、スパイキングニューラルネットワーク（SNN）技術を基盤とした、次世代のAIチャットシステムです。DIコンテナの導入により、研究開発からアプリケーション化までをシームレスに繋ぐ、高い保守性と拡張性を持つアーキテクチャに刷新されました。

### **1.1. 設計思想**

* **関心の分離:** snn\_research（SNNコア技術の研究開発）と app（モデルを利用するアプリケーション）を明確に分離。  
* **依存性の注入 (DI):** dependency-injector を用い、クラス間の依存関係を外部コンテナで管理することで、疎結合でテスト容易性の高い設計を実現。  
* **設定の外部化:** モデルの構造や学習パラメータを configs/ ディレクトリのYAMLファイルで管理し、コードの変更なしに実験条件を変更可能に。

### **1.2. ディレクトリ構成**

* **app/**: Gradio UI、LangChainアダプタなど、SNNモデルを利用するアプリケーション層。  
* **configs/**: プロジェクト全体の設定を管理するYAMLファイル。  
* **snn\_research/**: SNNモデルのコアロジック、データセット、学習アルゴリズムなど、研究開発に関わるコードを集約。  
* **scripts/**: データ準備やベンチマーク実行など、独立して実行可能なスクリプト。  
* **train.py**: 全ての学習（通常、分散、蒸留）を統合した単一の実行スクリプト。

## **2\. 使い方 (How to Use)**

### **ステップ1: 環境設定**

まず、プロジェクトに必要なライブラリをインストールします。

pip install \-r requirements.txt

### **ステップ2: データ準備 (オプション)**

WikiTextのような大規模データセットを準備する場合、以下のスクリプトを実行します。

python \-m scripts.data\_preparation

これにより、data/wikitext-103\_train.jsonl が生成されます。

### **ステップ3: モデルの学習**

新しい統合学習スクリプト train.py を使用します。学習の挙動は設定ファイルで制御します。

例1: 基本的な学習  
configs/base\_config.yaml の設定で学習を開始します。  
python train.py --config configs/base_config.yaml --data_path data/wikitext-103_train.jsonl

例2: 分散学習 (GPUが2つの場合)  
configs/base\_config.yaml の training.type を distributed に変更し、以下を実行します。  
torchrun \--nproc\_per\_node=2 train.py \--config configs/base\_config.yaml

例3: 知識蒸留  
configs/base\_config.yaml の training.type を distillation に変更し、以下を実行します。  
torchrun \--nproc\_per\_node=2 train.py \--config configs/base\_config.yaml

### **ステップ4: 対話アプリケーションの起動**

学習済みのモデルを使って、GradioベースのチャットUIを起動します。

python \-m app.main \--model\_path path/to/your\_model.pth

ブラウザで http://0.0.0.0:7860 を開いてください。

### **ステップ5: ベンチマークによる性能評価**

SST-2データセットを用いて、SNNモデルとANNベースラインモデルの感情分析性能を比較評価します。

python \-m scripts.run\_benchmark

## **3\. 技術スタック**

| カテゴリ | 技術 |
| :---- | :---- |
| プログラミング言語 | Python 3.10+ |
| 機械学習バックエンド | PyTorch 2.0+ |
| SNNフレームワーク | SpikingJelly |
| DIコンテナ | dependency-injector |
| アプリケーション | Gradio, LangChain |

