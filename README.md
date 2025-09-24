# **SNNベース AIチャットシステム (v2.2 \- 高速知識蒸留パイプライン実装)**

## **1\. 概要**

本プロジェクトは、スパイキングニューラルネットワーク（SNN）技術を基盤とした、次世代のAIチャットシステムです。DIコンテナの導入により、研究開発からアプリケーション化までをシームレスに繋ぐ、高い保守性と拡張性を持つアーキテクチャに刷新されました。

### **1.1. 設計思想**

* **関心の分離:** snn\_research（SNNコア技術の研究開発）と app（モデルを利用するアプリケーション）を明確に分離。  
* **依存性の注入 (DI):** dependency-injector を用い、クラス間の依存関係を外部コンテナで管理することで、疎結合でテスト容易性の高い設計を実現。  
* **設定の外部化:** モデルの構造や学習パラメータを configs/ ディレクトリのYAMLファイルで管理し、コードの変更なしに実験条件を変更可能に。

## **2\. 使い方 (How to Use)**

### **ステップ1: 環境設定**

まず、プロジェクトに必要なライブラリをインストールします。

pip install \-r requirements.txt

### **ステップ2: データ準備**

#### **2.1. 通常学習用データ (オプション)**

WikiTextのような大規模データセットを準備する場合、以下のスクリプトを実行します。

python \-m scripts.data\_preparation

#### **2.2. 知識蒸留用データ (必須)**

知識蒸留を行う前に、教師モデルのロジットを事前計算する必要があります。

\# 例: sample\_data.jsonl から蒸留用データを作成し、 precomputed\_data/ ディレクトリに保存  
python \-m scripts.prepare\_distillation\_data \\  
    \--input\_file data/sample\_data.jsonl \\  
    \--output\_dir precomputed\_data/

### **ステップ3: モデルの学習**

新しい統合学習スクリプト train.py を使用します。学習の挙動は設定ファイルで制御します。

**例1: 基本的な学習**

\# configs/base\_config.yaml の設定で学習を開始します。  
python train.py \--config configs/base\_config.yaml \--data\_path data/sample\_data.jsonl

**例2: 知識蒸留 (GPUが2つ以上ある場合)**

\# 1\. configs/base\_config.yaml の training.type を "distillation" に変更  
\# 2\. 事前計算済みデータディレクトリを指定して学習を実行  
\#    (スクリプトが自動でGPUを検出し、分散学習を開始します)  
python train.py \\  
    \--config configs/base\_config.yaml \\  
    \--data\_path precomputed\_data/

### **ステップ4: 対話アプリケーションの起動**

学習済みのモデルを使って、GradioベースのチャットUIを起動します。

python \-m app.main \--model\_path breakthrough\_snn\_model.pth

http://0.0.0.0:7860 を開いてください。

### **ステップ5: ベンチマークによる性能評価**

SST-2データセットを用いて、SNNモデルとANNベースラインモデルの性能を比較評価します。

python \-m scripts.run\_benchmark  
