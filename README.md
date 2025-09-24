# **SNNベース AIチャットシステム (v2.3 \- 設定ファイルモジュール化)**

## **1\. 概要**

本プロジェクトは、スパイキングニューラルネットワーク（SNN）技術を基盤とした、次世代のAIチャットシステムです。DIコンテナの導入により、研究開発からアプリケーション化までをシームレスに繋ぐ、高い保守性と拡張性を持つアーキテクチャに刷新されました。

### **1.1. 設計思想**

* **関心の分離:** snn\_research（SNNコア技術の研究開発）と app（モデルを利用するアプリケーション）を明確に分離。  
* **依存性の注入 (DI):** dependency-injector を用い、クラス間の依存関係を外部コンテナで管理することで、疎結合でテスト容易性の高い設計を実現。  
* **設定のモジュール化:** 学習設定（configs/base\_config.yaml）とモデルアーキテクチャ設定（configs/models/\*.yaml）を分離し、実験の組み合わせを容易に。

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

\# 例: sample\_data.jsonl から蒸留用データを作成  
python \-m scripts.prepare\_distillation\_data \\  
    \--input\_file data/sample\_data.jsonl \\  
    \--output\_dir precomputed\_data/

### **ステップ3: モデルの学習**

統合学習スクリプト train.py を使用します。--configでベース設定を、--model\_configでモデルのアーキテクチャを指定します。

**例1: 基本的な学習（smallモデル）**

\# smallモデルのアーキテクチャで学習を開始します。  
python train.py \\  
    \--config configs/base\_config.yaml \\  
    \--model\_config configs/models/small.yaml \\  
    \--data\_path data/sample\_data.jsonl

**例2: 知識蒸留（mediumモデル, GPUが2つ以上ある場合）**

\# 1\. configs/base\_config.yaml の training.type を "distillation" に変更  
\# 2\. mediumモデルのアーキテクチャで知識蒸留を実行  
\#    (スクリプトが自動でGPUを検出し、分散学習を開始します)  
python train.py \\  
    \--config configs/base\_config.yaml \\  
    \--model\_config configs/models/medium.yaml \\  
    \--data\_path precomputed\_data/

### **ステップ4: 対話アプリケーションの起動**

学習済みのモデルを使って、GradioベースのチャットUIを起動します。**学習時に使用したモデル設定ファイルを指定してください。**

**例: mediumモデルを起動**

python \-m app.main \\  
    \--config configs/base\_config.yaml \\  
    \--model\_config configs/models/medium.yaml

http://0.0.0.0:7860 を開いてください。

### **ステップ5: ベンチマークによる性能評価**

SST-2データセットを用いて、SNNモデルとANNベースラインモデルの性能を比較評価します。

python \-m scripts.run\_benchmark  
