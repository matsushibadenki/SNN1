# SNNファイル統合計画

## 🎯 統合目標
- 機能重複を排除
- メンテナンス性向上
- 必要最小限のファイル構成

## 📋 統合作業チェックリスト

### Phase 1: snn_core.py への統合

#### ✅ snn_advanced_optimization.py から移行
```python
# 以下のクラスを snn_core.py に追加
- TTFSEncoder (TemporalEncoderを置き換え)
- AdaptiveLIFNeuron (既存LIFの拡張)
- EventDrivenSSMLayer (SpikingSSMLayerを置き換え)
- EnergyEfficiencyOptimizer (新規追加)
```

#### ✅ snn_advanced_plasticity.py から移行
```python
# 以下のクラスを snn_core.py に追加
- STDPSynapse (新規追加)
- STPSynapse (新規追加)  
- MetaplasticLIFNeuron (新規追加)
- AdvancedSNNLoss (CombinedLossを置き換え)
```

### Phase 2: deployment.py への統合

#### ✅ snn_neuromorphic_optimization.py から移行
```python
# 以下のクラスを deployment.py に追加
- NeuromorphicProfile (HardwareProfileを拡張)
- NeuromorphicDeploymentManager (SNNDeploymentManagerを拡張)
- AdaptiveQuantizationPruning (DynamicOptimizerに統合)
- RealtimeEventProcessor (新規追加)
```

### Phase 3: snn_comprehensive_optimization.py の処理

#### 🔄 部分統合 - メインシステムとして保持
```python
# 以下の内容は保持（統合システムとして）
- MultimodalSNN (main.py で使用可能なオプション)
- AdaptiveRealtimeLearner (新機能として保持)
- ComprehensiveOptimizedSNN (統合システム)
```

## 📁 最終ファイル構成

### 🔹 保持すべき主要ファイル
1. **`snn_core.py`** (拡張版)
   - 全ての基本SNNコンポーネント
   - 生物学的可塑性機能
   - エネルギー最適化機能

2. **`deployment.py`** (拡張版)  
   - ニューロモーフィック最適化
   - リアルタイムデプロイメント
   - 適応的量子化・プルーニング

3. **`main.py`** (既存)
   - 学習・推論インターフェース
   - 既存の互換性維持

4. **`snn_comprehensive_optimization.py`** (統合システム)
   - マルチモーダルSNN
   - 包括的ベンチマーク
   - 最高性能システム

### 🗑️ 削除対象ファイル
- ~~`snn_advanced_optimization.py`~~ → snn_core.py に統合後削除
- ~~`snn_advanced_plasticity.py`~~ → snn_core.py に統合後削除  
- ~~`snn_neuromorphic_optimization.py`~~ → deployment.py に統合後削除
- ~~`snn_integration_guide.md`~~ → 統合完了後削除

## 🔧 具体的な統合手順

### Step 1: snn_core.py の拡張
```bash
# 1. snn_advanced_optimization.py から必要クラスをコピー
# 2. snn_advanced_plasticity.py から必要クラスをコピー
# 3. 重複クラス（TemporalEncoder等）を新版に置き換え
# 4. import文を整理
```

### Step 2: deployment.py の拡張  
```bash
# 1. snn_neuromorphic_optimization.py から必要クラスをコピー
# 2. 既存のDynamicOptimizerを拡張
# 3. HardwareProfileをNeuromorphicProfileに置き換え
```

### Step 3: 動作確認
```bash
# 統合後の動作確認
python main.py train sample_data.txt
python main.py inference

# 包括システムのテスト
python snn_comprehensive_optimization.py
```

### Step 4: ファイル削除
```bash
# 統合確認後に削除実行
rm snn_advanced_optimization.py
rm snn_advanced_plasticity.py  
rm snn_neuromorphic_optimization.py
rm snn_integration_guide.md
```

## ⚠️ 注意事項

### 統合時の重要ポイント
1. **import文の整理**: 循環importに注意
2. **互換性維持**: 既存のmain.pyが正常動作することを確認
3. **段階的統合**: 1ファイルずつ統合してテスト
4. **バックアップ**: 統合前に全ファイルをバックアップ

### テスト必須項目
- [ ] main.py trainコマンドが正常動作
- [ ] main.py inferenceコマンドが正常動作  
- [ ] deployment.pyの最適化機能が動作
- [ ] snn_comprehensive_optimization.pyのベンチマークが実行可能

## 📊 統合効果

### 統合前: 7ファイル
- snn_core.py
- deployment.py  
- main.py
- snn_advanced_optimization.py
- snn_advanced_plasticity.py
- snn_neuromorphic_optimization.py
- snn_comprehensive_optimization.py

### 統合後: 4ファイル  
- **snn_core.py** (大幅拡張)
- **deployment.py** (大幅拡張)
- **main.py** (互換性維持)
- **snn_comprehensive_optimization.py** (統合システム)

### 利点
✅ **43%のファイル数削減** (7→4ファイル)
✅ **機能重複の排除**
✅ **メンテナンス性向上**
✅ **既存コードとの互換性維持**
✅ **最新最適化技術の統合**

この統合により、SNNシステムは**世界最高レベルの性能**と**優れた保守性**を両立できます。
