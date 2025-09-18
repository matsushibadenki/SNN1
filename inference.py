# /path/to/your/project/inference.py
# 学習済みSNNモデルを使用してテキストの感情分析を行う推論エンジン

import torch
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import neuron, surrogate, functional
from collections import Counter
import itertools
import os

# ----------------------------------------
# 必要なクラス定義（学習時と同じものを再定義）
# ----------------------------------------

class Vocabulary:
    """テキストとIDを相互変換するための語彙クラス"""
    def __init__(self, all_texts):
        self.word2idx = {"<UNK>": 0, "<PAD>": 1}
        self.idx2word = {0: "<UNK>", 1: "<PAD>"}
        # 推論時はダミーのテキストで初期化（実際にはロードしたオブジェクトで上書き）
        if all_texts:
            self.build_vocab(all_texts)

    def build_vocab(self, all_texts):
        all_words = list(itertools.chain.from_iterable(txt.split() for txt, _ in all_texts))
        word_counts = Counter(all_words)
        for word, _ in word_counts.items():
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def text_to_sequence(self, text: str) -> list[int]:
        return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in text.split()]

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

class TextSNN(nn.Module):
    """テキスト分類用のシンプルなSNNモデル（学習時と同一構造）"""
    def __init__(self, input_features: int, hidden_features: int, output_features: int):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(hidden_features, output_features)
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)
        out_spikes_counter = torch.zeros(x.shape[0], self.fc2.out_features, device=x.device)
        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            y = self.fc1(x_t)
            y = self.lif1(y)
            y = self.fc2(y)
            y = self.lif2(y)
            out_spikes_counter += y
        return out_spikes_counter

# ----------------------------------------
# 推論エンジン本体
# ----------------------------------------

class SNNInferenceEngine:
    def __init__(self, model_path: str, device: str = "cpu"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
            
        self.device = torch.device(device)
        print(f"推論デバイス: {self.device}")

        # 1. モデルと関連データをファイルからロード
        # weights_only=False を追加して、Vocabularyオブジェクトも読み込むことを許可する
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.vocab = checkpoint['vocab']
        embedding_state_dict = checkpoint['embedding_layer_state_dict']
        snn_model_state_dict = checkpoint['snn_model_state_dict']

        # ロードした情報からモデルのパラメータを特定
        embedding_dim = embedding_state_dict['weight'].shape[1]
        hidden_dim = snn_model_state_dict['fc1.weight'].shape[0]
        output_dim = snn_model_state_dict['fc2.weight'].shape[0]
        
        # 2. モデルと埋め込み層を初期化し、学習済み重みを読み込む
        self.embedding_layer = nn.Embedding(self.vocab.vocab_size, embedding_dim).to(self.device)
        self.embedding_layer.load_state_dict(embedding_state_dict)
        
        self.model = TextSNN(embedding_dim, hidden_dim, output_dim).to(self.device)
        self.model.load_state_dict(snn_model_state_dict)

        # 3. 推論モードに設定（重要）
        self.embedding_layer.eval()
        self.model.eval()
        
        # ハイパーパラメータ（学習時と合わせる）
        self.time_steps = 30
        
        # ラベルのマッピング
        self.class_labels = {0: "ネガティブ", 1: "ポジティブ"}

    def _preprocess(self, text: str) -> torch.Tensor:
        """テキストをSNN入力用のスパイク列に変換する内部メソッド"""
        # 単語IDシーケンスに変換
        text_seq = self.vocab.text_to_sequence(text)
        if not text_seq:
            # 空のテキストの場合はゼロのテンソルを返す
            return torch.zeros(1, self.time_steps, self.embedding_layer.embedding_dim, device=self.device)
        
        # IDを埋め込みベクトルに変換
        with torch.no_grad():
            embedding_vectors = self.embedding_layer(torch.tensor(text_seq, dtype=torch.long, device=self.device))
        
        # 文章ベクトルを計算（平均化）
        sentence_vector = torch.mean(embedding_vectors, dim=0)
        
        # レートコーディングでスパイク列を生成
        firing_rates = torch.clamp(sentence_vector, 0, 1)
        spike_train = torch.rand(self.time_steps, len(firing_rates), device=self.device) < firing_rates
        
        # バッチ次元を追加して形状を (1, time_steps, features) にする
        return spike_train.float().unsqueeze(0)

    def predict(self, text: str) -> str:
        """
        単一のテキスト文章を受け取り、感情分析の推論結果を返す。
        """
        print(f"\n入力文章: '{text}'")
        
        # 1. テキストを前処理してスパイク列に変換
        spike_train_batch = self._preprocess(text)

        # 2. SNNモデルで推論を実行
        with torch.no_grad(): # 勾配計算は不要
            outputs = self.model(spike_train_batch)

        # 3. 結果を解釈
        # outputsは各クラスの合計スパイク数（または膜電位）
        # 最もスパイク数が多かったニューロンのインデックスを取得
        _, predicted_idx_tensor = torch.max(outputs.data, 1)
        predicted_idx = predicted_idx_tensor.item()
        
        # 4. ラベルに変換して返す
        return self.class_labels[predicted_idx]

# ----------------------------------------
# 実行ブロック
# ----------------------------------------
if __name__ == "__main__":
    MODEL_FILE_PATH = "snn_text_model.pth"
    
    try:
        # 推論エンジンを初期化
        engine = SNNInferenceEngine(model_path=MODEL_FILE_PATH)

        # --- 推論したい文章をここに入力 ---
        test_sentence_1 = "an unforgettable experience truly a masterpiece"
        test_sentence_2 = "the plot was confusing and the characters were boring"
        test_sentence_3 = "i will watch it again"
        test_sentence_4 = "what a mess"
        
        # 推論を実行して結果を表示
        prediction_1 = engine.predict(test_sentence_1)
        print(f"推論結果: {prediction_1}")

        prediction_2 = engine.predict(test_sentence_2)
        print(f"推論結果: {prediction_2}")

        prediction_3 = engine.predict(test_sentence_3)
        print(f"推論結果: {prediction_3}")

        prediction_4 = engine.predict(test_sentence_4)
        print(f"推論結果: {prediction_4}")

    except FileNotFoundError as e:
        print(e)
        print("エラー: 学習済みモデルファイルが必要です。先に 'train_text_snn.py' を実行してください。")