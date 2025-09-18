# /path/to/your/project/train_text_snn.py
# テキストデータ（感情分析）からSNNモデルを学習・生成するプログラム
#
# このプログラムは以下の処理を実行します。
# 1. サンプルとなるテキストデータ（ポジティブ/ネガティブな文章）を準備する。
# 2. テキストを単語IDに変換し、単語埋め込みベクトルに変換する。
# 3. 埋め込みベクトルを「レートコーディング」によりスパイク列に変換する。
# 4. SNNモデルを定義し、生成したスパイク列で感情分析タスクを学習させる。
# 5. 学習済みモデルをファイルに保存する。

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from spikingjelly.activation_based import neuron, surrogate, functional
from collections import Counter
import itertools

# ----------------------------------------
# 1. データ準備と語彙の構築
# ----------------------------------------

# サンプルデータ（文章, ラベル）
# ラベル 0: ネガティブ, 1: ポジティブ
TRAIN_DATA = [
    ("this movie was terrible and a waste of time", 0),
    ("i absolutely loved it the acting was superb", 1),
    ("a complete disappointment from start to finish", 0),
    ("one of the best films i have ever seen", 1),
    ("the plot was confusing and the characters were boring", 0),
    ("a heartwarming story with brilliant performances", 1),
    ("i would not recommend this to anyone", 0),
    ("an unforgettable experience truly a masterpiece", 1),
    ("the script felt lazy and uninspired", 0),
    ("i was captivated from the very beginning", 1),
    ("horrible special effects and a predictable story", 0),
    ("two thumbs up a must see", 1),
    ("i fell asleep halfway through", 0),
    ("an emotional and powerful film", 1),
    ("what a mess i want my money back", 0),
    ("simply fantastic i will watch it again", 1)
]

class Vocabulary:
    """テキストとIDを相互変換するための語彙クラス"""
    def __init__(self, all_texts):
        self.word2idx = {"<UNK>": 0, "<PAD>": 1} # 未知語とパディング
        self.idx2word = {0: "<UNK>", 1: "<PAD>"}
        self.build_vocab(all_texts)

    def build_vocab(self, all_texts):
        # 全ての単語をフラットなリストにする
        all_words = list(itertools.chain.from_iterable(txt.split() for txt, _ in all_texts))
        word_counts = Counter(all_words)
        
        for word, _ in word_counts.items():
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def text_to_sequence(self, text):
        return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in text.split()]

    @property
    def vocab_size(self):
        return len(self.word2idx)


# ----------------------------------------
# 2. テキストからスパイクへの変換
# ----------------------------------------
def text_to_spike_train(text_sequence, embedding_layer, time_steps):
    """
    単語IDシーケンスをレートコーディングされたスパイク列に変換する。
    """
    # 1. 単語IDを埋め込みベクトルに変換
    with torch.no_grad():
        embedding_vectors = embedding_layer(torch.tensor(text_sequence, dtype=torch.long))
    
    # 2. 文章全体のベクトルを計算（ここでは単純な平均を使用）
    sentence_vector = torch.mean(embedding_vectors, dim=0)

    # 3. レートコーディング: ベクトル値をスパイクの発火率と見なす
    # ベクトル値を [0, 1] の範囲にクランプして発火率とする
    firing_rates = torch.clamp(sentence_vector, 0, 1)

    # 4. スパイク列を生成
    # 各タイムステップで、発火率に基づいてランダムにスパイクを生成
    spike_train = torch.rand(time_steps, len(firing_rates)) < firing_rates
    
    return spike_train.float()


class TextSpikeDataset(Dataset):
    """テキストデータをスパイク列に変換するカスタムDataset"""
    def __init__(self, data, vocab, embedding_layer, time_steps):
        self.data = data
        self.vocab = vocab
        self.embedding_layer = embedding_layer
        self.time_steps = time_steps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        text_seq = self.vocab.text_to_sequence(text)
        spike_train = text_to_spike_train(text_seq, self.embedding_layer, self.time_steps)
        return spike_train, torch.tensor(label, dtype=torch.long)


# ----------------------------------------
# 3. SNNモデルの定義、学習、保存
# ----------------------------------------
class TextSNN(nn.Module):
    """テキスト分類用のシンプルなSNNモデル"""
    def __init__(self, input_features, hidden_features, output_features):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(hidden_features, output_features)
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())

    def forward(self, x: torch.Tensor):
        # xの形状: (batch_size, time_steps, input_features)
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

def main():
    """メインの学習処理"""
    # ハイパーパラメータ
    embedding_dim = 64
    hidden_dim = 128
    output_dim = 2 # ポジティブ/ネガティブ
    time_steps = 30
    learning_rate = 0.01
    batch_size = 4
    num_epochs = 50

    # 1. 語彙と埋め込み層の準備
    print("語彙を構築中...")
    vocab = Vocabulary(TRAIN_DATA)
    embedding_layer = nn.Embedding(vocab.vocab_size, embedding_dim)
    print(f"語彙サイズ: {vocab.vocab_size}")

    # 2. データセットとデータローダーの作成
    print("データセットをスパイク列に変換中...")
    dataset = TextSpikeDataset(TRAIN_DATA, vocab, embedding_layer, time_steps)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. モデル、損失関数、オプティマイザの定義
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    model = TextSNN(embedding_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    # 埋め込み層のパラメータも一緒に学習させる
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(embedding_layer.parameters()), 
        lr=learning_rate
    )

    # 4. 学習ループ
    print("\n--- SNNモデルの学習開始 ---")
    for epoch in range(num_epochs):
        model.train()
        embedding_layer.train()
        for i, (spike_trains, labels) in enumerate(train_loader):
            spike_trains = spike_trains.to(device)
            labels = labels.to(device)

            outputs = model(spike_trains)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"エポック [{epoch+1}/{num_epochs}], 損失: {loss.item():.4f}")

    print("--- 学習完了 ---")
    
    # 5. 学習済みモデルの保存
    model_path = "snn_text_model.pth"
    # SNNモデルと埋め込み層の両方を保存
    torch.save({
        'snn_model_state_dict': model.state_dict(),
        'embedding_layer_state_dict': embedding_layer.state_dict(),
        'vocab': vocab
    }, model_path)
    print(f"✅ 学習済みモデルと語彙を '{model_path}' に保存しました。")

if __name__ == "__main__":
    main()