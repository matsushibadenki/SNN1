# matsushibadenki/snn/app/langchain_main.py
# LangChainと連携したSNNチャットアプリケーション
#
# 機能:
# - ロードマップ フェーズ2「2.4. プロトタイプ開発」に対応。
# - DIコンテナからSNNLangChainAdapterを取得。
# - LangChainのPromptTemplateとLLMChainを利用して、より構造化された応答を生成するデモ。
# - ストリーミング応答に対応。

import gradio as gr
import argparse
import sys
from pathlib import Path
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Iterator

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.containers import AppContainer

def main():
    parser = argparse.ArgumentParser(description="SNN + LangChain 連携AIプロトタイプ")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="設定ファイルのパス")
    parser.add_argument("--model_path", type=str, help="モデルのパス (設定ファイルを上書き)")
    args = parser.parse_args()

    # DIコンテナを初期化
    container = AppContainer()
    container.config.from_yaml(args.config)
    if args.model_path:
        container.config.model.path.from_value(args.model_path)

    # コンテナからLangChainアダプタを取得
    snn_llm = container.langchain_adapter()
    print(f"Loading SNN model from: {container.config.model.path()}")
    print("✅ SNN model loaded and wrapped for LangChain successfully.")

    # LangChainのプロンプトテンプレートを定義
    template = """
    あなたは、簡潔で役立つアシスタントです。ユーザーからの質問に日本語で答えてください。

    質問: {question}
    回答:
    """
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # LLMChainを作成
    llm_chain = LLMChain(prompt=prompt, llm=snn_llm)

# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def handle_message(message: str, history: list) -> Iterator[str]:
        """Gradioからの入力を処理し、LLMChainをストリーミング実行して応答を返す"""
        print("-" * 30)
        print(f"Input question to LLMChain: {message}")
        
        full_response = ""
        # LangChainのstreamメソッドを使用
        for chunk in llm_chain.stream({"question": message}):
            # streamは辞書を返すので、'text'キーの値を取得
            text_chunk = chunk.get('text', '')
            full_response += text_chunk
            yield full_response

        print(f"Generated answer: {full_response.strip()}")
        print("-" * 30)
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    # Gradioインターフェースの構築
    chatbot_interface = gr.ChatInterface(
        fn=handle_message,
        title="🤖 SNN + LangChain Prototype (Streaming)",
        description="SNNモデルをLangChain経由で利用するプロトタイプ。質問を入力してください。",
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="質問を入力...", container=False, scale=7),
        retry_btn=None,
        undo_btn="削除",
        clear_btn="クリア",
    )
    
    # Webアプリケーションの起動
    server_port = container.config.app.server_port() + 1 # ポートが衝突しないように+1する
    print("\nStarting Gradio web server for LangChain app...")
    print(f"Please open http://{container.config.app.server_name()}:{server_port} in your browser.")
    chatbot_interface.launch(
        server_name=container.config.app.server_name(),
        server_port=server_port,
    )

if __name__ == "__main__":
    main()
