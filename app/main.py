# matsushibadenki/snn/app/main.py
# DIコンテナを利用した、Gradioリアルタイム対話UIの起動スクリプト
#
# 機能:
# - DIコンテナを初期化し、設定を読み込む。
# - コンテナから完成品のChatServiceを取得してGradioに渡す。

import gradio as gr
import argparse
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from app.containers import AppContainer

def main():
    parser = argparse.ArgumentParser(description="SNNベース リアルタイム対話AI プロトタイプ")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="設定ファイルのパス")
    parser.add_argument("--model_path", type=str, help="モデルのパス (設定ファイルを上書き)")
    args = parser.parse_args()

    # DIコンテナを初期化し、設定ファイルを読み込む
    container = AppContainer()
    container.config.from_yaml(args.config)
    if args.model_path:
        container.config.model.path.from_value(args.model_path)

# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    # コンテナから完成品のChatServiceを取得
    chat_service = container.chat_service()
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    print(f"Loading SNN model from: {container.config.model.path()}")
    print("✅ SNN model loaded successfully via DI Container.")

    # Gradioインターフェースの構築
    chatbot_interface = gr.ChatInterface(
        fn=chat_service.handle_message,
        title="🤖 SNN-based AI Chat Prototype (DI Refactored)",
        description="進化したBreakthroughSNNモデルとのリアルタイム対話。メッセージを入力してEnterキーを押してください。",
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="SNNモデルに話しかける...", container=False, scale=7),
        retry_btn=None,
        undo_btn="削除",
        clear_btn="クリア",
    )

    # Webアプリケーションの起動
    print("\nStarting Gradio web server...")
    print(f"Please open http://{container.config.app.server_name()}:{container.config.app.server_port()} in your browser.")
    chatbot_interface.launch(
        server_name=container.config.app.server_name(),
        server_port=container.config.app.server_port(),
    )

if __name__ == "__main__":
    main()
