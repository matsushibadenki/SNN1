# matsushibadenki/snn/app/main.py
# DIコンテナを利用した、Gradioリアルタイム対話UIの起動スクリプト
#
# 機能:
# - DIコンテナを初期化し、設定を読み込む。
# - コンテナから完成品のChatServiceを取得してGradioに渡す。
# - Gradio Blocksを使用して、チャット画面とリアルタイム統計情報パネルを持つUIを構築。
# - 会話履歴をクリアする機能を追加。

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

    container = AppContainer()
    container.config.from_yaml(args.config)
    if args.model_path:
        container.config.model.path.from_value(args.model_path)

    chat_service = container.chat_service()

    print(f"Loading SNN model from: {container.config.model.path()}")
    print("✅ SNN model loaded successfully via DI Container.")

    # Gradio Blocks を使用してUIを構築
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky")) as demo:
        gr.Markdown(
            """
            # 🤖 SNN-based AI Chat Prototype
            進化したBreakthroughSNNモデルとのリアルタイム対話。
            右側のパネルには、推論時間や総スパイク数（エネルギー効率の代理指標）などの統計情報がリアルタイムで表示されます。
            """
        )
        
        initial_stats_md = """
        **Inference Time:** `N/A`
        **Tokens/Second:** `N/A`
        ---
        **Total Spikes:** `N/A`
        **Spikes/Second:** `N/A`
        """

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="SNN Chat", height=500, avatar_images=("user.png", "assistant.png"))
            with gr.Column(scale=1):
                stats_display = gr.Markdown(value=initial_stats_md, label="📊 Inference Stats")

        with gr.Row():
            msg_textbox = gr.Textbox(
                show_label=False,
                placeholder="SNNモデルに話しかける...",
                container=False,
                scale=6,
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear", scale=1)

        def clear_all():
            """チャット履歴、テキストボックス、統計表示をクリアする"""
            return [], "", initial_stats_md

        # `submit` アクションの定義
        submit_event = msg_textbox.submit(
            fn=chat_service.stream_response,
            inputs=[msg_textbox, chatbot],
            outputs=[chatbot, stats_display]
        )
        submit_event.then(fn=lambda: "", inputs=None, outputs=msg_textbox)
        
        button_submit_event = submit_btn.click(
            fn=chat_service.stream_response,
            inputs=[msg_textbox, chatbot],
            outputs=[chatbot, stats_display]
        )
        button_submit_event.then(fn=lambda: "", inputs=None, outputs=msg_textbox)

        # `clear` アクションの定義
        clear_btn.click(
            fn=clear_all,
            inputs=None,
            outputs=[chatbot, msg_textbox, stats_display],
            queue=False # このアクションはキューに入れる必要がない
        )

    # Webアプリケーションの起動
    print("\nStarting Gradio web server...")
    print(f"Please open http://{container.config.app.server_name()}:{container.config.app.server_port()} in your browser.")
    demo.launch(
        server_name=container.config.app.server_name(),
        server_port=container.config.app.server_port(),
    )

if __name__ == "__main__":
    main()

