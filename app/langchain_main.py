# matsushibadenki/snn/app/langchain_main.py
# LangChainと連携したSNNチャットアプリケーション
#
# 機能:
# - ロードマップ フェーズ2「2.4. プロトタイプ開発」に対応。
# - DIコンテナからSNNLangChainAdapterを取得。
# - LangChainのPromptTemplateとLLMChainを利用して、より構造化された応答を生成するデモ。
# - Gradio Blocksを使用して、チャット画面とリアルタイム統計情報パネルを持つUIを構築。
# - ストリーミング応答に対応し、SNNの計算統計をリアルタイムで表示。

import gradio as gr
import argparse
import sys
import time
from pathlib import Path
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Iterator, Tuple, List

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

    # アバター用のSVGアイコンを定義
    user_avatar_svg = r"""
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-user"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
    """
    assistant_avatar_svg = r"""
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-zap"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
    """

    def stream_response(message: str, history: List[List[str]]) -> Iterator[Tuple[List[List[str]], str]]:
        """GradioのBlocks UIのために、チャット履歴と統計情報をストリーミング生成する。"""
        history.append([message, ""])
        
        print("-" * 30)
        print(f"Input question to LLMChain: {message}")
        
        start_time = time.time()
        
        full_response = ""
        token_count = 0
        
        # LangChainのstreamメソッドを使用
        for chunk in llm_chain.stream({"question": message}):
            # chunkは辞書形式なので、テキスト部分を取り出す
            response_piece = chunk.get('text', '')
            full_response += response_piece
            token_count += 1
            history[-1][1] = full_response
            
            duration = time.time() - start_time
            # LangChainアダプタ経由でSNNエンジンの統計情報を取得
            stats = snn_llm.snn_engine.last_inference_stats
            total_spikes = stats.get("total_spikes", 0)
            spikes_per_second = total_spikes / duration if duration > 0 else 0
            tokens_per_second = token_count / duration if duration > 0 else 0

            stats_md = f"""
            **Inference Time:** `{duration:.2f} s`
            **Tokens/Second:** `{tokens_per_second:.2f}`
            ---
            **Total Spikes:** `{total_spikes:,.0f}`
            **Spikes/Second:** `{spikes_per_second:,.0f}`
            """
            
            yield history, stats_md

        # Final log to console
        duration = time.time() - start_time
        stats = snn_llm.snn_engine.last_inference_stats
        total_spikes = stats.get("total_spikes", 0)
        print(f"\nGenerated response: {full_response.strip()}")
        print(f"Inference time: {duration:.4f} seconds")
        print(f"Total spikes: {total_spikes:,.0f}")
        print("-" * 30)

    # Gradio Blocks を使用してUIを構築
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="lime")) as demo:
        gr.Markdown(
            """
            # 🤖 SNN + LangChain Prototype
            SNNモデルをLangChain経由で利用するプロトタイプ。
            右側のパネルには、推論時間や総スパイク数などの統計情報がリアルタイムで表示されます。
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
                chatbot = gr.Chatbot(label="SNN+LangChain Chat", height=500, avatar_images=(user_avatar_svg, assistant_avatar_svg))
            with gr.Column(scale=1):
                stats_display = gr.Markdown(value=initial_stats_md, label="📊 Inference Stats")

        with gr.Row():
            msg_textbox = gr.Textbox(
                show_label=False,
                placeholder="質問を入力...",
                container=False,
                scale=6,
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear", scale=1)

        gr.Markdown("<footer><p>© 2025 SNN System Design Project. All rights reserved.</p></footer>")

        def clear_all():
            """チャット履歴、テキストボックス、統計表示をクリアする"""
            return [], "", initial_stats_md

        # `submit` アクションの定義
        submit_event = msg_textbox.submit(
            fn=stream_response,
            inputs=[msg_textbox, chatbot],
            outputs=[chatbot, stats_display]
        )
        submit_event.then(fn=lambda: "", inputs=None, outputs=msg_textbox)
        
        button_submit_event = submit_btn.click(
            fn=stream_response,
            inputs=[msg_textbox, chatbot],
            outputs=[chatbot, stats_display]
        )
        button_submit_event.then(fn=lambda: "", inputs=None, outputs=msg_textbox)

        # `clear` アクションの定義
        clear_btn.click(
            fn=clear_all,
            inputs=None,
            outputs=[chatbot, msg_textbox, stats_display],
            queue=False
        )
    
    # Webアプリケーションの起動
    server_port = container.config.app.server_port() + 1 # ポートが衝突しないように+1する
    print("\nStarting Gradio web server for LangChain app...")
    print(f"Please open http://{container.config.app.server_name()}:{server_port} in your browser.")
    demo.launch(
        server_name=container.config.app.server_name(),
        server_port=server_port,
    )

if __name__ == "__main__":
    main()

