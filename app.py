# /path/to/your/project/app.py
# SNNモデルをバックエンドとして使用するリアルタイム対話AIのプロトタイプ
#
# 目的:
# - ロードマップ フェーズ2「2.4. プロトタイプ開発」に対応。
# - 開発したSNNモデルの対話能力を実際に体験できるWebアプリケーションを提供する。
#
# 実行方法:
# python app.py --model_path <学習済みモデルのパス>
# 例: python app.py --model_path snn_distilled_model.pth

import gradio as gr
import argparse
import time

from main import SNNInferenceEngine

# グローバル変数として推論エンジンを保持
inference_engine = None

def chat_function(message: str, history: list) -> str:
    """
    GradioのChatInterfaceに渡すためのメインのチャット処理関数。

    Args:
        message (str): ユーザーからの新しい入力メッセージ。
        history (list): これまでの対話履歴。Gradioによって管理される。
                        形式: [[user_msg_1, bot_msg_1], [user_msg_2, bot_msg_2], ...]

    Returns:
        str: SNNモデルが生成した応答テキスト。
    """
    if inference_engine is None:
        return "エラー: SNNモデルがロードされていません。アプリケーションを正しいモデルパスで起動してください。"

    # 対話履歴を連結して、モデルへの入力プロンプトを作成
    # シンプルな実装として、最新の数ターンのみを使用することも可能
    prompt = ""
    for user_msg, bot_msg in history:
        prompt += f"User: {user_msg}\nAssistant: {bot_msg}\n"
    prompt += f"User: {message}\nAssistant:"

    print("-" * 30)
    print(f"Input prompt to SNN:\n{prompt}")

    # SNNモデルで応答を生成
    start_time = time.time()
    generated_text = inference_engine.generate(prompt, max_len=50)
    duration = time.time() - start_time
    
    # "Assistant:" の部分を除去して整形
    response = generated_text.replace(prompt, "").strip()

    print(f"Generated response: {response}")
    print(f"Inference time: {duration:.4f} seconds")
    print("-" * 30)
    
    return response

def main(args):
    global inference_engine
    
    print(f"Loading SNN model from: {args.model_path}")
    try:
        inference_engine = SNNInferenceEngine(model_path=args.model_path)
        print("✅ SNN model loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at '{args.model_path}'.")
        print("Please provide a valid path to a trained model using the --model_path argument.")
        return

    # Gradioインターフェースの構築
    chatbot_interface = gr.ChatInterface(
        fn=chat_function,
        title="🤖 SNN-based AI Chat Prototype",
        description="進化したBreakthroughSNNモデルとのリアルタイム対話。メッセージを入力してEnterキーを押してください。",
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="SNNモデルに話しかける...", container=False, scale=7),
        retry_btn=None,
        undo_btn="削除",
        clear_btn="クリア",
    )

    # Webアプリケーションの起動
    print("\nStarting Gradio web server...")
    print("Please open the following URL in your browser:")
    chatbot_interface.launch(server_name="0.0.0.0", server_port=args.port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNNベース リアルタイム対話AI プロトタイプ")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="対話に使用する学習済みSNNモデルのパス (.pth)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Webインターフェースを起動するポート番号"
    )
    args = parser.parse_args()
    main(args)
