# matsushibadenki/snn/app/services/chat_service.py
# チャット機能のビジネスロジックを担うサービス
#
# 機能:
# - DIコンテナから推論エンジンを受け取る。
# - Gradioからの入力を処理し、整形して推論エンジンに渡す。
# - 推論結果をGradioが扱える形式で返す。

import time
from snn_research.deployment import SNNInferenceEngine

class ChatService:
    def __init__(self, snn_engine: SNNInferenceEngine, max_len: int):
        self.snn_engine = snn_engine
        self.max_len = max_len

    def handle_message(self, message: str, history: list) -> str:
        """
        GradioのChatInterfaceに渡すためのメインのチャット処理関数。
        """
        prompt = ""
        for user_msg, bot_msg in history:
            prompt += f"User: {user_msg}\nAssistant: {bot_msg}\n"
        prompt += f"User: {message}\nAssistant:"

        print("-" * 30)
        print(f"Input prompt to SNN:\n{prompt}")

        start_time = time.time()
        generated_text = self.snn_engine.generate(prompt, max_len=self.max_len)
        duration = time.time() - start_time
        
        response = generated_text.replace(prompt, "").strip()

        print(f"Generated response: {response}")
        print(f"Inference time: {duration:.4f} seconds")
        print("-" * 30)
        
        return response