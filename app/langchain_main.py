# matsushibadenki/snn/app/adapters/snn_langchain_adapter.py
# SNNモデルをLangChainのLLMインターフェースに適合させるアダプタ
#
# 機能:
# - LangChainのカスタムLLMとしてSNNモデルをラップする。
# - これにより、SNNをLangChainエコシステム（Chain, Agentなど）で利用可能になる。
# - ストリーミング応答をサポート (`_stream` メソッドを実装)。
# - mypyの型エラーを修正 (`Iterator[GenerationChunk]`を返すように変更)。

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
from typing import Any, List, Mapping, Optional, Iterator
from snn_research.deployment import SNNInferenceEngine

class SNNLangChainAdapter(LLM):
    """SNNInferenceEngineをラップするLangChainカスタムLLMクラス。"""
    
    snn_engine: SNNInferenceEngine

    @property
    def _llm_type(self) -> str:
        return "snn_breakthrough"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # _streamメソッドの結果（GenerationChunk）からテキストを抽出して結合
        return "".join(
            chunk.text for chunk in self._stream(prompt, stop, run_manager, **kwargs)
        )

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """SNNエンジンからテキストをストリーミングし、LangChainコールバックを呼び出す。"""
        max_len = self.snn_engine.config.get("max_len", 50)
        
        # SNNInferenceEngineのジェネレータからテキストチャンクを取得
        for chunk_text in self.snn_engine.generate(prompt, max_len=max_len, stop_sequences=stop):
            # テキストチャンクをGenerationChunkオブジェクトにラップしてyieldする
            chunk = GenerationChunk(text=chunk_text)
            yield chunk
            if run_manager:
                # コールバックには文字列を渡す
                run_manager.on_llm_new_token(chunk.text)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """モデルの識別パラメータを返す。"""
        return {"model_path": self.snn_engine.config.get("path")}

