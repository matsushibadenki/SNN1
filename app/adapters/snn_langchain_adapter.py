# matsushibadenki/snn/app/adapters/snn_langchain_adapter.py
# SNNモデルをLangChainのLLMインターフェースに適合させるアダプタ
#
# 機能:
# - LangChainのカスタムLLMとしてSNNモデルをラップする。
# - これにより、SNNをLangChainエコシステム（Chain, Agentなど）で利用可能になる。

from langchain_core.language_models.llms import LLM
from typing import Any, List, Mapping, Optional
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
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            pass # 将来的にstopシーケンスをサポート
        
        max_len = self.snn_engine.config.get("max_len", 50)
        response = self.snn_engine.generate(prompt, max_len=max_len)
        return response.replace(prompt, "").strip()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """モデルの識別パラメータを返す。"""
        return {"model_path": self.snn_engine.config.get("path")}