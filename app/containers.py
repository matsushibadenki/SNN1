# matsushibadenki/snn/app/containers.py
# DIコンテナの定義ファイル
# 
# 機能:
# - プロジェクト全体の依存関係を一元管理する。
# - 設定ファイルに基づいてオブジェクトを生成・設定する。
# - 学習用とアプリ用のコンテナを分離し、関心を分離。

from dependency_injector import containers, providers
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM

# プロジェクト内モジュールのインポート
from snn_research.core.snn_core import BreakthroughSNN
from snn_research.deployment import SNNInferenceEngine
from snn_research.data.datasets import Vocabulary
from snn_research.training.losses import CombinedLoss, DistillationLoss
from snn_research.training.trainers import BreakthroughTrainer, DistillationTrainer
from .services.chat_service import ChatService
from .adapters.snn_langchain_adapter import SNNLangChainAdapter

class TrainingContainer(containers.DeclarativeContainer):
    """学習に関連するオブジェクトの依存関係を管理するコンテナ。"""
    config = providers.Configuration()

    # --- データ関連 ---
    vocabulary = providers.Factory(Vocabulary)

    # --- モデル関連 ---
    snn_model = providers.Factory(
        BreakthroughSNN,
        d_model=config.model.d_model,
        d_state=config.model.d_state,
        num_layers=config.model.num_layers,
        time_steps=config.model.time_steps,
        n_head=config.model.n_head,
    )

    # --- 学習コンポーネント ---
    optimizer = providers.Factory(
        AdamW,
        lr=config.training.learning_rate,
    )
    scheduler = providers.Factory(
        CosineAnnealingLR,
        T_max=config.training.epochs,
    )
    
    # --- 損失関数 ---
    standard_loss = providers.Factory(
        CombinedLoss,
        ce_weight=config.training.loss.ce_weight,
        spike_reg_weight=config.training.loss.spike_reg_weight,
        pad_id=0, # 後から設定
    )
    distillation_loss = providers.Factory(
        DistillationLoss,
        ce_weight=config.training.distillation.loss.ce_weight,
        distill_weight=config.training.distillation.loss.distill_weight,
        spike_reg_weight=config.training.distillation.loss.spike_reg_weight,
        temperature=config.training.distillation.loss.temperature,
        student_pad_id=0, # 後から設定
    )
    
    # --- 蒸留用教師モデル ---
    teacher_tokenizer = providers.Factory(
        AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=config.training.distillation.teacher_model
    )
    teacher_model = providers.Factory(
        AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=config.training.distillation.teacher_model
    )

    # --- トレーナー (Factoryパターンで学習タイプに応じて切り替え) ---
    @providers.provider
    def trainer(self, model, optimizer, scheduler, device, rank) -> providers.Provider:
        training_type = self.config.training.type()
        
        common_args = {
            "model": model, "optimizer": optimizer, "scheduler": scheduler,
            "device": device, "grad_clip_norm": self.config.training.grad_clip_norm(), "rank": rank,
        }
        
        if training_type == "distillation":
            return providers.Factory(
                DistillationTrainer,
                criterion=self.distillation_loss,
                teacher_model=self.teacher_model,
                **common_args
            )
        else: # standard or distributed
            return providers.Factory(
                BreakthroughTrainer,
                criterion=self.standard_loss,
                **common_args
            )


class AppContainer(containers.DeclarativeContainer):
    """GradioアプリやAPIなど、アプリケーション層の依存関係を管理するコンテナ。"""
    config = providers.Configuration()

    # --- 推論エンジン (Singletonで単一インスタンスを保証) ---
    snn_inference_engine = providers.Singleton(
        SNNInferenceEngine,
        model_path=config.model.path,
        device=config.inference.device,
    )
    
    # --- サービス ---
    chat_service = providers.Factory(
        ChatService,
        snn_engine=snn_inference_engine,
    )
    
    # --- LangChainアダプタ ---
    langchain_adapter = providers.Factory(
        SNNLangChainAdapter,
        snn_engine=snn_inference_engine,
    )