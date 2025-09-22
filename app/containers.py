# matsushibadenki/snn/app/containers.py
# DIコンテナの定義ファイル
# 
# 機能:
# - プロジェクト全体の依存関係を一元管理する。
# - 設定ファイルに基づいてオブジェクトを生成・設定する。
# - 学習用とアプリ用のコンテナを分離し、関心を分離。
# - 学習安定化のため、ウォームアップ付きの学習率スケジューラを導入。

from dependency_injector import containers, providers
from torch.optim import AdamW
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
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
    
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    # --- 学習率スケジューラ (ウォームアップ付き) ---
    warmup_scheduler = providers.Factory(
        LinearLR,
        optimizer=optimizer,
        start_factor=1e-3, # 開始時の学習率を通常時の0.001倍にする
        total_iters=config.training.warmup_epochs,
    )
    
    main_scheduler = providers.Factory(
        CosineAnnealingLR,
        optimizer=optimizer,
        T_max=config.training.epochs - config.training.warmup_epochs,
    )

    scheduler = providers.Factory(
        SequentialLR,
        optimizer=optimizer,
        schedulers=providers.List(warmup_scheduler, main_scheduler),
        milestones=providers.List(config.training.warmup_epochs),
    )
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    
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

    # --- トレーナー定義 (静的に両方定義する) ---
    standard_trainer = providers.Factory(
        BreakthroughTrainer,
        criterion=standard_loss,
        grad_clip_norm=config.training.grad_clip_norm,
    )

    distillation_trainer = providers.Factory(
        DistillationTrainer,
        criterion=distillation_loss,
        teacher_model=teacher_model,
        grad_clip_norm=config.training.grad_clip_norm,
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
