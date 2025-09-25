# matsushibadenki/snn/app/containers.py
# DIコンテナの定義ファイル
# 
# 機能:
# - プロジェクト全体の依存関係を一元管理する。
# - 設定ファイルに基づいてオブジェクトを生成・設定する。
# - 学習用とアプリ用のコンテナを分離し、関心を分離。
# - 独自Vocabularyを廃止し、Hugging Face Tokenizerに全面的に移行。
# - トークナイザの読み込み元をdistillation設定から共通設定に変更。
# - 損失関数にpad_idではなくtokenizerプロバイダを渡すように修正し、依存関係の解決を遅延させる。

from dependency_injector import containers, providers
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer, AutoModelForCausalLM

# プロジェクト内モジュールのインポート
from snn_research.core.snn_core import BreakthroughSNN
from snn_research.deployment import SNNInferenceEngine
from snn_research.training.losses import CombinedLoss, DistillationLoss
from snn_research.training.trainers import BreakthroughTrainer, DistillationTrainer
from .services.chat_service import ChatService
from .adapters.snn_langchain_adapter import SNNLangChainAdapter

def _calculate_t_max(epochs: int, warmup_epochs: int) -> int:
    """学習率スケジューラのT_maxを計算する"""
    return max(1, epochs - warmup_epochs) # 1未満にならないようにする

class TrainingContainer(containers.DeclarativeContainer):
    """学習に関連するオブジェクトの依存関係を管理するコンテナ。"""
    config = providers.Configuration()

    # --- トークナイザ ---
    # 共通設定からトークナイザ名を読み込む
    tokenizer = providers.Factory(
        AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=config.data.tokenizer_name
    )

    # --- モデル関連 ---
    snn_model = providers.Factory(
        BreakthroughSNN,
        vocab_size=tokenizer.provided.vocab_size,
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
    
    # --- 学習率スケジューラ (ウォームアップ付き) ---
    warmup_scheduler = providers.Factory(
        LinearLR,
        optimizer=optimizer,
        start_factor=1e-3,
        total_iters=config.training.warmup_epochs,
    )
    
    main_scheduler_t_max = providers.Factory(
        _calculate_t_max,
        epochs=config.training.epochs.as_(int),
        warmup_epochs=config.training.warmup_epochs.as_(int),
    )

    main_scheduler = providers.Factory(
        CosineAnnealingLR,
        optimizer=optimizer,
        T_max=main_scheduler_t_max,
    )

    scheduler = providers.Factory(
        SequentialLR,
        optimizer=optimizer,
        schedulers=providers.List(warmup_scheduler, main_scheduler),
        milestones=providers.List(config.training.warmup_epochs),
    )
    
    # --- 損失関数 ---
    standard_loss = providers.Factory(
        CombinedLoss,
        ce_weight=config.training.loss.ce_weight,
        spike_reg_weight=config.training.loss.spike_reg_weight,
        tokenizer=tokenizer,
    )
    distillation_loss = providers.Factory(
        DistillationLoss,
        ce_weight=config.training.distillation.loss.ce_weight,
        distill_weight=config.training.distillation.loss.distill_weight,
        spike_reg_weight=config.training.distillation.loss.spike_reg_weight,
        temperature=config.training.distillation.loss.temperature,
        tokenizer=tokenizer,
    )
    
    # --- 蒸留用教師モデル ---
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
        max_len=config.inference.max_len,
    )
    
    # --- LangChainアダプタ ---
    langchain_adapter = providers.Factory(
        SNNLangChainAdapter,
        snn_engine=snn_inference_engine,
    )
