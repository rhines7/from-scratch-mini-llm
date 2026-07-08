"""
Qwen3-small: a from-scratch decoder-only language model with a downstream
sentiment classifier.

The package exposes every stage of the pipeline (architecture, tokenizer, data
preparation, pre-training, generation, sentiment classification) plus the CLI
orchestration. The thin qwen3_pipeline.py entry script re-exports this same API,
so `import qwen3` and `import qwen3_pipeline` are interchangeable.
"""

from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    FIGURES_DIR,
    ARTIFACTS_DIR,
    CHECKPOINT_DIR,
    SENTIMENT_CHECKPOINT_DIR,
    LOG_DIR,
    DEFAULT_SEED,
    ensure_directories,
)

from .architecture import (
    Qwen3Config,
    RMSNorm,
    RotaryPositionEmbedding,
    SwiGLU,
    MultiHeadAttention,
    Qwen3TransformerBlock,
    Qwen3Model,
    create_qwen3_small,
)

from .tokenizer import (
    Qwen3SmallTokenizer,
    create_tokenizer_from_texts,
)

from .data import (
    DataConfig,
    TinyStoriesDataset,
    load_tiny_stories,
    split_dataset,
    analyze_sequence_lengths,
    prepare_tiny_stories_dataset,
    test_dataloader,
)

from .pretrain import (
    TrainingConfig,
    default_model_config,
    get_cosine_schedule_with_warmup,
    compute_loss,
    validate,
    save_checkpoint,
    load_checkpoint,
    train_qwen3_small,
)

from .generate import (
    GenerationConfig,
    apply_repetition_penalty,
    apply_no_repeat_ngram,
    top_k_filtering,
    top_p_filtering,
    sample_next_token,
    generate,
    batch_generate,
    test_generation_quality,
    compare_sampling_strategies,
    load_model_and_tokenizer,
    interactive_generate,
)

from .downstream import (
    SentimentConfig,
    EmotionsDataset,
    load_emotions_data,
    split_data,
    SentimentClassifier,
    compute_metrics,
    confusion_counts,
    collect_misclassified,
    bootstrap_metric_cis,
    oov_error_analysis,
    evaluate,
    train_sentiment_classifier,
    evaluate_sentiment_test,
)

from .baselines import run_baselines

from .artifacts import (
    write_model_summary,
    write_pretrain_history,
    write_sentiment_metrics,
    write_finetuned_metrics,
    write_sentiment_errors,
    write_sentiment_oov_error,
    write_generation_samples,
    write_baseline_metrics,
)

from .viz import (
    plot_sequence_length_distribution,
    plot_training_curves,
    plot_confusion_matrix,
    plot_confusion_matrix_from_counts,
    plot_baseline_comparison,
    plot_accuracy_by_oov,
    plot_per_class_metrics,
)

from .pipeline import (
    main,
    build_configs,
    run_architecture,
    run_data_prep,
    run_pretrain,
    run_generation,
    run_sentiment,
    run_finetune,
    regenerate_figures,
)

__all__ = [
    # config
    "PROJECT_ROOT", "DATA_DIR", "FIGURES_DIR", "ARTIFACTS_DIR", "CHECKPOINT_DIR",
    "SENTIMENT_CHECKPOINT_DIR", "LOG_DIR", "DEFAULT_SEED", "ensure_directories",
    # architecture
    "Qwen3Config", "RMSNorm", "RotaryPositionEmbedding", "SwiGLU",
    "MultiHeadAttention", "Qwen3TransformerBlock", "Qwen3Model", "create_qwen3_small",
    # tokenizer
    "Qwen3SmallTokenizer", "create_tokenizer_from_texts",
    # data
    "DataConfig", "TinyStoriesDataset", "load_tiny_stories", "split_dataset",
    "analyze_sequence_lengths", "prepare_tiny_stories_dataset", "test_dataloader",
    # pretrain
    "TrainingConfig", "default_model_config", "get_cosine_schedule_with_warmup",
    "compute_loss", "validate", "save_checkpoint", "load_checkpoint", "train_qwen3_small",
    # generate
    "GenerationConfig", "apply_repetition_penalty", "apply_no_repeat_ngram",
    "top_k_filtering", "top_p_filtering", "sample_next_token", "generate",
    "batch_generate", "test_generation_quality", "compare_sampling_strategies",
    "load_model_and_tokenizer", "interactive_generate",
    # downstream
    "SentimentConfig", "EmotionsDataset", "load_emotions_data", "split_data",
    "SentimentClassifier", "compute_metrics", "confusion_counts",
    "collect_misclassified", "bootstrap_metric_cis", "oov_error_analysis",
    "evaluate", "train_sentiment_classifier", "evaluate_sentiment_test",
    # baselines
    "run_baselines",
    # artifacts
    "write_model_summary", "write_pretrain_history", "write_sentiment_metrics",
    "write_finetuned_metrics", "write_sentiment_errors", "write_sentiment_oov_error",
    "write_generation_samples", "write_baseline_metrics",
    # viz
    "plot_sequence_length_distribution", "plot_training_curves",
    "plot_confusion_matrix", "plot_confusion_matrix_from_counts",
    "plot_baseline_comparison", "plot_accuracy_by_oov", "plot_per_class_metrics",
    # pipeline
    "main", "build_configs", "run_architecture", "run_data_prep", "run_pretrain",
    "run_generation", "run_sentiment", "run_finetune", "regenerate_figures",
]

__version__ = "1.0.0"
