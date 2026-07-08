"""
Canonical paths and shared constants for the pipeline.

All stage modules resolve their inputs and outputs through the values defined
here so the repo has one place that describes where checkpoints, artifacts, and
figures live. Stage-specific hyperparameter dataclasses stay in their own
modules (DataConfig, TrainingConfig, GenerationConfig, SentimentConfig) and are
re-exported from the package __init__ and the entry script.

Paths are absolute (anchored at the repo root) so commands behave the same
regardless of the working directory they are launched from.
"""

from pathlib import Path

# Repo root is the parent of this package directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Top-level directories
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
SENTIMENT_CHECKPOINT_DIR = PROJECT_ROOT / "sentiment_checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

# Reproducibility
DEFAULT_SEED = 42

# Regenerated tokenizer vocabulary (rebuilt during data prep, not committed)
TOKENIZER_VOCAB_PATH = PROJECT_ROOT / "qwen3_tokenizer_vocab.json"

# Raw data locations (not committed; see data/README.md)
EMOTIONS_CSV_PATH = DATA_DIR / "emotions_classified.csv"

# Pre-training checkpoints and raw history
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
TRAINING_HISTORY_PATH = LOG_DIR / "training_history.json"

# Sentiment checkpoints and raw results
BEST_SENTIMENT_MODEL_PATH = SENTIMENT_CHECKPOINT_DIR / "best_sentiment_model.pt"
SENTIMENT_HISTORY_PATH = SENTIMENT_CHECKPOINT_DIR / "sentiment_training_history.json"
SENTIMENT_RESULTS_PATH = SENTIMENT_CHECKPOINT_DIR / "final_results.json"

# Committed summary artifacts (backing the README/report metrics)
DATASET_STATS_PATH = ARTIFACTS_DIR / "dataset_statistics.json"
PRETRAIN_HISTORY_ARTIFACT = ARTIFACTS_DIR / "pretrain_history.json"
SENTIMENT_METRICS_ARTIFACT = ARTIFACTS_DIR / "sentiment_metrics.json"
MODEL_SUMMARY_ARTIFACT = ARTIFACTS_DIR / "model_summary.json"
# Raw training-set token lengths, committed so --figures-only can rebuild the
# sequence-length distribution without re-downloading the corpus
SEQUENCE_LENGTHS_ARTIFACT = ARTIFACTS_DIR / "sequence_lengths.json"

# Baselines and analysis artifacts that contextualize the frozen-probe result:
# a majority-class and lexical baseline plus the tokenizer OOV rate on the
# sentiment domain, the fine-tuned comparison, curated generation samples, and a
# handful of misclassified examples for error analysis.
BASELINE_METRICS_ARTIFACT = ARTIFACTS_DIR / "baseline_metrics.json"
GENERATION_SAMPLES_ARTIFACT = ARTIFACTS_DIR / "generation_samples.json"
SENTIMENT_FINETUNED_METRICS_ARTIFACT = ARTIFACTS_DIR / "sentiment_finetuned_metrics.json"
SENTIMENT_ERRORS_ARTIFACT = ARTIFACTS_DIR / "sentiment_errors.json"
# Per-sentence OOV rate versus classification correctness: tests whether the
# tokenizer domain gap (not model capacity) is what caps the probe.
SENTIMENT_OOV_ERROR_ARTIFACT = ARTIFACTS_DIR / "sentiment_oov_error.json"

# Committed figures
SEQ_LENGTH_FIGURE_PATH = FIGURES_DIR / "sequence_length_analysis.png"
TRAINING_CURVES_FIGURE_PATH = FIGURES_DIR / "training_curves.png"
CONFUSION_MATRIX_FIGURE_PATH = FIGURES_DIR / "confusion_matrix_sentiment.png"
# Analysis figures assembled from the committed metric artifacts
BASELINE_COMPARISON_FIGURE_PATH = FIGURES_DIR / "baseline_comparison.png"
OOV_ACCURACY_FIGURE_PATH = FIGURES_DIR / "accuracy_by_oov.png"
PER_CLASS_METRICS_FIGURE_PATH = FIGURES_DIR / "per_class_metrics.png"


def ensure_directories() -> None:
    """Create the writable output directories if they do not already exist."""
    for directory in (
        FIGURES_DIR,
        ARTIFACTS_DIR,
        CHECKPOINT_DIR,
        SENTIMENT_CHECKPOINT_DIR,
        LOG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
