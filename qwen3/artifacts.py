"""
Writers for the committed summary artifacts under artifacts/.

These functions are the single place that produces the JSON files backing the
README and report. Keeping them here (rather than inline in each stage) means a
full pipeline run regenerates every committed artifact deterministically instead
of relying on a manual trimming step. Large per-example arrays (predictions,
labels) are dropped before writing so the committed files stay small and stable.

The module intentionally depends only on the standard library and config paths,
so importing it never pulls in torch, sklearn, or matplotlib.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import (
    ARTIFACTS_DIR,
    MODEL_SUMMARY_ARTIFACT,
    PRETRAIN_HISTORY_ARTIFACT,
    SENTIMENT_METRICS_ARTIFACT,
    SENTIMENT_FINETUNED_METRICS_ARTIFACT,
    SENTIMENT_ERRORS_ARTIFACT,
    SENTIMENT_OOV_ERROR_ARTIFACT,
    GENERATION_SAMPLES_ARTIFACT,
    BASELINE_METRICS_ARTIFACT,
    TRAINING_HISTORY_PATH,
)

# Per-example arrays are useful at runtime (confusion matrix rendering) but are
# not committed; the confusion counts summarize them compactly.
_TRIMMED_METRIC_KEYS = ("predictions", "labels")


def _write_json(path, data: Dict) -> str:
    """Write data as indented JSON, creating the artifacts directory if needed."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved artifact {path.name} to {path}")
    return str(path)


def _trim_metrics(metrics: Dict) -> Dict:
    """Drop large per-example arrays so only summary scalars are committed."""
    return {k: v for k, v in metrics.items() if k not in _TRIMMED_METRIC_KEYS}


def write_model_summary(model, path=MODEL_SUMMARY_ARTIFACT) -> str:
    """Record realized parameter counts and the architecture configuration.

    Sourced from model.count_parameters() so the committed count always matches
    the instantiated model rather than a hand-copied number.
    """
    counts = model.count_parameters()
    config = model.config

    summary = {
        "parameters_total": counts["actual_total"],
        "parameters_trainable": counts["actual_trainable"],
        "parameters_millions": counts["actual_millions"],
        "expected_total": counts["expected_total"],
        "difference": counts["difference"],
        "config": {
            "vocab_size": config.vocab_size,
            "d_model": config.d_model,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "intermediate_size": config.intermediate_size,
            "max_seq_len": config.max_seq_len,
            "tie_embeddings": config.tie_embeddings,
        },
    }
    return _write_json(path, summary)


def write_pretrain_history(source_path=TRAINING_HISTORY_PATH,
                           path=PRETRAIN_HISTORY_ARTIFACT) -> Optional[str]:
    """Copy the raw training history into a committed artifact plus a summary.

    The curve arrays are preserved because --figures-only rebuilds the loss and
    perplexity plots from them; a small summary block records the headline final
    and best values the README/report quote.
    """
    source_path = Path(source_path)
    if not source_path.exists():
        print(f"No training history at {source_path}; skipping pretrain_history artifact.")
        return None

    with open(source_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    def _last(seq):
        return seq[-1] if seq else None

    val_loss = history.get("val_loss", [])
    history["summary"] = {
        "total_steps": _last(history.get("steps", [])),
        "final_train_loss": _last(history.get("train_loss", [])),
        "final_train_perplexity": _last(history.get("train_perplexity", [])),
        "final_val_loss": _last(val_loss),
        "final_val_perplexity": _last(history.get("val_perplexity", [])),
        "best_val_loss": min(val_loss) if val_loss else None,
    }
    return _write_json(path, history)


def write_sentiment_metrics(test_metrics: Dict,
                            confusion_matrix: Sequence[Sequence[int]],
                            best_val_f1: float,
                            path=SENTIMENT_METRICS_ARTIFACT) -> str:
    """Commit trimmed sentiment test metrics plus confusion counts.

    Per-class precision/recall/F1 are already present in test_metrics; the raw
    prediction/label arrays are dropped and replaced by the confusion counts so
    the confusion figure can be rebuilt via --figures-only.
    """
    payload = _trim_metrics(test_metrics)
    payload["best_val_macro_f1"] = best_val_f1
    payload["confusion_matrix"] = [list(map(int, row)) for row in confusion_matrix]
    return _write_json(path, payload)


def write_finetuned_metrics(test_metrics: Dict,
                            confusion_matrix: Sequence[Sequence[int]],
                            best_val_f1: float,
                            path=SENTIMENT_FINETUNED_METRICS_ARTIFACT) -> str:
    """Commit the fine-tuned (unfrozen base) comparison metrics."""
    return write_sentiment_metrics(test_metrics, confusion_matrix, best_val_f1, path=path)


def write_sentiment_errors(examples: List[Dict],
                           path=SENTIMENT_ERRORS_ARTIFACT) -> str:
    """Commit a small sample of misclassified test examples for error analysis."""
    payload = {"count": len(examples), "examples": examples}
    return _write_json(path, payload)


def write_sentiment_oov_error(analysis: Dict,
                              path=SENTIMENT_OOV_ERROR_ARTIFACT) -> str:
    """Commit the per-sentence OOV-rate versus correctness analysis."""
    return _write_json(path, analysis)


def write_generation_samples(results: Dict,
                             path=GENERATION_SAMPLES_ARTIFACT) -> str:
    """Commit curated generation samples (decoding config plus prompt/output pairs)."""
    return _write_json(path, results)


def write_baseline_metrics(baseline_data: Dict,
                           path=BASELINE_METRICS_ARTIFACT) -> str:
    """Commit the majority-class and lexical baselines plus the OOV analysis."""
    return _write_json(path, baseline_data)
