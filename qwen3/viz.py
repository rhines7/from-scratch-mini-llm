"""
Figure generation for the pipeline.

All plots are rendered with a non-interactive backend and written as PNGs to the
figures/ directory so they can be regenerated headlessly (e.g. via
--figures-only) and committed. Each function is driven by committed artifacts or
in-memory arrays, never by re-running training.
"""

from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # Headless: no display needed on CI or servers
import matplotlib.pyplot as plt
import numpy as np

from .config import (
    SEQ_LENGTH_FIGURE_PATH,
    TRAINING_CURVES_FIGURE_PATH,
    CONFUSION_MATRIX_FIGURE_PATH,
    BASELINE_COMPARISON_FIGURE_PATH,
    OOV_ACCURACY_FIGURE_PATH,
    PER_CLASS_METRICS_FIGURE_PATH,
    FIGURES_DIR,
)


def _ensure_parent(path) -> str:
    """Create the figure's parent directory and return the path as a string."""
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def plot_sequence_length_distribution(
    lengths: Sequence[int],
    stats: Dict,
    save_path=SEQ_LENGTH_FIGURE_PATH
) -> str:
    """Histogram and cumulative-coverage plot of token lengths.

    The 95th-percentile marker is the recommended max_seq_len; the cumulative
    panel shows how much of the corpus a given length covers.
    """
    save_path = _ensure_parent(save_path)
    lengths_array = np.asarray(lengths)
    recommended = int(stats.get('percentile_95', np.percentile(lengths_array, 95)))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(lengths_array, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(stats['mean'], color='red', linestyle='--', label=f"Mean: {stats['mean']:.0f}")
    plt.axvline(stats['median'], color='green', linestyle='--', label=f"Median: {stats['median']:.0f}")
    plt.axvline(recommended, color='blue', linestyle='--', label=f"95th %ile: {recommended}")
    plt.xlabel('Sequence length (tokens)')
    plt.ylabel('Frequency')
    plt.title('Sequence length distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    sorted_lengths = np.sort(lengths_array)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
    plt.plot(sorted_lengths, cumulative)
    plt.axhline(95, color='red', linestyle='--', label='95% coverage')
    plt.axvline(recommended, color='blue', linestyle='--', label=f'Length: {recommended}')
    plt.xlabel('Sequence length (tokens)')
    plt.ylabel('Cumulative coverage (%)')
    plt.title('Cumulative coverage vs sequence length')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sequence-length figure to {save_path}")
    return save_path


def plot_training_curves(
    history: Dict,
    save_path=TRAINING_CURVES_FIGURE_PATH
) -> str:
    """Plot pre-training loss and perplexity against training step.

    Accepts the training_history dict (train arrays keyed by 'steps', validation
    arrays sampled at validation points). Validation points are spaced evenly
    across the recorded steps when their own step index is not stored.
    """
    save_path = _ensure_parent(save_path)

    steps = history.get('steps', [])
    train_loss = history.get('train_loss', [])
    train_ppl = history.get('train_perplexity', [])
    val_loss = history.get('val_loss', [])
    val_ppl = history.get('val_perplexity', [])

    # Approximate validation x-positions across the training-step axis
    def val_positions(n):
        if not steps or n == 0:
            return list(range(n))
        return list(np.linspace(steps[0], steps[-1], n))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if steps and train_loss:
        plt.plot(steps[:len(train_loss)], train_loss, label='Train loss', alpha=0.8)
    if val_loss:
        plt.plot(val_positions(len(val_loss)), val_loss,
                 label='Val loss', marker='o', linestyle='--')
    plt.xlabel('Training step')
    plt.ylabel('Loss')
    plt.title('Pre-training loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    if steps and train_ppl:
        plt.plot(steps[:len(train_ppl)], train_ppl, label='Train perplexity', alpha=0.8)
    if val_ppl:
        plt.plot(val_positions(len(val_ppl)), val_ppl,
                 label='Val perplexity', marker='o', linestyle='--')
    plt.xlabel('Training step')
    plt.ylabel('Perplexity')
    plt.title('Pre-training perplexity')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training-curves figure to {save_path}")
    return save_path


def plot_confusion_matrix(
    predictions: List[int],
    labels: List[int],
    save_path=CONFUSION_MATRIX_FIGURE_PATH,
    class_names: Optional[List[str]] = None
) -> str:
    """Render the sentiment confusion matrix from raw predictions and labels."""
    class_names = class_names or ['Negative', 'Positive']
    num_classes = len(class_names)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(labels, predictions):
        cm[true_label][pred_label] += 1

    return plot_confusion_matrix_from_counts(cm, save_path, class_names)


def plot_confusion_matrix_from_counts(
    cm,
    save_path=CONFUSION_MATRIX_FIGURE_PATH,
    class_names: Optional[List[str]] = None
) -> str:
    """Render a confusion matrix from a precomputed counts array (rows = true).

    Working from committed counts lets --figures-only rebuild the figure without
    shipping the full per-example prediction arrays.
    """
    save_path = _ensure_parent(save_path)
    class_names = class_names or ['Negative', 'Positive']
    cm = np.asarray(cm, dtype=int)
    num_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    # Anchor the color scale at zero so cell shade tracks absolute count,
    # keeping the white/black text-contrast threshold meaningful
    ax.imshow(cm, cmap='Blues', vmin=0)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Sentiment confusion matrix')

    # Annotate counts; switch text color for contrast on dark cells
    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center',
                    color='white' if cm[i][j] > threshold else 'black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion-matrix figure to {save_path}")
    return save_path


def _asymmetric_yerr(entries):
    """Build a 2xN yerr array (lower, upper offsets) from point/lower/upper dicts."""
    lower = [max(0.0, e["point"] - e["lower"]) for e in entries]
    upper = [max(0.0, e["upper"] - e["point"]) for e in entries]
    return np.array([lower, upper])


def plot_baseline_comparison(
    entries: List[Dict],
    metric_label: str = "Macro-F1",
    save_path=BASELINE_COMPARISON_FIGURE_PATH
) -> str:
    """Bar chart comparing methods on one metric with 95% bootstrap CIs.

    entries is an ordered list of {"name", "point", "lower", "upper"}. Plotting
    the frozen probe between the majority floor and the lexical ceiling, with
    error bars, is what makes the "modest but honest" result legible at a glance.
    """
    save_path = _ensure_parent(save_path)

    names = [e["name"] for e in entries]
    points = [e["point"] for e in entries]
    yerr = _asymmetric_yerr(entries)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    ax.bar(x, points, yerr=yerr, capsize=6, alpha=0.85)

    for xi, point in zip(x, points):
        ax.text(xi, point + 0.02, f"{point:.3f}", ha='center', va='bottom')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel(metric_label)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{metric_label} by method (95% bootstrap CI)")
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved baseline-comparison figure to {save_path}")
    return save_path


def plot_accuracy_by_oov(
    buckets: List[Dict],
    correlation: Optional[float] = None,
    save_path=OOV_ACCURACY_FIGURE_PATH
) -> str:
    """Bar chart of test accuracy across tokenizer OOV-rate buckets.

    Visualizes the tokenizer domain gap as a performance ceiling: if accuracy
    falls as OOV rises, the vocabulary (not model capacity) is the lever. Empty
    buckets are dropped; per-bucket counts are annotated for context.
    """
    save_path = _ensure_parent(save_path)

    usable = [b for b in buckets if b.get("count") and b.get("accuracy") is not None]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(usable))
    accuracies = [b["accuracy"] for b in usable]
    ax.bar(x, accuracies, alpha=0.85)

    for xi, bucket in zip(x, usable):
        ax.text(xi, bucket["accuracy"] + 0.02, f"n={bucket['count']}",
                ha='center', va='bottom')

    ax.set_xticks(x)
    ax.set_xticklabels([b["range"] for b in usable])
    ax.set_xlabel("Sentence OOV rate")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    title = "Accuracy by tokenizer OOV rate"
    if correlation is not None:
        title += f" (OOV vs error r = {correlation:.2f})"
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved OOV-accuracy figure to {save_path}")
    return save_path


def plot_per_class_metrics(
    metrics: Dict,
    cis: Optional[Dict] = None,
    save_path=PER_CLASS_METRICS_FIGURE_PATH
) -> str:
    """Grouped bars of precision/recall/F1 per class, with CI error bars if given.

    Complements the confusion matrix by exposing the negative-vs-positive
    asymmetry directly, and the bootstrap CIs show whether that gap is real.
    """
    save_path = _ensure_parent(save_path)

    metric_names = ["precision", "recall", "f1"]
    classes = ["negative", "positive"]
    x = np.arange(len(metric_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, class_name in zip((-width / 2, width / 2), classes):
        values = [metrics[f"{class_name}_{m}"] for m in metric_names]
        yerr = None
        if cis is not None:
            lower, upper = [], []
            for m in metric_names:
                band = cis.get(f"{class_name}_{m}")
                point = metrics[f"{class_name}_{m}"]
                lower.append(max(0.0, point - band["lower"]) if band else 0.0)
                upper.append(max(0.0, band["upper"] - point) if band else 0.0)
            yerr = np.array([lower, upper])
        ax.bar(x + offset, values, width, yerr=yerr, capsize=5,
               alpha=0.85, label=class_name.capitalize())

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metric_names])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class sentiment metrics (95% bootstrap CI)")
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class-metrics figure to {save_path}")
    return save_path
