"""The analysis figures render to PNG from synthetic artifact-shaped inputs."""

import os

from qwen3 import (
    plot_baseline_comparison,
    plot_accuracy_by_oov,
    plot_per_class_metrics,
)


def test_baseline_comparison_writes_png(tmp_path):
    entries = [
        {"name": "Majority", "point": 0.36, "lower": 0.34, "upper": 0.38},
        {"name": "Frozen probe", "point": 0.64, "lower": 0.61, "upper": 0.66},
        {"name": "TF-IDF + LogReg", "point": 0.93, "lower": 0.92, "upper": 0.94},
    ]
    out = tmp_path / "baseline_comparison.png"
    path = plot_baseline_comparison(entries, save_path=out)
    assert os.path.isfile(path)


def test_accuracy_by_oov_writes_png(tmp_path):
    buckets = [
        {"range": "0%", "count": 800, "accuracy": 0.70, "mean_oov": 0.0},
        {"range": "(0,10%]", "count": 500, "accuracy": 0.62, "mean_oov": 5.0},
        {"range": ">25%", "count": 100, "accuracy": 0.48, "mean_oov": 40.0},
    ]
    out = tmp_path / "accuracy_by_oov.png"
    path = plot_accuracy_by_oov(buckets, correlation=0.31, save_path=out)
    assert os.path.isfile(path)


def test_per_class_metrics_writes_png(tmp_path):
    metrics = {
        "negative_precision": 0.66, "negative_recall": 0.63, "negative_f1": 0.64,
        "positive_precision": 0.61, "positive_recall": 0.64, "positive_f1": 0.62,
    }
    cis = {
        "negative_precision": {"point": 0.66, "lower": 0.63, "upper": 0.69},
        "negative_recall": {"point": 0.63, "lower": 0.60, "upper": 0.66},
        "negative_f1": {"point": 0.64, "lower": 0.61, "upper": 0.67},
        "positive_precision": {"point": 0.61, "lower": 0.58, "upper": 0.64},
        "positive_recall": {"point": 0.64, "lower": 0.61, "upper": 0.67},
        "positive_f1": {"point": 0.62, "lower": 0.59, "upper": 0.65},
    }
    out = tmp_path / "per_class_metrics.png"
    path = plot_per_class_metrics(metrics, cis, save_path=out)
    assert os.path.isfile(path)
