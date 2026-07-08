"""
Command-line orchestration for the Qwen3-small pipeline.

Stages run in order: architecture demo, data preparation, pre-training,
generation, and sentiment classification. The full run reproduces the committed
results; --quick shrinks every stage for a fast smoke test, and the remaining
flags target individual phases or figure regeneration.

Numbered phases (for --phase / --through):
    1  architecture demo
    2  data preparation (tokenizer + dataset + sequence-length figure)
    3  pre-training
    4  generation
    5  sentiment classification
"""

import argparse
import json
from typing import Optional

import torch

from .architecture import Qwen3Config, Qwen3Model
from .data import DataConfig, prepare_tiny_stories_dataset, test_dataloader
from .pretrain import TrainingConfig, train_qwen3_small, default_model_config
from .generate import (
    GenerationConfig,
    load_model_and_tokenizer,
    test_generation_quality,
    compare_sampling_strategies,
    interactive_generate,
)
from .downstream import (
    SentimentConfig,
    train_sentiment_classifier,
    evaluate_sentiment_test,
)
from .baselines import run_baselines
from . import viz
from . import artifacts
from .config import (
    DEFAULT_SEED,
    BEST_MODEL_PATH,
    TRAINING_HISTORY_PATH,
    PRETRAIN_HISTORY_ARTIFACT,
    SENTIMENT_METRICS_ARTIFACT,
    SENTIMENT_FINETUNED_METRICS_ARTIFACT,
    SENTIMENT_OOV_ERROR_ARTIFACT,
    BASELINE_METRICS_ARTIFACT,
    SEQUENCE_LENGTHS_ARTIFACT,
    ensure_directories,
)

PHASE_NAMES = {
    1: "architecture",
    2: "data preparation",
    3: "pre-training",
    4: "generation",
    5: "sentiment classification",
}


# ---------------------------------------------------------------------------
# Config builders: one canonical set that reproduces the committed run, and a
# shrunken set for --quick smoke tests.
# ---------------------------------------------------------------------------

def build_configs(quick: bool = False):
    """Return (DataConfig, TrainingConfig, SentimentConfig) for the chosen mode."""
    if quick:
        data_config = DataConfig(
            num_stories=300,
            max_seq_len=128,
            batch_size=8,
            random_seed=DEFAULT_SEED,
        )
        # Tiny model so the smoke test finishes quickly on CPU
        model_config = Qwen3Config(
            vocab_size=data_config.vocab_size,  # replaced after tokenizer builds
            d_model=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            intermediate_size=256,
            max_seq_len=data_config.max_seq_len,
        )
        training_config = TrainingConfig(
            model_config=None,  # rebuilt against the real vocab inside training
            data_config=data_config,
            num_epochs=1,
            warmup_steps=10,
            validate_every_n_steps=20,
            save_every_n_steps=40,
            log_every_n_steps=10,
        )
        training_config._quick_model = model_config  # stash for the runner
        sentiment_config = SentimentConfig(
            num_epochs=2,
            eval_every_n_steps=20,
        )
        return data_config, training_config, sentiment_config

    data_config = DataConfig()  # 10k stories, 256 tokens (committed defaults)
    training_config = TrainingConfig(data_config=data_config)
    sentiment_config = SentimentConfig()
    return data_config, training_config, sentiment_config


# ---------------------------------------------------------------------------
# Individual phases
# ---------------------------------------------------------------------------

def run_architecture(quick: bool = False) -> Qwen3Model:
    """Phase 1: instantiate the canonical model and report its parameter count."""
    print("\n" + "#" * 70)
    print("# Phase 1: architecture")
    print("#" * 70)

    _, training_config, _ = build_configs(quick)
    max_seq_len = training_config.data_config.max_seq_len
    if quick:
        config = training_config._quick_model
    else:
        config = default_model_config(vocab_size=7392, max_seq_len=max_seq_len)

    model = Qwen3Model(config)
    counts = model.count_parameters()
    print(f"Total parameters: {counts['actual_total']:,} "
          f"({counts['actual_millions']:.2f}M)")

    # Only the canonical model backs the committed summary; the tiny --quick
    # model must not overwrite it.
    if not quick:
        artifacts.write_model_summary(model)
    return model


def run_data_prep(data_config: Optional[DataConfig] = None,
                  write_artifacts: bool = True):
    """Phase 2: build the tokenizer and dataset, and save the sequence-length figure."""
    print("\n" + "#" * 70)
    print("# Phase 2: data preparation")
    print("#" * 70)

    data_config = data_config or DataConfig()
    train_loader, val_loader, test_loader, tokenizer, stats = \
        prepare_tiny_stories_dataset(config=data_config, verbose=True,
                                     make_figure=True, write_artifacts=write_artifacts)
    test_dataloader(train_loader, tokenizer)
    return stats


def run_pretrain(training_config: Optional[TrainingConfig] = None,
                 quick: bool = False,
                 resume_from: Optional[str] = None) -> Qwen3Model:
    """Phase 3: pre-train the model on TinyStories."""
    print("\n" + "#" * 70)
    print("# Phase 3: pre-training")
    print("#" * 70)

    if training_config is None:
        _, training_config, _ = build_configs(quick)

    # In quick mode, force the tiny model config after the real vocab is known by
    # letting training build it; here we pre-seed the stashed quick model.
    if quick and getattr(training_config, "_quick_model", None) is not None:
        training_config.model_config = training_config._quick_model

    model = train_qwen3_small(training_config, resume_from=resume_from,
                              write_artifacts=not quick)
    if not quick:
        artifacts.write_pretrain_history()
    return model


def run_generation(quick: bool = False):
    """Phase 4: sample text from the trained model across decoding strategies."""
    print("\n" + "#" * 70)
    print("# Phase 4: generation")
    print("#" * 70)

    model, tokenizer = load_model_and_tokenizer()
    if model is None:
        return None

    config = GenerationConfig(
        max_new_tokens=40 if quick else 60,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        device="cpu",
    )
    results = test_generation_quality(model, tokenizer, config)
    if not quick:
        compare_sampling_strategies(model, tokenizer, max_new_tokens=40)
        artifacts.write_generation_samples(results)
    return results


def run_sentiment(sentiment_config: Optional[SentimentConfig] = None,
                  quick: bool = False):
    """Phase 5: train and evaluate the downstream sentiment classifier."""
    print("\n" + "#" * 70)
    print("# Phase 5: sentiment classification")
    print("#" * 70)

    if sentiment_config is None:
        _, _, sentiment_config = build_configs(quick)
    return train_sentiment_classifier(config=sentiment_config,
                                      write_artifacts=not quick)


# ---------------------------------------------------------------------------
# Figure regeneration from committed artifacts
# ---------------------------------------------------------------------------

def regenerate_figures() -> None:
    """Rebuild committed figures from committed artifacts, without training.

    Training curves come from the pre-training history artifact, the confusion
    matrix from the stored count matrix, and the sequence-length distribution
    from the saved raw lengths (written during a real data-prep run).
    """
    ensure_directories()

    # Pre-training curves
    history_path = PRETRAIN_HISTORY_ARTIFACT if PRETRAIN_HISTORY_ARTIFACT.exists() else TRAINING_HISTORY_PATH
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        viz.plot_training_curves(history)
    else:
        print(f"No pre-training history at {history_path}; skipping training curves.")

    # Sentiment confusion matrix (from committed counts, else raw predictions)
    if SENTIMENT_METRICS_ARTIFACT.exists():
        with open(SENTIMENT_METRICS_ARTIFACT) as f:
            sentiment = json.load(f)
        cm = sentiment.get("confusion_matrix")
        if cm is not None:
            viz.plot_confusion_matrix_from_counts(cm)
        else:
            print("Sentiment metrics artifact has no confusion_matrix counts; skipping.")
    else:
        print(f"No sentiment metrics at {SENTIMENT_METRICS_ARTIFACT}; skipping confusion matrix.")

    # Sequence-length distribution (needs raw lengths from a data-prep run)
    if SEQUENCE_LENGTHS_ARTIFACT.exists():
        import numpy as np
        with open(SEQUENCE_LENGTHS_ARTIFACT) as f:
            lengths = json.load(f)
        arr = np.asarray(lengths)
        stats = {
            'mean': float(arr.mean()),
            'median': float(np.median(arr)),
            'percentile_95': float(np.percentile(arr, 95)),
        }
        viz.plot_sequence_length_distribution(arr, stats)
    else:
        print(f"No raw lengths at {SEQUENCE_LENGTHS_ARTIFACT}; "
              f"run phase 2 (data preparation) to regenerate the sequence-length figure.")

    # --- Analysis figures assembled from committed metric artifacts ---
    def _load(path):
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def _flatten_sentiment(payload):
        """Normalize old (nested test_metrics) and new (flat) artifact shapes."""
        if payload is None:
            return None
        flat = dict(payload)
        nested = payload.get('test_metrics')
        if isinstance(nested, dict):
            flat.update(nested)
        return flat

    baseline = _load(BASELINE_METRICS_ARTIFACT)
    sentiment_metrics = _flatten_sentiment(_load(SENTIMENT_METRICS_ARTIFACT))
    finetuned = _load(SENTIMENT_FINETUNED_METRICS_ARTIFACT)
    oov = _load(SENTIMENT_OOV_ERROR_ARTIFACT)

    # Baseline comparison on macro-F1, ordered floor -> probe -> fine-tuned -> ceiling
    entries = []
    if baseline:
        mj = baseline.get('majority_class', {}).get('confidence_intervals', {}).get('macro_f1')
        if mj:
            entries.append({'name': 'Majority', **mj})
    if sentiment_metrics:
        probe = sentiment_metrics.get('confidence_intervals', {}).get('macro_f1')
        if probe:
            entries.append({'name': 'Frozen probe', **probe})
    if finetuned:
        ft = finetuned.get('confidence_intervals', {}).get('macro_f1')
        if ft:
            entries.append({'name': 'Fine-tuned', **ft})
    if baseline:
        tf = baseline.get('tfidf_logreg', {}).get('confidence_intervals', {}).get('macro_f1')
        if tf:
            entries.append({'name': 'TF-IDF + LogReg', **tf})

    if len(entries) >= 2:
        viz.plot_baseline_comparison(entries)
    else:
        print("Fewer than two method artifacts available; skipping baseline-comparison figure.")

    # Accuracy by tokenizer OOV bucket
    if oov is not None:
        viz.plot_accuracy_by_oov(oov.get('buckets', []),
                                 oov.get('correlation_oov_vs_error'))
    else:
        print(f"No OOV-error artifact at {SENTIMENT_OOV_ERROR_ARTIFACT}; "
              f"run the sentiment phase to regenerate the accuracy-by-OOV figure.")

    # Per-class precision/recall/F1 with bootstrap CIs
    per_class_keys = [f"{c}_{m}" for c in ("negative", "positive")
                      for m in ("precision", "recall", "f1")]
    if sentiment_metrics is not None and all(k in sentiment_metrics for k in per_class_keys):
        viz.plot_per_class_metrics(sentiment_metrics,
                                   sentiment_metrics.get('confidence_intervals'))
    else:
        print(f"Per-class metrics not present in {SENTIMENT_METRICS_ARTIFACT} "
              f"(older artifact format); rerun the sentiment phase to regenerate "
              f"the per-class figure.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-small pre-training and sentiment pipeline."
    )
    parser.add_argument("--quick", action="store_true",
                        help="Shrink every stage for a fast smoke test.")
    parser.add_argument("--figures-only", action="store_true",
                        help="Regenerate committed figures from artifacts and exit.")
    parser.add_argument("--phase", type=int, choices=sorted(PHASE_NAMES),
                        help="Run only this numbered phase.")
    parser.add_argument("--through", type=int, choices=sorted(PHASE_NAMES),
                        help="Run phases 1..N in order.")
    parser.add_argument("--skip-prep", action="store_true",
                        help="Skip data preparation in the full run.")
    parser.add_argument("--skip-pretrain", action="store_true",
                        help="Skip pre-training in the full run.")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip generation in the full run.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume pre-training from a checkpoint path.")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch the interactive generation REPL and exit.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate the trained sentiment classifier on the test set and exit.")
    parser.add_argument("--baseline", action="store_true",
                        help="Compute majority-class and TF-IDF baselines plus the "
                             "tokenizer OOV rate, then exit.")
    parser.add_argument("--finetune", action="store_true",
                        help="Train the sentiment classifier with the base model "
                             "unfrozen (fine-tuned comparison), then exit.")
    return parser


def run_finetune():
    """Train the sentiment classifier with the base model unfrozen.

    This is the comparison point for the frozen probe: it measures how much
    additional signal fine-tuning the base recovers, at a much higher CPU cost.
    Results are written to the fine-tuned artifact rather than the headline one.
    """
    print("\n" + "#" * 70)
    print("# Fine-tuned sentiment comparison (base unfrozen)")
    print("#" * 70)

    config = SentimentConfig(freeze_base_model=False)
    return train_sentiment_classifier(config=config, write_artifacts=True,
                                      variant="finetuned")


def run_single_phase(phase: int, quick: bool, resume: Optional[str]):
    if phase == 1:
        run_architecture(quick)
    elif phase == 2:
        data_config, _, _ = build_configs(quick)
        run_data_prep(data_config, write_artifacts=not quick)
    elif phase == 3:
        _, training_config, _ = build_configs(quick)
        run_pretrain(training_config, quick=quick, resume_from=resume)
    elif phase == 4:
        run_generation(quick)
    elif phase == 5:
        _, _, sentiment_config = build_configs(quick)
        run_sentiment(sentiment_config, quick=quick)


def main(argv: Optional[list] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    ensure_directories()
    torch.manual_seed(DEFAULT_SEED)

    # Exit-early modes
    if args.figures_only:
        regenerate_figures()
        return
    if args.interactive:
        interactive_generate()
        return
    if args.eval_only:
        evaluate_sentiment_test()
        return
    if args.baseline:
        run_baselines()
        return
    if args.finetune:
        run_finetune()
        return

    if args.phase is not None:
        run_single_phase(args.phase, args.quick, args.resume)
        return

    if args.through is not None:
        for phase in range(1, args.through + 1):
            run_single_phase(phase, args.quick, args.resume)
        return

    # Default: full pipeline honoring skip flags
    data_config, training_config, sentiment_config = build_configs(args.quick)

    run_architecture(args.quick)

    if not args.skip_pretrain:
        # Pre-training internally runs data preparation
        run_pretrain(training_config, quick=args.quick, resume_from=args.resume)
    elif not args.skip_prep:
        run_data_prep(data_config, write_artifacts=not args.quick)

    if not args.skip_generate:
        run_generation(args.quick)

    run_sentiment(sentiment_config, quick=args.quick)


if __name__ == "__main__":
    main()
