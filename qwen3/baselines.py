"""
Baselines and tokenizer-coverage analysis for the sentiment task.

The frozen-probe accuracy is only interpretable next to reference points. This
module computes two: a majority-class baseline (the floor any classifier must
clear) and a TF-IDF + logistic-regression baseline (a strong lexical model on
the same split). It also measures the tokenizer OOV rate on the sentiment
sentences, which is the most likely ceiling on the probe because the vocabulary
was fit on TinyStories rather than emotion text.

Everything reuses the exact split from downstream.py so the numbers compare to
the probe on the same held-out test set. No Pandas is used.
"""

from typing import Dict, Optional

from .config import TOKENIZER_VOCAB_PATH
from .downstream import (
    SentimentConfig,
    load_emotions_data,
    split_data,
    compute_metrics,
    bootstrap_metric_cis,
)
from .tokenizer import Qwen3SmallTokenizer
from . import artifacts


def _majority_baseline(train_labels, test_labels) -> Dict:
    """Predict the training-majority class for every test example."""
    majority_label = max(set(train_labels), key=train_labels.count)
    predictions = [majority_label] * len(test_labels)
    metrics = compute_metrics(predictions, test_labels)
    metrics["predicted_label"] = majority_label
    # Same bootstrap CIs as the probe so the numbers compare on equal footing
    metrics["confidence_intervals"] = bootstrap_metric_cis(predictions, test_labels)
    return metrics


def _tfidf_logreg_baseline(train_texts, train_labels, test_texts, test_labels,
                           random_seed: int) -> Dict:
    """Fit TF-IDF features plus logistic regression on the training split."""
    # Imported lazily so importing the package never requires scikit-learn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer()
    train_features = vectorizer.fit_transform(train_texts)
    test_features = vectorizer.transform(test_texts)

    # A fixed seed keeps the solver's result reproducible run to run
    classifier = LogisticRegression(max_iter=1000, random_state=random_seed)
    classifier.fit(train_features, train_labels)

    predictions = classifier.predict(test_features).tolist()
    metrics = compute_metrics(predictions, test_labels)
    metrics["confidence_intervals"] = bootstrap_metric_cis(predictions, test_labels)
    return metrics


def _oov_analysis(all_texts) -> Optional[Dict]:
    """Report the tokenizer OOV rate on the sentiment corpus, if a vocab exists."""
    if not TOKENIZER_VOCAB_PATH.exists():
        print(f"No tokenizer vocabulary at {TOKENIZER_VOCAB_PATH}; "
              f"run data preparation to enable the OOV analysis. Skipping.")
        return None

    tokenizer = Qwen3SmallTokenizer(vocab_size=20000)
    tokenizer.load_vocab(str(TOKENIZER_VOCAB_PATH))
    return tokenizer.coverage(all_texts)


def run_baselines(config: Optional[SentimentConfig] = None,
                  write_artifacts: bool = True) -> Dict:
    """Compute majority-class and TF-IDF baselines plus the OOV analysis.

    Uses the same data file, split ratios, and seed as the probe so the baseline
    numbers sit on the identical held-out test set.
    """
    if config is None:
        config = SentimentConfig()

    print("=" * 70)
    print("Baselines and tokenizer coverage")
    print("=" * 70)

    texts, _, labels = load_emotions_data(config.data_file)
    train_texts, train_labels, _, _, test_texts, test_labels = split_data(
        texts, labels, config.train_split, config.val_split,
        config.test_split, config.random_seed
    )

    majority = _majority_baseline(train_labels, test_labels)
    print(f"Majority class ({majority['predicted_label']}): "
          f"accuracy {majority['accuracy']:.4f}, macro-F1 {majority['macro_f1']:.4f}")

    tfidf = _tfidf_logreg_baseline(train_texts, train_labels, test_texts,
                                   test_labels, config.random_seed)
    print(f"TF-IDF + logistic regression: "
          f"accuracy {tfidf['accuracy']:.4f}, macro-F1 {tfidf['macro_f1']:.4f}")

    oov = _oov_analysis(texts)
    if oov is not None:
        print(f"Tokenizer OOV rate on sentiment text: {oov['oov_rate']:.2f}% "
              f"(coverage {oov['coverage']:.2f}%)")

    results = {
        "dataset": {
            "total": len(texts),
            "positive": labels.count(1),
            "negative": labels.count(0),
            "test_size": len(test_labels),
        },
        "majority_class": majority,
        "tfidf_logreg": tfidf,
        "tokenizer_oov": oov,
    }

    if write_artifacts:
        artifacts.write_baseline_metrics(results)

    return results
