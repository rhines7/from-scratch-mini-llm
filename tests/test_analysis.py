"""Bootstrap confidence intervals and the OOV-versus-error analysis."""

from qwen3 import Qwen3SmallTokenizer, bootstrap_metric_cis, oov_error_analysis


def test_bootstrap_cis_bracket_the_point_estimate():
    # Alternating labels with a few deliberate errors keeps metrics off the
    # degenerate 0/1 boundary so the interval has non-zero width
    labels = [0, 1] * 25
    predictions = list(labels)
    for i in range(0, 20, 4):
        predictions[i] = 1 - predictions[i]

    cis = bootstrap_metric_cis(predictions, labels, n_boot=200, seed=42)

    for key in ("accuracy", "macro_f1", "negative_f1", "positive_f1"):
        band = cis[key]
        assert band["lower"] <= band["point"] <= band["upper"]
        assert 0.0 <= band["lower"] <= band["upper"] <= 1.0
    assert cis["confidence"] == 0.95


def test_bootstrap_is_deterministic_under_fixed_seed():
    labels = [0, 1, 0, 1, 1, 0]
    predictions = [0, 1, 1, 1, 0, 0]
    first = bootstrap_metric_cis(predictions, labels, n_boot=100, seed=7)
    second = bootstrap_metric_cis(predictions, labels, n_boot=100, seed=7)
    assert first == second


def _toy_tokenizer():
    tokenizer = Qwen3SmallTokenizer(vocab_size=100, min_frequency=1)
    tokenizer.build_vocab(["good great happy", "bad sad angry"], verbose=False)
    return tokenizer


def test_oov_error_analysis_links_oov_to_errors():
    tokenizer = _toy_tokenizer()
    texts = ["good great", "zzz qqq"]   # first fully in-vocab, second fully OOV
    labels = [1, 0]
    predictions = [1, 1]               # first correct, second wrong

    result = oov_error_analysis(texts, labels, predictions, tokenizer)

    assert result["n"] == 2
    assert result["mean_oov_correct"] == 0.0
    assert result["mean_oov_incorrect"] == 100.0
    # Higher OOV coincides with the error, so the correlation is positive
    assert result["correlation_oov_vs_error"] > 0
    assert sum(b["count"] for b in result["buckets"]) == 2
