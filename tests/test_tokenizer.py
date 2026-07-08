"""Tokenizer correctness: vocabulary build, round-trip, OOV, and coverage."""

from qwen3 import Qwen3SmallTokenizer

TOY_CORPUS = [
    "the cat sat on the mat",
    "the dog ran after the cat",
    "a boy and a girl went to the park",
]


def _trained_tokenizer():
    # min_frequency=1 so every toy token enters the vocabulary
    tokenizer = Qwen3SmallTokenizer(vocab_size=100, min_frequency=1)
    tokenizer.build_vocab(TOY_CORPUS, verbose=False)
    return tokenizer


def test_encode_decode_round_trip():
    tokenizer = _trained_tokenizer()
    text = "the cat sat on the mat"
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.decode(ids, skip_special_tokens=True) == text


def test_special_tokens_wrap_encoding():
    tokenizer = _trained_tokenizer()
    ids = tokenizer.encode("the cat", add_special_tokens=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id


def test_oov_maps_to_unk():
    tokenizer = _trained_tokenizer()
    ids = tokenizer.encode("zzzz", add_special_tokens=False)
    assert ids == [tokenizer.unk_id]


def test_coverage_reports_oov_rate():
    tokenizer = _trained_tokenizer()
    # "the cat" are in-vocab; "zzzz qqqq" are not: 2 of 4 tokens are OOV
    stats = tokenizer.coverage(["the cat", "zzzz qqqq"])
    assert stats["total_tokens"] == 4
    assert stats["oov_tokens"] == 2
    assert stats["oov_rate"] == 50.0
    assert stats["coverage"] == 50.0
