"""Architecture correctness: output shape, causal masking, and parameter count."""

import torch

from qwen3 import Qwen3Config, Qwen3Model


def _tiny_model():
    config = Qwen3Config(
        vocab_size=64,
        max_seq_len=16,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=64,
    )
    model = Qwen3Model(config)
    model.eval()
    return model, config


def test_forward_output_shape():
    model, config = _tiny_model()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    with torch.no_grad():
        out = model(input_ids, return_dict=True)
    assert out["logits"].shape == (2, 8, config.vocab_size)
    assert out["last_hidden_state"].shape == (2, 8, config.d_model)


def test_attention_is_causal():
    """Changing a later token must not affect logits at earlier positions."""
    model, config = _tiny_model()
    seq = torch.randint(0, config.vocab_size, (1, 8))

    modified = seq.clone()
    # Alter only the final position; earlier-position logits should be identical
    modified[0, -1] = (modified[0, -1] + 1) % config.vocab_size

    with torch.no_grad():
        base = model(seq, return_dict=True)["logits"]
        changed = model(modified, return_dict=True)["logits"]

    assert torch.allclose(base[:, :-1], changed[:, :-1], atol=1e-5)


def test_parameter_count_matches_analytical_estimate():
    model, _ = _tiny_model()
    counts = model.count_parameters()
    assert counts["difference"] == 0
    assert counts["actual_total"] == counts["expected_total"]
