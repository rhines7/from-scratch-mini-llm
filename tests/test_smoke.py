"""Smoke test: the package imports and the architecture phase runs in --quick mode.

Phase 1 needs neither the network nor the datasets, so it is safe in CI. Quick
mode also guarantees no committed artifacts are overwritten.
"""

import qwen3_pipeline


def test_entry_script_exposes_main():
    assert callable(qwen3_pipeline.main)


def test_import_parity_between_package_and_entry():
    import qwen3
    assert set(qwen3.__all__) == set(qwen3_pipeline.__all__)


def test_architecture_phase_quick_runs():
    # --phase 1 --quick builds the tiny model and reports its parameter count
    qwen3_pipeline.main(["--phase", "1", "--quick"])
