"""
Entry point for the Qwen3-small pipeline.

This thin facade re-exports the full public API from the qwen3 package and hands
command-line execution to qwen3.pipeline.main. Run stages from the shell:

    python qwen3_pipeline.py                 # full pipeline (reproduces results)
    python qwen3_pipeline.py --quick         # fast smoke test
    python qwen3_pipeline.py --figures-only  # rebuild figures from artifacts
    python qwen3_pipeline.py --phase 4       # run one numbered phase
    python qwen3_pipeline.py --interactive   # interactive generation REPL
    python qwen3_pipeline.py --eval-only     # evaluate the sentiment classifier

Or import it as a library:

    from qwen3_pipeline import Qwen3Model, GenerationConfig, generate
"""

from qwen3 import *          # noqa: F401,F403  (re-export the package API)
from qwen3 import __all__ as _package_all
from qwen3 import __version__  # module attribute, kept out of __all__ for parity
from qwen3.pipeline import main

__all__ = list(_package_all)

if __name__ == "__main__":
    main()
