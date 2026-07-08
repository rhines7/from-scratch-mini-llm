"""Pytest configuration.

Ensures the repository root is importable so tests can `import qwen3` and
`import qwen3_pipeline` regardless of pytest's working directory.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
