"""Shared test fixtures.

These tests deliberately do NOT load liblunavox, do NOT hit HuggingFace,
and do NOT touch GPU state. They cover pure-Python helpers so the CI
can stay fast and deterministic on all three platforms.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the in-tree package is importable when running `pytest` from the
# repo root without an editable install. This mirrors how CI invokes the
# test suite after `pip install -e .`.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
