"""Shared test fixtures.

These tests deliberately do NOT load liblunavox, do NOT hit HuggingFace,
and do NOT touch GPU state. They cover pure-Python helpers so the CI
can stay fast and deterministic on all three platforms.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

# Ensure the in-tree package is importable when running `pytest` from the
# repo root without an editable install. This mirrors how CI invokes the
# test suite after `pip install -e .`.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def gui_root() -> Iterator[Any]:
    """Shared ``customtkinter.CTk`` root for every GUI test.

    Before this fixture landed, every GUI test created and destroyed
    its own root, which hammered Tcl interpreter bootstrap four times
    per suite run and occasionally tripped a
    ``TclError: couldn't read file auto.tcl`` on miniconda setups.
    One root per session is strictly less risky and still gives each
    test a fresh widget subtree to build on.

    Skips cleanly (rather than erroring) if the ``[gui]`` extra isn't
    installed — matches the ``importorskip`` pattern the individual
    GUI test files use.
    """
    ctk = pytest.importorskip(
        "customtkinter",
        reason="GUI tests need the [gui] extra (customtkinter)",
        exc_type=ImportError,
    )
    # Headless CI runners (Linux without xvfb) can import customtkinter
    # but fail at root creation with `TclError: no display name`. Skip
    # rather than erroring so the rest of the suite keeps running.
    try:
        root = ctk.CTk()
    except Exception as err:  # noqa: BLE001 — tkinter raises a stdlib TclError subclass
        pytest.skip(f"GUI tests need a display ($DISPLAY): {err}")
    try:
        yield root
    finally:
        # Session teardown — swallow errors so a stale Tcl state on
        # miniconda doesn't break the final pytest exit status.
        with contextlib.suppress(Exception):
            root.destroy()
