"""Unit tests for the :class:`StatsCard` TTFB row + ttfb_ms kwarg.

TTFB is a GUI-only wall-clock metric — it doesn't ride on
:class:`SynthesisStats` — so its rendering path is worth pinning
independently from the engine-reported timings.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip(
    "customtkinter",
    reason="GUI tests need the [gui] extra (customtkinter)",
    exc_type=ImportError,
)


def _fresh_stats() -> Any:
    from lunavox.runtime import SynthesisStats

    # All-zero stats are fine — we're exercising the label-writing
    # behaviour, not the number formatting itself.
    return SynthesisStats()


def test_ttfb_renders_in_ms(gui_root: Any):
    from lunavox.gui.i18n import Translator
    from lunavox.gui.widgets.stats_card import StatsCard

    card = StatsCard(gui_root, translator=Translator(lang="en"))
    try:
        card.update_stats(_fresh_stats(), ttfb_ms=123.4)
        assert card._rows["ttfb"].cget("text") == "123 ms"  # noqa: SLF001
    finally:
        card.destroy()


def test_ttfb_rounds_to_integer(gui_root: Any):
    from lunavox.gui.i18n import Translator
    from lunavox.gui.widgets.stats_card import StatsCard

    card = StatsCard(gui_root, translator=Translator(lang="en"))
    try:
        card.update_stats(_fresh_stats(), ttfb_ms=88.9)
        assert card._rows["ttfb"].cget("text") == "89 ms"  # noqa: SLF001
    finally:
        card.destroy()


def test_none_resets_all_rows_including_ttfb(gui_root: Any):
    from lunavox.gui.i18n import Translator
    from lunavox.gui.widgets.stats_card import StatsCard

    card = StatsCard(gui_root, translator=Translator(lang="en"))
    try:
        card.update_stats(_fresh_stats(), ttfb_ms=42.0)
        assert card._rows["ttfb"].cget("text") == "42 ms"  # noqa: SLF001
        card.update_stats(None)
        for key in ("rtf", "ttfb", "duration", "total"):
            assert card._rows[key].cget("text") == "—"  # noqa: SLF001
    finally:
        card.destroy()


def test_stats_without_ttfb_leaves_ttfb_cell_dash(gui_root: Any):
    """When callers pass stats but no ttfb_ms (e.g. an error path that
    still recovered stats), the ttfb row stays at its initial '—'."""
    from lunavox.gui.i18n import Translator
    from lunavox.gui.widgets.stats_card import StatsCard

    card = StatsCard(gui_root, translator=Translator(lang="en"))
    try:
        card.update_stats(_fresh_stats())
        assert card._rows["ttfb"].cget("text") == "—"  # noqa: SLF001
    finally:
        card.destroy()
