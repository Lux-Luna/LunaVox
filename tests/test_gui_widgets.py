"""Widget-level tests for the GUI.

Constructors only — no mainloop, no per-test CTk root. Every test
receives the session-scoped ``gui_root`` fixture defined in the
top-level conftest, which avoids hammering Tcl interpreter bootstrap
across the suite.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip(
    "customtkinter",
    reason="GUI tests need the [gui] extra (customtkinter)",
    exc_type=ImportError,
)


def test_param_slider_group_roundtrip(gui_root: Any):
    from lunavox.gui.widgets.param_slider import FieldSpec, ParamSliderGroup

    group = ParamSliderGroup(
        gui_root,
        fields=[
            FieldSpec("temperature", "Temperature", 0.0, 1.5, 0.05, 0.6),
            FieldSpec("top_k", "Top-k", 1, 200, 1, 50, cast=int),
        ],
    )
    try:
        values = group.values()
        assert isinstance(values["temperature"], float)
        assert values["temperature"] == pytest.approx(0.6)
        assert isinstance(values["top_k"], int)
        assert values["top_k"] == 50

        group.set_values({"temperature": 0.9, "top_k": 20})
        updated = group.values()
        assert updated["temperature"] == pytest.approx(0.9)
        assert updated["top_k"] == 20
    finally:
        # Tear down just this widget subtree so the next test gets a
        # clean slate on the shared root.
        group.destroy()


def test_stats_card_update_with_none_resets(gui_root: Any):
    from lunavox.gui.i18n import Translator
    from lunavox.gui.widgets.stats_card import StatsCard

    card = StatsCard(gui_root, translator=Translator(lang="en"))
    try:
        card.update_stats(None)  # should not raise
    finally:
        card.destroy()


def test_stats_card_formats_a_real_stats_object(gui_root: Any):
    from lunavox.gui.i18n import Translator
    from lunavox.gui.widgets.stats_card import StatsCard
    from lunavox.runtime import SynthesisStats

    card = StatsCard(gui_root, translator=Translator(lang="en"))
    try:
        stats = SynthesisStats(
            t_tokenize_ms=12,
            t_encode_ms=34,
            t_generate_ms=567,
            t_decode_ms=89,
            t_total_ms=702,
            audio_duration_ms=2500,
            rtf=0.28,
        )
        card.update_stats(stats)
        # A labelled value must display as something other than the
        # empty placeholder ``—`` after update.
        assert card._rows["rtf"].cget("text") != "—"
        assert card._rows["total"].cget("text") != "—"
        assert card._rows["tokenize"].cget("text") != "—"
    finally:
        card.destroy()
