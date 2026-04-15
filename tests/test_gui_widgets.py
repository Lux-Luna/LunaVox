"""Widget-level tests for the GUI.

Constructors only — no mainloop, no long-lived CTk root between tests
(each test creates and destroys its own to keep tkinter state clean).
"""

from __future__ import annotations

import pytest

ctk = pytest.importorskip(
    "customtkinter",
    reason="GUI tests need the [gui] extra (customtkinter)",
    exc_type=ImportError,
)


def test_param_slider_group_roundtrip():
    from lunavox.gui.widgets.param_slider import FieldSpec, ParamSliderGroup

    root = ctk.CTk()
    try:
        group = ParamSliderGroup(
            root,
            fields=[
                FieldSpec("temperature", "Temperature", 0.0, 1.5, 0.05, 0.6),
                FieldSpec("top_k", "Top-k", 1, 200, 1, 50, cast=int),
            ],
        )
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
        root.destroy()


def test_stats_card_update_with_none_resets():
    from lunavox.gui.i18n import Translator
    from lunavox.gui.widgets.stats_card import StatsCard

    root = ctk.CTk()
    try:
        card = StatsCard(root, translator=Translator(lang="en"))
        card.update_stats(None)  # should not raise
    finally:
        root.destroy()


def test_stats_card_formats_a_real_stats_object():
    from lunavox.gui.i18n import Translator
    from lunavox.gui.widgets.stats_card import StatsCard
    from lunavox.runtime import SynthesisStats

    root = ctk.CTk()
    try:
        card = StatsCard(root, translator=Translator(lang="en"))
        stats = SynthesisStats(
            t_tokenize_ms=12,
            t_encode_ms=34,
            t_generate_ms=567,
            t_decode_ms=89,
            t_total_ms=702,
            audio_duration_ms=2500,
            rtf=0.28,
            rss_peak_bytes=1_500_000_000,
        )
        card.update_stats(stats)
        # A labelled value must display as something other than the
        # empty placeholder ``—`` after update.
        assert card._rows["rtf"].cget("text") != "—"
    finally:
        root.destroy()
