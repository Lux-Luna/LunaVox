"""Smoke test for the new GUI package.

The ``[gui]`` extra is not installed in CI, so every import-level
assertion is gated behind ``importorskip``. Locally the full chain
resolves and the tests exercise the widget constructors without
opening a real window.
"""

from __future__ import annotations

import pytest

ctk = pytest.importorskip(
    "customtkinter",
    reason="GUI tests need the [gui] extra (customtkinter)",
    exc_type=ImportError,
)


def test_gui_package_imports():
    import lunavox.gui  # noqa: F401
    import lunavox.gui.i18n  # noqa: F401
    import lunavox.gui.theme  # noqa: F401


def test_translator_fallback_to_english():
    from lunavox.gui.i18n import Translator

    t = Translator(lang="en")
    assert t("synth.generate") == "Generate"
    t.set_lang("zh")
    assert t("synth.generate") == "开始合成"
    # Unknown keys should return the key rather than crash.
    assert t("definitely.not.a.real.key") == "definitely.not.a.real.key"


def test_voice_picker_builds_base_voice_without_window():
    """VoicePicker must not crash on a headless CTk root (only widget
    construction — we don't call mainloop)."""
    from lunavox.gui.i18n import Translator
    from lunavox.gui.widgets.voice_picker import VoicePicker
    from lunavox.runtime import SynthesisMode

    root = ctk.CTk()
    try:
        picker = VoicePicker(root, translator=Translator(lang="en"))
        voice = picker.build_voice()
        assert voice is not None
        assert voice.mode is SynthesisMode.BASE
    finally:
        root.destroy()
