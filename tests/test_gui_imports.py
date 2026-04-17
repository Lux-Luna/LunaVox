"""Smoke test for the new GUI package.

The ``[gui]`` extra is not installed in CI, so every import-level
assertion is gated behind ``importorskip``. Locally the full chain
resolves and the tests exercise the widget constructors without
opening a real window.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip(
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


def test_voice_picker_headless_no_profile(gui_root: Any):
    """With no model profile the picker shows a hint and can't build a
    spec — but must not crash during construction."""
    from lunavox.gui.i18n import Translator
    from lunavox.gui.widgets.voice_picker import VoicePicker

    picker = VoicePicker(gui_root, translator=Translator(lang="en"))
    try:
        assert picker.build_voice_spec() is None
    finally:
        picker.destroy()


def test_voice_picker_design_profile_builds_voice_spec(gui_root: Any):
    """A design profile enables the Design tab; entering an instruction
    produces a :class:`VoiceSpec` in design mode."""
    from lunavox.gui.i18n import Translator
    from lunavox.gui.widgets.voice_picker import VoicePicker
    from lunavox.model.profile import ModelProfile

    profile = ModelProfile(
        model_type="design",
        model_size="1.7b",
        instruct_support=True,
        raw={"speaker_names": []},
    )
    picker = VoicePicker(gui_root, translator=Translator(lang="en"), model_profile=profile)
    try:
        picker._instruct_var.set("Speak cheerfully.")  # noqa: SLF001
        picker._mode_var.set("design")  # noqa: SLF001
        spec = picker.build_voice_spec()
        assert spec is not None
        assert spec.mode == "design"
        assert spec.instruct == "Speak cheerfully."
    finally:
        picker.destroy()


def test_voice_picker_custom_small_hides_instruct(gui_root: Any):
    """0.6B custom model has instruct_support=False — the picker must
    still produce a VoiceSpec without any instruction text."""
    from lunavox.gui.i18n import Translator
    from lunavox.gui.widgets.voice_picker import VoicePicker
    from lunavox.model.profile import ModelProfile

    profile = ModelProfile(
        model_type="custom",
        model_size="0.6b",
        instruct_support=False,
        raw={"speaker_names": ["vivian", "aiden"]},
    )
    picker = VoicePicker(gui_root, translator=Translator(lang="en"), model_profile=profile)
    try:
        picker._mode_var.set("custom")  # noqa: SLF001
        picker._speaker_var.set("vivian")  # noqa: SLF001
        # Instruction text is ignored when the profile says the model
        # doesn't support it, even if something is in the var.
        picker._instruct_var.set("should not leak through")  # noqa: SLF001
        spec = picker.build_voice_spec()
        assert spec is not None
        assert spec.mode == "custom"
        assert spec.speaker == "vivian"
        assert spec.instruct == ""
    finally:
        picker.destroy()
