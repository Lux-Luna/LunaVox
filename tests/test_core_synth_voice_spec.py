"""Unit tests for :mod:`lunavox.core.synth.voice_spec`.

No liblunavox needed — we only exercise ``Voice`` dataclass
construction via the resolver, which raises pure-Python errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lunavox.core.synth.voice_spec import (
    VoiceResolutionError,
    VoiceSpec,
    resolve_voice,
)
from lunavox.runtime import SynthesisMode


def test_base_spec_resolves_to_base_voice():
    voice = resolve_voice(VoiceSpec.base())
    assert voice.mode is SynthesisMode.BASE


def test_clone_spec_resolves_to_clone_voice():
    voice = resolve_voice(VoiceSpec.clone("ref.wav"))
    assert voice.mode is SynthesisMode.CLONE_FILE
    assert voice.reference_path == Path("ref.wav")


def test_custom_spec_carries_speaker_and_instruct():
    voice = resolve_voice(VoiceSpec.custom("Vivian", "angry"))
    assert voice.mode is SynthesisMode.CUSTOM
    assert voice.speaker == "Vivian"
    assert voice.instruct == "angry"


def test_custom_spec_defaults_instruct_to_empty():
    voice = resolve_voice(VoiceSpec.custom("Vivian"))
    assert voice.instruct == ""


def test_design_spec_requires_instruct():
    voice = resolve_voice(VoiceSpec.design("Speak softly."))
    assert voice.mode is SynthesisMode.DESIGN
    assert voice.instruct == "Speak softly."


def test_clone_without_reference_raises_voice_resolution_error():
    with pytest.raises(VoiceResolutionError, match="reference"):
        resolve_voice(VoiceSpec(mode="clone"))


def test_custom_without_speaker_raises_voice_resolution_error():
    with pytest.raises(VoiceResolutionError, match="speaker"):
        resolve_voice(VoiceSpec(mode="custom"))


def test_design_without_instruct_raises_voice_resolution_error():
    with pytest.raises(VoiceResolutionError, match="instruct"):
        resolve_voice(VoiceSpec(mode="design"))
    with pytest.raises(VoiceResolutionError, match="instruct"):
        resolve_voice(VoiceSpec(mode="design", instruct="   "))


def test_unknown_mode_raises_voice_resolution_error():
    with pytest.raises(VoiceResolutionError, match="Unknown voice mode"):
        resolve_voice(VoiceSpec(mode="unknown"))  # type: ignore[arg-type]


def test_voice_spec_is_frozen():
    spec = VoiceSpec.base()
    with pytest.raises(Exception):
        spec.mode = "clone"  # type: ignore[misc]


def test_voice_resolution_error_is_value_error():
    """Catch-all callers that handle ValueError should catch our
    subclass for free — preserves std exception hierarchy."""
    assert issubclass(VoiceResolutionError, ValueError)
