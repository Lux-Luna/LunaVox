"""Unit tests for :class:`lunavox.runtime.Voice` factories.

These exercise the voice selector surface without loading liblunavox.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lunavox.runtime import SynthesisMode, Voice


def test_base_voice_has_base_mode():
    v = Voice.base()
    assert v.mode is SynthesisMode.BASE
    assert v.reference_path is None
    assert v.speaker is None
    assert v.instruct is None


def test_clone_file_accepts_str_and_path():
    v1 = Voice.clone_file("ref.wav")
    assert v1.mode is SynthesisMode.CLONE_FILE
    assert v1.reference_path == Path("ref.wav")

    v2 = Voice.clone_file(Path("features") / "ref_0.6B.json")
    assert v2.reference_path == Path("features/ref_0.6B.json")


def test_custom_requires_speaker():
    with pytest.raises(ValueError, match="speaker"):
        Voice.custom("")


def test_custom_stores_speaker_and_instruct():
    v = Voice.custom("Vivian", instruct="Use angry tone.")
    assert v.mode is SynthesisMode.CUSTOM
    assert v.speaker == "Vivian"
    assert v.instruct == "Use angry tone."


def test_custom_default_instruct_is_empty():
    v = Voice.custom("Vivian")
    assert v.instruct == ""


def test_design_requires_non_empty_instruct():
    with pytest.raises(ValueError, match="instruct"):
        Voice.design("")
    with pytest.raises(ValueError, match="instruct"):
        Voice.design("   ")


def test_design_carries_instruct():
    v = Voice.design("Speak in an incredulous tone.")
    assert v.mode is SynthesisMode.DESIGN
    assert v.instruct == "Speak in an incredulous tone."


def test_voice_is_frozen():
    """Voice must be hashable so it can live in sets / as dict keys and
    be safely shared between threads without defensive copying."""
    v = Voice.base()
    with pytest.raises(Exception):
        v.mode = SynthesisMode.CUSTOM  # type: ignore[misc]
