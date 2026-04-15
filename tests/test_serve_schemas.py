"""Tests for the pydantic request/response schemas.

Gated behind the ``[serve]`` extra via ``importorskip`` so the
``[dev]``-only CI job cleanly skips them. Locally the tests verify
the schema surface without touching FastAPI, numpy, or the engine.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "pydantic",
    reason="serve tests need the [serve] extra",
    exc_type=ImportError,
)


def test_synth_request_defaults_to_base_voice():
    from lunavox.serve.schemas import SynthRequest

    req = SynthRequest(text="Hello")
    assert req.voice == "base"
    assert req.reference is None
    assert req.speaker is None
    assert req.temperature is None


def test_synth_request_rejects_empty_text():
    from pydantic import ValidationError

    from lunavox.serve.schemas import SynthRequest

    with pytest.raises(ValidationError):
        SynthRequest(text="")


def test_synth_request_accepts_every_voice_mode():
    from lunavox.serve.schemas import SynthRequest

    for mode in ("base", "clone", "custom", "design"):
        req = SynthRequest(text="Hi", voice=mode)
        assert req.voice == mode


def test_synth_request_rejects_unknown_voice():
    from pydantic import ValidationError

    from lunavox.serve.schemas import SynthRequest

    with pytest.raises(ValidationError):
        SynthRequest(text="Hi", voice="bogus")  # type: ignore[arg-type]


def test_temperature_range_enforced():
    from pydantic import ValidationError

    from lunavox.serve.schemas import SynthRequest

    # Inside range is fine.
    SynthRequest(text="Hi", temperature=0.7)
    # Above range is rejected.
    with pytest.raises(ValidationError):
        SynthRequest(text="Hi", temperature=3.0)


def test_models_response_roundtrip():
    from lunavox.serve.schemas import ModelInfo, ModelsResponse

    payload = ModelsResponse(
        active="base_small",
        models=[
            ModelInfo(
                name="base_small",
                display_name="Qwen3-TTS 0.6B Base",
                repo_id="Qwen/Qwen3-TTS-0.6B-Base",
                installed=True,
            ),
        ],
    )
    dumped = payload.model_dump()
    assert dumped["active"] == "base_small"
    assert dumped["models"][0]["installed"] is True


def test_health_response_status_enum():
    from pydantic import ValidationError

    from lunavox.serve.schemas import HealthResponse

    HealthResponse(status="ok", model="base_small", sample_rate=24000)
    with pytest.raises(ValidationError):
        HealthResponse(status="weird", model=None, sample_rate=None)  # type: ignore[arg-type]
