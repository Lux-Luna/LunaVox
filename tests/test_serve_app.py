"""Smoke tests for the FastAPI app factory.

Verifies the app creates cleanly and registers every expected route
without starting uvicorn or loading the C engine. Gated behind the
``[serve]`` extra.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip(
    "fastapi",
    reason="serve tests need the [serve] extra",
    exc_type=ImportError,
)
pytest.importorskip(
    "prometheus_client",
    reason="serve tests need the [serve] extra",
    exc_type=ImportError,
)


def test_create_app_registers_routes(tmp_path: Path):
    from lunavox.serve.server import create_app

    model_dir = tmp_path / "base_small"
    model_dir.mkdir()

    app = create_app(model_dir, n_threads=2)
    assert app.title == "LunaVox"

    paths = {getattr(route, "path", None) for route in app.routes}
    expected_routes = {
        "/health",
        "/metrics",  # Phase 5C
        "/v1/models",
        "/v1/synth",
        "/v1/stream",
        "/v1/stream/text",  # Phase 5C
    }
    for expected in expected_routes:
        assert expected in paths, f"Missing route {expected}; have {paths}"


def test_engine_holder_constructs_without_loading(tmp_path: Path):
    """Phase 6 moved parameter construction out of EngineHolder into
    :class:`SynthesisParams.from_overrides`. The holder only wraps
    lifecycle + pool + pipeline now."""
    from lunavox.serve.engine_holder import EngineHolder

    holder = EngineHolder(model_dir=tmp_path, n_threads=1, batch_size=2)
    assert holder.batch_size == 2
    assert holder.n_threads == 1
    assert holder.auto_split_threshold == 240  # default


def test_engine_holder_batch_and_pipeline_raise_before_load(tmp_path: Path):
    from lunavox.serve.engine_holder import EngineHolder

    holder = EngineHolder(model_dir=tmp_path, batch_size=1)
    import pytest

    with pytest.raises(RuntimeError, match="load"):
        _ = holder.batch
    with pytest.raises(RuntimeError, match="load"):
        _ = holder.pipeline
    with pytest.raises(RuntimeError, match="load"):
        _ = holder.sample_rate


def test_synth_request_round_trip_to_voice_spec():
    """Schemas now carry the VoiceSpec translation themselves so
    handlers don't have to — verify the mapping is correct."""
    from lunavox.serve.schemas import SynthRequest

    req = SynthRequest(
        text="hi",
        voice="custom",
        speaker="Vivian",
        instruct="angry",
        temperature=0.8,
    )
    spec = req.to_voice_spec()
    assert spec.mode == "custom"
    assert spec.speaker == "Vivian"
    assert spec.instruct == "angry"

    overrides = req.param_overrides()
    assert overrides["temperature"] == 0.8
    assert overrides["top_p"] is None  # None means "no override"
