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


def test_create_app_registers_routes(tmp_path: Path):
    from lunavox.serve.server import create_app

    model_dir = tmp_path / "base_small"
    model_dir.mkdir()

    app = create_app(model_dir, n_threads=2)
    assert app.title == "LunaVox"

    paths = {getattr(route, "path", None) for route in app.routes}
    for expected in {"/health", "/v1/models", "/v1/synth", "/v1/stream"}:
        assert expected in paths, f"Missing route {expected}; have {paths}"


def test_engine_holder_build_params(tmp_path: Path):
    from lunavox.serve.engine_holder import EngineHolder

    holder = EngineHolder(model_dir=tmp_path, n_threads=1, batch_size=2)
    assert holder.batch_size == 2
    assert holder.n_threads == 1

    params = holder.build_params(temperature=0.9, top_k=20)
    assert params.temperature == 0.9
    assert params.top_k == 20
    # Untouched fields keep their defaults.
    assert params.top_p == 1.0
    assert params.n_threads == 1


def test_engine_holder_batch_raises_before_load(tmp_path: Path):
    from lunavox.serve.engine_holder import EngineHolder

    holder = EngineHolder(model_dir=tmp_path, batch_size=1)
    import pytest

    with pytest.raises(RuntimeError, match="load"):
        _ = holder.batch
    with pytest.raises(RuntimeError, match="load"):
        _ = holder.sample_rate
