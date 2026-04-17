"""Unit tests for :class:`EngineSession` — the GUI-owned holder for
an optional pre-loaded :class:`Engine`.

The real ``Engine`` constructor loads native libraries, so every test
injects a fake factory instead. We're pinning the holder's lifecycle
semantics (load caches / unload releases / acquire does not silently
promote throwaways), not any actual synthesis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip(
    "customtkinter",
    reason="EngineSession lives in the [gui] package; skip when the extra is absent",
    exc_type=ImportError,
)


class _FakeEngine:
    """Records close() calls so tests can assert eager teardown."""

    def __init__(self, model_dir: Path, n_threads: int) -> None:
        self.model_dir = Path(model_dir)
        self.n_threads = int(n_threads)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _factory(record: list[_FakeEngine]) -> Any:
    def make(model_dir: Path, n_threads: int) -> _FakeEngine:
        eng = _FakeEngine(model_dir, n_threads)
        record.append(eng)
        return eng

    return make


def test_acquire_without_load_is_throwaway(tmp_path: Path):
    """No prior load() → acquire() builds a throwaway engine and closes
    it on context exit."""
    from lunavox.gui.engine_session import EngineSession

    built: list[_FakeEngine] = []
    sess = EngineSession(engine_factory=_factory(built))

    with sess.acquire(tmp_path / "m", n_threads=4):
        assert not built[0].closed
    assert built[0].closed
    assert sess.loaded_model is None, "throwaway must not promote into cache"


def test_load_caches_for_acquire(tmp_path: Path):
    """load() populates the cache; subsequent acquire() with the same
    key yields the cached engine and does NOT close it on exit."""
    from lunavox.gui.engine_session import EngineSession

    built: list[_FakeEngine] = []
    sess = EngineSession(engine_factory=_factory(built))
    model = tmp_path / "m"

    sess.load(model, 4)
    assert len(built) == 1
    assert sess.is_loaded_for(model, 4)

    with sess.acquire(model, n_threads=4):
        pass
    assert not built[0].closed, "cached engine must survive acquire context exit"
    assert len(built) == 1, "acquire on cache hit must not build a new engine"


def test_acquire_mismatch_uses_throwaway_without_disturbing_cache(tmp_path: Path):
    """A stray acquire() with a different key must build its own
    throwaway — it must NOT evict the user's explicitly pre-loaded
    engine. Otherwise a mis-plumbed call site could silently replace
    the user's cache with an unrelated model."""
    from lunavox.gui.engine_session import EngineSession

    built: list[_FakeEngine] = []
    sess = EngineSession(engine_factory=_factory(built))
    model_a = tmp_path / "a"
    model_b = tmp_path / "b"

    sess.load(model_a, 4)
    with sess.acquire(model_b, n_threads=4):
        pass

    # The A engine is still cached, B engine was throwaway and closed.
    assert sess.is_loaded_for(model_a, 4)
    assert not built[0].closed  # A survived
    assert built[1].closed  # B was torn down


def test_load_replaces_prior_cache(tmp_path: Path):
    """Explicit second load() releases the first one before caching
    the new one — that's the intentional 'switch the cached model'
    path (e.g. user unticks A, ticks B)."""
    from lunavox.gui.engine_session import EngineSession

    built: list[_FakeEngine] = []
    sess = EngineSession(engine_factory=_factory(built))

    sess.load(tmp_path / "a", 4)
    sess.load(tmp_path / "b", 4)

    assert built[0].closed, "old cache must be released when load() replaces it"
    assert not built[1].closed
    assert sess.is_loaded_for(tmp_path / "b", 4)


def test_unload_is_idempotent(tmp_path: Path):
    from lunavox.gui.engine_session import EngineSession

    built: list[_FakeEngine] = []
    sess = EngineSession(engine_factory=_factory(built))
    sess.load(tmp_path / "m", 4)

    sess.unload()
    sess.unload()  # second call must not re-close or raise
    assert built[0].closed
    assert sess.loaded_model is None


def test_is_loaded_for_matches_key(tmp_path: Path):
    from lunavox.gui.engine_session import EngineSession

    sess = EngineSession(engine_factory=_factory([]))
    model = tmp_path / "m"

    assert not sess.is_loaded_for(model, 4)
    sess.load(model, 4)
    assert sess.is_loaded_for(model, 4)
    assert not sess.is_loaded_for(model, 8)
    assert not sess.is_loaded_for(tmp_path / "other", 4)
