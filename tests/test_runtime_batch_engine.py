"""Unit tests for :class:`lunavox.runtime.BatchEngine`.

These exercise the construction / validation surface without loading
the real C engine — the pool's lifecycle methods (``load``, ``close``,
``submit``) need a live engine, so they're left for manual
verification via ``lunavox serve``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lunavox.runtime import BatchEngine


def test_batch_size_must_be_positive(tmp_path: Path):
    with pytest.raises(ValueError, match="batch_size"):
        BatchEngine(tmp_path, batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        BatchEngine(tmp_path, batch_size=-1)


def test_batch_engine_default_batch_size(tmp_path: Path):
    batch = BatchEngine(tmp_path)
    assert batch.batch_size == 4
    assert batch.n_threads == 4
    assert batch.model_dir == tmp_path
    # Pool starts empty and is populated by load().
    assert batch._engines == []
    assert batch._idle is None


def test_batch_engine_custom_sizes(tmp_path: Path):
    batch = BatchEngine(tmp_path, batch_size=8, n_threads=2)
    assert batch.batch_size == 8
    assert batch.n_threads == 2


def test_sample_rate_raises_before_load(tmp_path: Path):
    batch = BatchEngine(tmp_path, batch_size=1)
    with pytest.raises(RuntimeError, match="load"):
        _ = batch.sample_rate


def test_submit_raises_before_load(tmp_path: Path):
    import asyncio

    batch = BatchEngine(tmp_path, batch_size=1)

    async def _run():
        with pytest.raises(RuntimeError, match="load"):
            await batch.submit("hi")

    asyncio.run(_run())


def test_close_is_idempotent(tmp_path: Path):
    batch = BatchEngine(tmp_path, batch_size=1)
    batch.close()
    batch.close()  # second close must be a no-op, not a crash
    assert batch._closed is True
