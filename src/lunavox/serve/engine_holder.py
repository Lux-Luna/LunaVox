"""Pool holder sitting between the FastAPI app and the synth pipeline.

Phase 5A held a single :class:`Engine` + ``asyncio.Lock`` to serialise
concurrent requests. 5B replaced that with a :class:`BatchEngine` pool
whose idle queue provides back-pressure. This Phase 6 refactor folds
in one more layer: the holder now exposes an
:class:`AsyncSynthesisPipeline` as its primary handle, so endpoints
never touch the batch pool directly. Voice resolution and parameter
merging also leave this class — they're single-sourced in
:mod:`lunavox.core.synth`.

The class still owns the lifecycle (lazy load / close) of the pool
so the FastAPI ``lifespan`` hook has one object to drive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from lunavox.core.synth import AsyncSynthesisPipeline
from lunavox.runtime import BatchEngine


class EngineHolder:
    """Owns one :class:`BatchEngine` and the :class:`AsyncSynthesisPipeline`
    that wraps it.

    Endpoints read ``pipeline`` for every synthesis call; the pool is
    still exposed via ``batch`` for the ``/metrics`` endpoint which
    needs pool-level telemetry (idle / busy counts).
    """

    def __init__(
        self,
        model_dir: Path,
        *,
        batch_size: int = 4,
        n_threads: int = 4,
        auto_split_threshold: int = 240,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.auto_split_threshold = auto_split_threshold
        self._batch: Optional[BatchEngine] = None
        self._pipeline: Optional[AsyncSynthesisPipeline] = None

    async def load(self) -> None:
        if self._batch is not None:
            return
        self._batch = BatchEngine(
            self.model_dir,
            batch_size=self.batch_size,
            n_threads=self.n_threads,
        )
        await self._batch.load()
        self._pipeline = AsyncSynthesisPipeline(
            self._batch,
            auto_split_threshold=self.auto_split_threshold,
        )

    def close(self) -> None:
        if self._batch is not None:
            self._batch.close()
            self._batch = None
            self._pipeline = None

    @property
    def batch(self) -> BatchEngine:
        if self._batch is None:
            raise RuntimeError("EngineHolder.load() has not been awaited yet")
        return self._batch

    @property
    def pipeline(self) -> AsyncSynthesisPipeline:
        if self._pipeline is None:
            raise RuntimeError("EngineHolder.load() has not been awaited yet")
        return self._pipeline

    @property
    def sample_rate(self) -> int:
        return self.batch.sample_rate
