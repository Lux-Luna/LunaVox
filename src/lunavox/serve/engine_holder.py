"""Pool holder sitting between the FastAPI app and :class:`BatchEngine`.

Phase 5A held a single :class:`Engine` and an ``asyncio.Lock`` to
serialise concurrent requests. 5B replaces that with a
:class:`BatchEngine` pool of ``N`` engines whose internal
``asyncio.Queue`` already provides back-pressure — so the lock is
gone and handlers just call into the pool directly.

The class name :class:`EngineHolder` is kept for backwards
compatibility of the import site (the serve module's handlers); it
now wraps a :class:`BatchEngine` instead of a single engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from lunavox.runtime import BatchEngine, SynthesisParams


class EngineHolder:
    """Thin adapter around :class:`BatchEngine` for the FastAPI layer.

    Construction is lazy — the underlying pool is loaded by
    :meth:`load`, which the FastAPI ``lifespan`` context manager
    awaits on startup. Handlers read ``batch`` directly once loaded.
    """

    def __init__(
        self,
        model_dir: Path,
        *,
        batch_size: int = 4,
        n_threads: int = 4,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.batch_size = batch_size
        self.n_threads = n_threads
        self._batch: Optional[BatchEngine] = None

    async def load(self) -> None:
        if self._batch is not None:
            return
        self._batch = BatchEngine(
            self.model_dir,
            batch_size=self.batch_size,
            n_threads=self.n_threads,
        )
        await self._batch.load()

    def close(self) -> None:
        if self._batch is not None:
            self._batch.close()
            self._batch = None

    @property
    def batch(self) -> BatchEngine:
        if self._batch is None:
            raise RuntimeError("EngineHolder.load() has not been awaited yet")
        return self._batch

    @property
    def sample_rate(self) -> int:
        return self.batch.sample_rate

    def build_params(
        self,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_audio_tokens: Optional[int] = None,
    ) -> SynthesisParams:
        """Merge request-level overrides onto :class:`SynthesisParams` defaults."""
        params = SynthesisParams()
        if temperature is not None:
            params.temperature = float(temperature)
        if top_p is not None:
            params.top_p = float(top_p)
        if top_k is not None:
            params.top_k = int(top_k)
        if max_audio_tokens is not None:
            params.max_audio_tokens = int(max_audio_tokens)
        params.n_threads = self.n_threads
        return params
