"""Thin wrapper around a single :class:`lunavox.runtime.Engine`.

In Phase 5A the HTTP layer owns exactly one Engine instance; requests
serialise on an ``asyncio.Lock`` so there is never more than one
synthesis in flight per GPU. This keeps the server correct (no C++
state corruption) and lines up the abstraction so Phase 5B can swap
the single Engine for a C++ BatchEngine without changing any of the
FastAPI handlers.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from lunavox.runtime import Engine, SynthesisParams


class EngineHolder:
    """Own a lazily-loaded Engine and a lock that serialises synth calls."""

    def __init__(self, model_dir: Path, n_threads: int = 4) -> None:
        self.model_dir = Path(model_dir)
        self.n_threads = n_threads
        self._engine: Optional[Engine] = None
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        """Load the underlying C++ engine in a background thread.

        Engine construction is synchronous and can take seconds to
        warm up, so we run it via ``run_in_executor`` to keep the
        FastAPI startup event non-blocking.
        """
        if self._engine is not None:
            return
        loop = asyncio.get_running_loop()
        self._engine = await loop.run_in_executor(
            None, lambda: Engine(self.model_dir, n_threads=self.n_threads)
        )

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            raise RuntimeError("EngineHolder.load() has not been awaited yet")
        return self._engine

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

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
