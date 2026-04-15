"""Concurrent-request pool built on top of :class:`Engine`.

Phase 5B's shipping design is a **context pool**: ``BatchEngine``
holds ``N`` independent :class:`Engine` instances loaded from the
same model directory, each with its own llama.cpp contexts, ONNX
decoder state, and scratch buffers. A scheduler hands one engine
out per active request, releasing it back into the pool when the
request finishes.

This gives real concurrent GPU utilisation without a llama.cpp
multi-sequence rewrite — the cost is ``N ×`` per-engine KV cache
footprint (~200 MB on a 0.6B model, so ~800 MB extra at N=4), which
is a small fraction of a consumer GPU's VRAM. The trade-off was
explicitly accepted during Phase 5B planning; the alternative
(true continuous batching via ``n_seq_max > 1`` in
``llama_wrapper.cpp``) is an order of magnitude more invasive and
gains only the KV cache VRAM savings.

The public API — :meth:`submit`, :meth:`synthesize_stream`,
:meth:`close` — is designed to be stable across a future upgrade
to real batching: the internal pool would become a single
multi-sequence engine, and everything above would keep working.
Serving code (:mod:`lunavox.serve`) only touches this class, never
individual :class:`Engine` instances.

Example::

    batch = BatchEngine(model_dir="models/base_small", batch_size=4)
    try:
        result = await batch.submit("Hello", voice=Voice.base())
        async for chunk in batch.synthesize_stream("Hi", voice=Voice.base()):
            play(chunk.audio)
    finally:
        batch.close()
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any, Optional, Union

from .engine import Engine
from .params import SynthesisChunk, SynthesisParams, SynthesisResult
from .voice import Voice


class BatchEngine:
    """Context-pool wrapper that owns ``batch_size`` :class:`Engine` instances.

    Concurrency model:

    * ``_idle`` is an :class:`asyncio.Queue` of currently-free engines.
      :meth:`submit` / :meth:`synthesize_stream` ``await`` on it, so
      excess callers back-pressure on the queue rather than racing
      for the GPU.
    * When ``batch_size=1`` the class still works — it's equivalent
      to an ``asyncio.Lock``-guarded single engine, but with the
      future-proof API. Serve code can default to 1 for low-RAM
      deployments without a code change.

    Lifecycle: construction loads every engine sequentially (model
    load is I/O-bound + GPU warmup; parallelising here would just
    thrash the disk cache). :meth:`close` frees them in reverse
    order. The class is safe as an async context manager.
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        *,
        batch_size: int = 4,
        n_threads: int = 4,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.model_dir = Path(model_dir)
        self.batch_size = batch_size
        self.n_threads = n_threads
        self._engines: list[Engine] = []
        self._idle: Optional[asyncio.Queue[Engine]] = None
        self._idle_lock = threading.Lock()
        self._closed = False

    # --- lifecycle -----------------------------------------------------

    async def load(self) -> None:
        """Load every engine in the pool.

        Safe to call multiple times — subsequent calls are no-ops
        once the pool is populated. Loading is serialised because
        model load touches the same disk cache and GPU initialisation
        paths; doing it in parallel just slows the cold path.
        """
        if self._engines:
            return
        loop = asyncio.get_running_loop()

        def _load_one() -> Engine:
            return Engine(self.model_dir, n_threads=self.n_threads)

        for _ in range(self.batch_size):
            engine = await loop.run_in_executor(None, _load_one)
            self._engines.append(engine)

        self._idle = asyncio.Queue(maxsize=self.batch_size)
        for engine in self._engines:
            self._idle.put_nowait(engine)

    async def __aenter__(self) -> BatchEngine:
        await self.load()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release every engine in reverse order. Swallow errors so a
        partial close still frees the rest of the pool."""
        if self._closed:
            return
        self._closed = True
        while self._engines:
            engine = self._engines.pop()
            with contextlib.suppress(Exception):
                engine.close()
        self._idle = None

    # --- introspection -------------------------------------------------

    @property
    def sample_rate(self) -> int:
        if not self._engines:
            raise RuntimeError("BatchEngine.load() has not been awaited yet")
        return self._engines[0].sample_rate

    def _require_idle(self) -> asyncio.Queue[Engine]:
        if self._idle is None:
            raise RuntimeError("BatchEngine.load() has not been awaited yet")
        return self._idle

    # --- one-shot ------------------------------------------------------

    async def submit(
        self,
        text: str,
        voice: Optional[Voice] = None,
        params: Optional[SynthesisParams] = None,
    ) -> SynthesisResult:
        """Run one synthesis on the next available engine.

        Waits on the idle queue if every engine in the pool is busy.
        Execution happens on a background thread via
        ``run_in_executor`` so the caller's event loop stays
        responsive while the C engine blocks.
        """
        idle = self._require_idle()
        engine = await idle.get()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: engine.synthesize(text, voice=voice, params=params),
            )
        finally:
            idle.put_nowait(engine)

    # --- streaming -----------------------------------------------------

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[Voice] = None,
        params: Optional[SynthesisParams] = None,
    ) -> AsyncIterator[SynthesisChunk]:
        """Async iterator over streaming PCM chunks.

        Acquires one engine from the pool for the full duration of
        the stream, releases it on exit / exception. The underlying
        :meth:`Engine.synthesize_stream` is a sync generator running
        on a background thread; this method bridges it onto the
        event loop via :class:`asyncio.Queue` so the caller sees a
        standard async for-loop.

        Works with every voice mode (base / clone / custom / design)
        because Phase 5B wires up the matching C streaming symbols.
        """
        idle = self._require_idle()
        engine = await idle.get()
        loop = asyncio.get_running_loop()
        # asyncio.Queue without a typed parameter because SynthesisChunk
        # and a sentinel object share the queue to signal completion.
        chunk_queue: asyncio.Queue[Any] = asyncio.Queue()
        done_sentinel = object()
        err_holder: dict[str, Optional[BaseException]] = {"err": None}

        def _producer() -> None:
            try:
                for chunk in engine.synthesize_stream(text, voice=voice, params=params):
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
            except BaseException as err:
                err_holder["err"] = err
            finally:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, done_sentinel)

        worker = loop.run_in_executor(None, _producer)

        try:
            while True:
                item = await chunk_queue.get()
                if item is done_sentinel:
                    break
                yield item
            if err_holder["err"] is not None:
                raise err_holder["err"]
        finally:
            # Make sure the producer has drained before we put the
            # engine back — otherwise the next caller could race with
            # the tail of the previous synthesis.
            await worker
            idle.put_nowait(engine)


def default_batch_size(models: Iterable[Engine]) -> int:
    """Return the recommended ``batch_size`` for the given engine list.

    Kept trivial today so callers have a single import point when
    Phase 5C adds VRAM-aware auto-sizing.
    """
    return max(1, sum(1 for _ in models))
