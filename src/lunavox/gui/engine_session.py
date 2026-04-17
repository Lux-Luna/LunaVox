"""Optional pre-loaded engine holder shared by all GUI views.

The GUI's default behaviour is cold-load-per-Generate: each click
spins up its own :class:`Engine` and tears it down on exit. When the
user ticks the "Pre-load" checkbox next to the model dropdown, we
call :meth:`load` synchronously on a background thread — once it
returns, subsequent Generates hit :meth:`acquire` which yields the
cached engine and skips the multi-second cold-load penalty.

The checkbox reflects the cache directly: presence of a cached
engine = checked, no cache = unchecked. :meth:`acquire` never
promotes a throwaway run into the cache on its own; caching only
happens via an explicit :meth:`load` call. This keeps the UX honest
— the box never spontaneously "becomes checked" because some other
code path happened to touch the session.

The Engine itself lives on the app, not on SynthView, because the
app destroys and recreates views whenever the user flips sidebar
tabs — pinning the engine there would defeat the whole point.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Optional

from lunavox.runtime import Engine

EngineFactory = Callable[[Path, int], Engine]


class EngineSession:
    """Owns the optional cached :class:`Engine` for the GUI.

    Takes an ``engine_factory`` so tests can swap in a fake Engine
    without a real native-library load.
    """

    def __init__(self, engine_factory: EngineFactory = Engine) -> None:
        self._factory = engine_factory
        self._engine: Optional[Engine] = None
        # (absolute model_dir, n_threads) — the tuple the cached
        # engine was constructed with. A mismatch in acquire() falls
        # through to a throwaway build without disturbing this slot.
        self._key: Optional[tuple[Path, int]] = None
        # load()/unload() run on background threads spawned by the
        # UI; acquire() also runs from a worker. The lock guards
        # the swap points so neither racer reads a half-torn state.
        self._lock = threading.Lock()

    # --- public state -------------------------------------------------

    @property
    def loaded_model(self) -> Optional[Path]:
        return self._key[0] if self._key is not None else None

    def is_loaded_for(self, model_dir: Path, n_threads: int) -> bool:
        with self._lock:
            return self._engine is not None and self._key == (
                Path(model_dir),
                int(n_threads),
            )

    # --- mutation -----------------------------------------------------

    def load(self, model_dir: Path, n_threads: int) -> None:
        """Build a fresh engine and cache it. Releases any prior cache.

        Blocks for the full native-library load — GUI callers must
        dispatch this from a worker thread so the Tk main loop stays
        responsive.
        """
        resolved = Path(model_dir)
        key = (resolved, int(n_threads))
        # Construct outside the lock: the factory is the slow step
        # (GGUF weights + ONNX warmup) and has no reason to hold the
        # lock against acquire() callers on the current cache.
        new_engine = self._factory(resolved, int(n_threads))
        with self._lock:
            old = self._engine
            self._engine = new_engine
            self._key = key
        if old is not None:
            # Closing outside the lock is safe because we've already
            # replaced it in the session state — no future caller can
            # acquire this instance.
            with contextlib.suppress(Exception):
                old.close()

    def unload(self) -> None:
        """Release the cached engine (idempotent)."""
        with self._lock:
            old = self._engine
            self._engine = None
            self._key = None
        if old is not None:
            with contextlib.suppress(Exception):
                old.close()

    # --- acquisition --------------------------------------------------

    @contextlib.contextmanager
    def acquire(self, model_dir: Path, n_threads: int) -> Iterator[Engine]:
        """Yield an Engine for ``(model_dir, n_threads)``.

        * Cached engine matches the key → yield it as-is (no close on
          exit). The cache is not touched.
        * No cache, or cache key differs → build a throwaway engine
          and close it on exit. **The cached engine, if any, is left
          intact** — we do not silently replace the user's explicit
          pre-load with whatever ad-hoc model the current Generate
          asked for.
        """
        resolved = Path(model_dir)
        key = (resolved, int(n_threads))
        with self._lock:
            cached = self._engine if self._key == key else None
        if cached is not None:
            yield cached
            return

        engine = self._factory(resolved, int(n_threads))
        try:
            yield engine
        finally:
            with contextlib.suppress(Exception):
                engine.close()
