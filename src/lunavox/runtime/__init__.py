"""In-process runtime bindings for the LunaVox C++ engine.

The public surface is intentionally small:

* :class:`Engine` — RAII wrapper around the C++ engine handle. One
  ``synthesize(text, voice, params)`` entry point covers every mode.
* :class:`Voice` — first-class voice selector. Use the factory
  classmethods (``Voice.base()``, ``Voice.clone_file(path)``,
  ``Voice.custom(speaker, instruct=...)``, ``Voice.design(instruct=...)``).
* :class:`SynthesisParams` / :class:`SynthesisResult` / :class:`SynthesisStats`
  — structured request/response/telemetry dataclasses.
* :class:`SynthesisMode` — enum echoed back on each result for logging.
* :class:`LunavoxLibraryError` / :class:`LunavoxSynthesisError` — the
  two exception classes the engine raises.
* :func:`library_path` — introspection for the dlopen'd library path.

Logging is an :class:`Engine` instance method (``engine.on_log(cb)``);
there is no module-global log callback. Every voice mode goes through
:meth:`Engine.synthesize` — there is no family of ``synthesize_*``
variants to remember.
"""

from __future__ import annotations

from ._capi import library_path
from .engine import Engine, LogCallback
from .errors import LunavoxLibraryError, LunavoxSynthesisError
from .params import (
    SynthesisChunk,
    SynthesisMode,
    SynthesisParams,
    SynthesisResult,
    SynthesisStats,
    default_params,
)
from .voice import Voice

__all__ = [
    "Engine",
    "LogCallback",
    "LunavoxLibraryError",
    "LunavoxSynthesisError",
    "SynthesisChunk",
    "SynthesisMode",
    "SynthesisParams",
    "SynthesisResult",
    "SynthesisStats",
    "Voice",
    "default_params",
    "library_path",
]
