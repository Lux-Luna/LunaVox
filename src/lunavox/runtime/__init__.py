"""In-process runtime bindings for the LunaVox C++ engine.

``lunavox.runtime.binding`` owns the ctypes wrapper that loads
``liblunavox.{dll,so,dylib}`` and exposes the :class:`Engine` class. Call
sites (GUI, Python scripts, future notebooks) import from here rather
than shelling out to ``lunavox-cli`` so synthesis, stats, and logs all
flow through a structured API instead of stdout scraping.
"""

from __future__ import annotations

from .binding import (
    Engine,
    LunavoxLibraryError,
    LunavoxSynthesisError,
    SynthesisMode,
    SynthesisParams,
    SynthesisResult,
    SynthesisStats,
    default_params,
    library_path,
    set_log_callback,
)

__all__ = [
    "Engine",
    "LunavoxLibraryError",
    "LunavoxSynthesisError",
    "SynthesisMode",
    "SynthesisParams",
    "SynthesisResult",
    "SynthesisStats",
    "default_params",
    "library_path",
    "set_log_callback",
]
