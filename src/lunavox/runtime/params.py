"""Public dataclasses for synthesis parameters, stats, and results.

These types are the structured surface the rest of LunaVox (CLI, GUI,
tests, future serving layer) talks to. They intentionally hold native
Python values — the ctypes marshalling lives in :mod:`lunavox.runtime._capi`
and the :class:`Engine` wrapper, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class SynthesisMode(Enum):
    """Which voice path produced a :class:`SynthesisResult`.

    Only the four modes the Python API actually exposes are listed.
    The C ABI also accepts raw sample/embedding clone calls, but those
    are not currently reachable from Python — adding them means adding
    a :class:`~lunavox.runtime.voice.Voice` classmethod, not mutating
    this enum independently.
    """

    BASE = "base"
    CLONE_FILE = "clone_file"
    CUSTOM = "custom"
    DESIGN = "design"


@dataclass
class SynthesisParams:
    """Python-side mirror of ``LunavoxSynthesisParams``.

    Defaults match ``lunavox_default_params`` in the C API; overriding
    any field produces a fresh struct passed to the next synthesize
    call. The ``ref_text`` field is an optional prompt hint used by
    voice-clone and voice-design flows.
    """

    max_audio_tokens: int = 0
    temperature: float = 0.6
    top_p: float = 1.0
    top_k: int = 50
    n_threads: int = 4
    repetition_penalty: float = 1.05
    language_id: int = -1
    ref_text: Optional[str] = None


@dataclass
class SynthesisStats:
    """Per-run timing and memory stats echoed from the C engine."""

    t_tokenize_ms: int = 0
    t_encode_ms: int = 0
    t_generate_ms: int = 0
    t_decode_ms: int = 0
    t_total_ms: int = 0
    audio_duration_ms: int = 0
    rtf: float = 0.0
    rss_peak_bytes: int = 0
    rss_end_bytes: int = 0


@dataclass
class SynthesisResult:
    """Structured output of a single synthesize call.

    ``audio`` is a ``numpy.float32`` ndarray in [-1, 1] (mono). Typed as
    ``Any`` here so importing this module does not pay the numpy import
    cost — the actual array is only materialised by
    :class:`~lunavox.runtime.engine.Engine` when it calls into the C API.
    """

    audio: Any
    sample_rate: int
    stats: SynthesisStats
    mode: SynthesisMode


def default_params() -> SynthesisParams:
    """Return a fresh :class:`SynthesisParams` with the C-defined defaults."""
    return SynthesisParams()
