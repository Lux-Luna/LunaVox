"""Public dataclasses for synthesis parameters, stats, and results.

These types are the structured surface the rest of LunaVox (CLI, GUI,
tests, future serving layer) talks to. They intentionally hold native
Python values — the ctypes marshalling lives in :mod:`lunavox.runtime._capi`
and the :class:`Engine` wrapper, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
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

    This dataclass is the **single source of truth** for sampler
    defaults across CLI / HTTP / WS / GUI. All entries override fields
    via :meth:`from_overrides` rather than re-declaring defaults.
    """

    max_audio_tokens: int = 0
    temperature: float = 0.6
    top_p: float = 1.0
    top_k: int = 50
    n_threads: int = 4
    repetition_penalty: float = 1.05
    language_id: int = -1
    ref_text: Optional[str] = None

    @classmethod
    def from_overrides(cls, **overrides: Any) -> "SynthesisParams":
        """Build a fresh instance, applying only the non-``None`` overrides.

        Unknown keys raise ``TypeError`` so typos surface at the call
        site instead of silently ignoring an option. ``None`` values
        are dropped — they mean "no override, use the default" — so
        callers can pass through optional CLI / API fields directly
        without per-field ``if`` guards.
        """
        valid = {f.name for f in fields(cls)}
        filtered: dict[str, Any] = {}
        for key, value in overrides.items():
            if key not in valid:
                raise TypeError(f"Unknown SynthesisParams field: {key!r}")
            if value is not None:
                filtered[key] = value
        return cls(**filtered)


@dataclass(frozen=True)
class MemStats:
    """Process-level RSS + VRAM snapshots for one synthesize call.

    All three time points (``start`` / ``end`` / ``peak``) come from the
    same in-engine sampling loop, so derived deltas (peak − start =
    synthesis-driven growth, end − start = post-run residual) are
    self-consistent — there is no Python-side baseline to race against.

    VRAM is attributed to this process's PID via
    ``nvmlDevice*RunningProcesses``; ``vram_measured`` is the
    authoritative "did NVML return real numbers" flag. When it is
    ``False`` the ``vram_*`` fields are undefined — callers must render
    "—" / "N/A" rather than ``0.00 GB``. The old convention of using
    ``vram_peak_bytes > 0`` as the gate was incorrect: a zero reading on
    a CPU-only run is a legitimate measurement.
    """

    rss_start_bytes: int = 0
    rss_end_bytes: int = 0
    rss_peak_bytes: int = 0
    vram_start_bytes: int = 0
    vram_end_bytes: int = 0
    vram_peak_bytes: int = 0
    vram_measured: bool = False

    @property
    def rss_peak_delta_bytes(self) -> int:
        """Synthesis-driven RSS high-water growth."""
        return max(0, self.rss_peak_bytes - self.rss_start_bytes)

    @property
    def rss_leak_bytes(self) -> int:
        """RSS still above start at end-of-run — useful for leak triage."""
        return max(0, self.rss_end_bytes - self.rss_start_bytes)

    @property
    def vram_peak_delta_bytes(self) -> int:
        return max(0, self.vram_peak_bytes - self.vram_start_bytes)

    @property
    def vram_leak_bytes(self) -> int:
        return max(0, self.vram_end_bytes - self.vram_start_bytes)


@dataclass
class SynthesisStats:
    """Per-run timing + memory stats echoed from the C engine."""

    t_tokenize_ms: int = 0
    t_encode_ms: int = 0
    t_generate_ms: int = 0
    t_decode_ms: int = 0
    t_total_ms: int = 0
    audio_duration_ms: int = 0
    rtf: float = 0.0
    mem: MemStats = field(default_factory=MemStats)


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


@dataclass
class SynthesisChunk:
    """One PCM slice produced by :meth:`Engine.synthesize_stream`.

    ``audio`` is a ``numpy.float32`` ndarray — the new samples produced
    for this chunk, not a running total. ``is_last`` is ``True`` for
    exactly one chunk per stream (the terminal one). On the terminal
    chunk ``stats`` is populated with the full per-run timing and
    memory snapshot pulled from the C engine once synthesis finished;
    it is ``None`` on every intermediate chunk so callers can key the
    end-of-stream signal off either ``is_last`` or ``stats is not None``.
    """

    audio: Any
    sample_rate: int
    is_last: bool
    stats: Optional[SynthesisStats] = None


def default_params() -> SynthesisParams:
    """Return a fresh :class:`SynthesisParams` with the C-defined defaults."""
    return SynthesisParams()
