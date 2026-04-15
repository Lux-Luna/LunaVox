"""Structured schema for the ``--stats-json`` output and C API audio
stats.

Three producers emit the same shape:

1. ``src/main.cpp`` → writes a JSON file when the user passes
   ``--stats-json report.json``.
2. ``src/lunavox_c_api.cpp::to_c_audio`` → fills
   ``LunavoxAudio`` with the same fields so the Python ctypes binding
   gets them as struct members.
3. ``src/lunavox/runtime/binding.py::SynthesisStats`` → the Python
   dataclass the GUI and any script consumes.

This module ties all three together with :class:`StatsJSON` (the
top-level object) and :class:`RunStats` (per-run entry). Consumers like
``tests/bench_baseline.py`` and the GUI can ``from lunavox.core.stats_schema
import StatsJSON`` instead of reaching into free-form dicts.

The schema is intentionally flat and boring: adding a new field means
editing this file *and* ``src/main.cpp`` + ``src/lunavox_c_api.h`` in the
same commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class TimingMs(TypedDict, total=False):
    """Per-stage wall timings in milliseconds. Fields that the engine
    does not measure for a given synth path are 0, not absent."""

    tokenize: int
    encode: int
    generate: int
    decode: int
    total: int
    # Wall time from synth start to the moment the first decoder chunk
    # finishes and the first PCM samples become available (streaming
    # pipeline). 0 for code paths that don't drive the threaded decoder.
    first_audio: int

    # Fine-grained sub-timings populated when the engine instrumentation
    # is compiled with the detailed-timing flag. Consumers should treat
    # them as optional and defensive.
    llama_prefill: int
    llama_decode_loop: int
    talker_post: int
    predictor_sample: int
    talker_decode: int
    decoder_tensor_prep: int
    decoder_ort_run: int
    decoder_tensor_extract: int
    decoder_state_trim: int
    pcm_gather: int


class StreamStats(TypedDict, total=False):
    """Streaming pipeline diagnostics. Populated for every synth call —
    the threaded decoder is always on."""

    first_chunk_frames: int
    t_first_audio_ms: int


class MemoryBytes(TypedDict, total=False):
    """Process memory snapshots in bytes, sampled at specific run
    checkpoints. ``rss_peak`` is the high-water mark seen during the
    synth call; ``rss_start`` / ``rss_end`` bracket the call."""

    rss_start: int
    rss_end: int
    rss_peak: int
    phys_start: int
    phys_end: int
    phys_peak: int


class RunStats(TypedDict, total=False):
    """One element of ``StatsJSON['runs']``."""

    run_id: int
    sample_rate: int
    n_samples: int
    audio_duration_s: float
    rtf: float
    timing_ms: TimingMs
    stream: StreamStats
    mem: MemoryBytes
    effective_language_id: int


class StatsJSON(TypedDict, total=False):
    """Top-level ``--stats-json`` payload.

    ``t_load_ms`` is the wall time spent inside a single
    ``Engine::load_models`` call (includes warmup). ``t_warmup_ms`` is
    the warmup portion of that load time, broken out so cold-start
    regressions can be diagnosed without re-parsing DEBUG logs. ``runs``
    is the sequence of synth calls that followed — order matches the
    CLI's ``--repeat`` loop.
    """

    t_load_ms: int
    t_warmup_ms: int
    runs: list[RunStats]


# ---------------------------------------------------------------------------
# Lightweight parser used by bench_baseline and the GUI. It's not a schema
# validator — just a structural load that catches the common "file exists
# but is truncated / wrong shape" failure mode early with a clear message.
# ---------------------------------------------------------------------------


@dataclass
class ParsedStats:
    load_ms: int = 0
    warmup_ms: int = 0
    runs: list[dict] = field(default_factory=list)

    @property
    def first_run(self) -> dict:
        return self.runs[0] if self.runs else {}

    def rtf(self, index: int = 0) -> float:
        if index >= len(self.runs):
            return 0.0
        return float(self.runs[index].get("rtf", 0.0))


def parse_stats_json(data: Any) -> ParsedStats:
    """Coerce a parsed JSON dict (or str path) into a :class:`ParsedStats`.

    Raises ``ValueError`` with the offending field name on shape mismatch.
    """
    import json
    from pathlib import Path

    if isinstance(data, (str, Path)):
        with open(data, "r", encoding="utf-8") as f:
            data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"stats JSON must be an object with t_load_ms / runs, got {type(data).__name__}"
        )

    runs = data.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError(f"stats JSON 'runs' must be a list, got {type(runs).__name__}")

    return ParsedStats(
        load_ms=int(data.get("t_load_ms", 0) or 0),
        warmup_ms=int(data.get("t_warmup_ms", 0) or 0),
        runs=list(runs),
    )
