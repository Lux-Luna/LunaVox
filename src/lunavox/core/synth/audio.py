"""Vectorized audio encoding helpers shared by every entry.

Replaces three near-identical per-sample loops that lived in the CLI,
the HTTP handler, and the WebSocket handler. The loop versions ran
``struct.pack("<h", int(clipped * 32767.0))`` per sample — correct
but O(n) in Python bytecode, which costs real time on a 30 s clip
(~720 K samples at 24 kHz).

The numpy path is a single ``clip → scale → cast`` that stays in C
and is roughly two orders of magnitude faster on typical audio
lengths. It also happens to be simpler to read.
"""

from __future__ import annotations

import io
import wave
from typing import Any

import numpy as np


def f32_to_pcm16(audio: Any) -> bytes:
    """Convert a float32 array in [-1, 1] to raw int16 LE PCM bytes.

    Accepts any array-like (ndarray, list, tuple, ctypes array) for
    compatibility with callers that haven't materialised a real
    ndarray yet. Clips out-of-range samples rather than raising — the
    C engine can occasionally overshoot ±1 by a ULP on rounding, and
    failing the whole stream for that would be user-hostile.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return b""
    # numpy requires a writable buffer for in-place clip; asarray may
    # have returned a view of the caller's memory (e.g. when ``audio``
    # is already an ndarray). Copy unconditionally — the cost is ~1 µs
    # per KB and guarantees correctness.
    clipped = arr.clip(-1.0, 1.0)
    scaled = np.rint(clipped * 32767.0).astype(np.int16, copy=False)
    return scaled.tobytes()


def pcm16_to_wav(pcm: bytes, sample_rate: int, *, channels: int = 1) -> bytes:
    """Wrap raw int16 LE PCM bytes in a canonical RIFF/WAVE container.

    Uses :mod:`wave` rather than writing the header by hand — same
    48-byte overhead but far less likely to drift from the spec.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm)
    return buf.getvalue()


def f32_to_wav(audio: Any, sample_rate: int, *, channels: int = 1) -> bytes:
    """Convenience: ``f32_to_pcm16`` + ``pcm16_to_wav`` in one call.

    Most non-streaming callers want both steps; streaming callers
    use :func:`f32_to_pcm16` directly on each chunk and never build
    a WAV container.
    """
    pcm = f32_to_pcm16(audio)
    return pcm16_to_wav(pcm, sample_rate, channels=channels)
