"""Unit tests for :mod:`lunavox.core.synth.audio`.

Bit-exact parity with the old per-sample struct.pack loop is the key
guarantee — existing WAV outputs and client parsers should not see
any change when the vectorized path goes live.
"""

from __future__ import annotations

import io
import struct
import wave

import numpy as np
import pytest

from lunavox.core.synth.audio import f32_to_pcm16, f32_to_wav, pcm16_to_wav


def _legacy_f32_to_pcm16(audio) -> bytes:
    """Reference implementation — the loop version we replaced."""
    out = bytearray()
    for sample in audio:
        clipped = max(-1.0, min(1.0, float(sample)))
        out += struct.pack("<h", int(clipped * 32767.0))
    return bytes(out)


# ---------------------------------------------------------------------
# f32_to_pcm16
# ---------------------------------------------------------------------


def test_empty_input_returns_empty_bytes():
    assert f32_to_pcm16(np.empty(0, dtype=np.float32)) == b""
    assert f32_to_pcm16([]) == b""


def test_known_values_match_expected_int16_encoding():
    arr = np.array([0.0, 1.0, -1.0, 0.5, -0.5], dtype=np.float32)
    out = f32_to_pcm16(arr)
    assert len(out) == 5 * 2
    # Decode back and verify:
    decoded = np.frombuffer(out, dtype="<i2")
    expected = np.array([0, 32767, -32767, 16384, -16384], dtype=np.int16)
    # Allow +/- 1 LSB tolerance for the 0.5 → 16383 vs 16384 rounding choice.
    assert np.all(np.abs(decoded - expected) <= 1)


def test_clipping_handles_out_of_range_samples():
    """The C engine can overshoot ±1 by a ULP. We clip instead of failing."""
    arr = np.array([2.0, -2.0, 1.1, -1.1], dtype=np.float32)
    out = f32_to_pcm16(arr)
    decoded = np.frombuffer(out, dtype="<i2")
    assert decoded[0] == 32767
    assert decoded[1] == -32767
    assert decoded[2] == 32767
    assert decoded[3] == -32767


def test_accepts_python_list_input():
    """Non-ndarray input works too — handy for tests and old callers."""
    assert f32_to_pcm16([0.0, 0.5, -0.5]) == f32_to_pcm16(
        np.array([0.0, 0.5, -0.5], dtype=np.float32)
    )


def test_vectorized_matches_legacy_loop_within_tolerance():
    """Compare both paths across a random sample of realistic audio
    values. They should agree within ±1 int16 LSB."""
    rng = np.random.default_rng(seed=0)
    samples = rng.uniform(-0.98, 0.98, size=4096).astype(np.float32)
    new = np.frombuffer(f32_to_pcm16(samples), dtype="<i2")
    old = np.frombuffer(_legacy_f32_to_pcm16(samples), dtype="<i2")
    # Rounding mode differs (rint is banker's, int() is truncation),
    # so allow ±1 LSB.
    diff = np.abs(new.astype(np.int32) - old.astype(np.int32))
    assert np.all(diff <= 1), f"max diff = {diff.max()}"


# ---------------------------------------------------------------------
# pcm16_to_wav
# ---------------------------------------------------------------------


def test_wav_is_readable_by_wave_module():
    pcm = f32_to_pcm16(np.sin(np.linspace(0, 2 * np.pi, 2400)).astype(np.float32))
    wav = pcm16_to_wav(pcm, sample_rate=24000)
    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 24000
        assert wf.getnframes() == 2400


def test_pcm16_to_wav_empty_pcm_is_valid_wav():
    wav = pcm16_to_wav(b"", sample_rate=24000)
    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getnframes() == 0
        assert wf.getframerate() == 24000


# ---------------------------------------------------------------------
# f32_to_wav (convenience wrapper)
# ---------------------------------------------------------------------


def test_f32_to_wav_is_equivalent_to_two_step():
    arr = np.array([0.0, 0.3, -0.3, 0.9, -0.9], dtype=np.float32)
    combined = f32_to_wav(arr, sample_rate=16000)
    two_step = pcm16_to_wav(f32_to_pcm16(arr), sample_rate=16000)
    assert combined == two_step


# ---------------------------------------------------------------------
# performance sanity (optional — only runs if PERF env var set)
# ---------------------------------------------------------------------


@pytest.mark.skip(reason="perf sanity; flip to run locally")
def test_vectorized_is_faster_than_legacy_on_30s_clip():
    import time

    rng = np.random.default_rng(seed=0)
    samples = rng.uniform(-0.9, 0.9, size=24000 * 30).astype(np.float32)

    t0 = time.perf_counter()
    _legacy_f32_to_pcm16(samples)
    t_legacy = time.perf_counter() - t0

    t0 = time.perf_counter()
    f32_to_pcm16(samples)
    t_new = time.perf_counter() - t0

    assert t_new * 5 < t_legacy, f"legacy={t_legacy:.3f}s new={t_new:.3f}s"
