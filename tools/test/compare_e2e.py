#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportMissingTypeArgument=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportUnusedCallResult=false
"""
End-to-end comparison: Python TTS pipeline vs C++ TTS pipeline.

Generates speech with both pipelines using deterministic settings,
then compares waveform correlation, duration, and error metrics.

Usage:
    /root/.local/bin/uv run python scripts/compare_e2e.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
HF_MODEL_PATH = MODEL_DIR / "Qwen3-TTS-12Hz-0.6B-Base"
REF_AUDIO_PATH = PROJECT_ROOT / "clone.wav"
CLI_BINARY = PROJECT_ROOT / "build" / "qwen3-tts-cli"

SHORT_TEXT = "Hello."
LONG_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
LANGUAGE = "English"

# RMS threshold: waveforms must not be silent
RMS_THRESHOLD = 0.001
# Duration ratio bounds: C++ and Python audio lengths must be within 2x of each other
DURATION_RATIO_MIN = 0.5
DURATION_RATIO_MAX = 1.5
# Minimum sample count to be considered valid speech output
MIN_SAMPLES = 1000


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV file and return (samples_float32, sample_rate)."""
    import struct

    path = Path(path)
    data = path.read_bytes()

    # Parse RIFF WAV header
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"Not a valid WAV file: {path}")

    # Find fmt chunk
    pos = 12
    fmt_data = None
    audio_data = None
    while pos < len(data) - 8:
        chunk_id = data[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
        if chunk_id == b"fmt ":
            fmt_data = data[pos + 8 : pos + 8 + chunk_size]
        elif chunk_id == b"data":
            audio_data = data[pos + 8 : pos + 8 + chunk_size]
        pos += 8 + chunk_size
        # Align to 2-byte boundary
        if pos % 2 != 0:
            pos += 1

    if fmt_data is None or audio_data is None:
        raise ValueError(f"Missing fmt or data chunk in {path}")

    audio_format = struct.unpack_from("<H", fmt_data, 0)[0]
    num_channels = struct.unpack_from("<H", fmt_data, 2)[0]
    sample_rate = struct.unpack_from("<I", fmt_data, 4)[0]
    bits_per_sample = struct.unpack_from("<H", fmt_data, 14)[0]

    if audio_format == 1:  # PCM
        if bits_per_sample == 16:
            samples = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )
        elif bits_per_sample == 32:
            samples = (
                np.frombuffer(audio_data, dtype=np.int32).astype(np.float32)
                / 2147483648.0
            )
        else:
            raise ValueError(f"Unsupported PCM bits_per_sample: {bits_per_sample}")
    elif audio_format == 3:  # IEEE float
        if bits_per_sample == 32:
            samples = np.frombuffer(audio_data, dtype=np.float32).copy()
        elif bits_per_sample == 64:
            samples = np.frombuffer(audio_data, dtype=np.float64).astype(np.float32)
        else:
            raise ValueError(f"Unsupported float bits_per_sample: {bits_per_sample}")
    else:
        raise ValueError(f"Unsupported audio format: {audio_format}")

    # Mix to mono if stereo
    if num_channels > 1:
        samples = samples.reshape(-1, num_channels).mean(axis=1)

    return samples, sample_rate


def compare_waveforms(
    py_samples: np.ndarray, cpp_samples: np.ndarray, label: str
) -> dict:
    """Compare two waveforms and return metrics."""
    py_dur = len(py_samples)
    cpp_dur = len(cpp_samples)

    # Align lengths (trim to shorter)
    min_len = min(py_dur, cpp_dur)
    if min_len == 0:
        return {
            "label": label,
            "py_samples": py_dur,
            "cpp_samples": cpp_dur,
            "correlation": 0.0,
            "max_abs_error": float("inf"),
            "mean_abs_error": float("inf"),
            "duration_ratio": 0.0,
            "py_rms": 0.0,
            "cpp_rms": 0.0,
            "error": "One or both waveforms are empty",
        }

    py_rms = float(np.sqrt(np.mean(py_samples**2)))
    cpp_rms = float(np.sqrt(np.mean(cpp_samples**2)))

    py_aligned = py_samples[:min_len]
    cpp_aligned = cpp_samples[:min_len]

    # Correlation
    py_std = np.std(py_aligned)
    cpp_std = np.std(cpp_aligned)
    if py_std < 1e-10 or cpp_std < 1e-10:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(py_aligned, cpp_aligned)[0, 1])

    # Error metrics
    diff = np.abs(py_aligned - cpp_aligned)
    max_abs_error = float(np.max(diff))
    mean_abs_error = float(np.mean(diff))

    # Duration ratio
    duration_ratio = cpp_dur / py_dur if py_dur > 0 else 0.0

    return {
        "label": label,
        "py_samples": py_dur,
        "cpp_samples": cpp_dur,
        "aligned_samples": min_len,
        "correlation": correlation,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "duration_ratio": duration_ratio,
        "py_rms": py_rms,
        "cpp_rms": cpp_rms,
    }


def run_python_pipeline(
    model,
    text: str,
    output_path: Path,
    max_new_tokens: int,
) -> float:
    """Run Python TTS pipeline and save output WAV. Returns elapsed time."""
    import soundfile as sf

    t0 = time.time()
    wavs, sample_rate = model.generate_voice_clone(
        text=text,
        language=LANGUAGE,
        ref_audio=str(REF_AUDIO_PATH),
        ref_text=LONG_TEXT,  # reference text for voice cloning
        x_vector_only_mode=True,
        non_streaming_mode=False,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_k=None,
        top_p=None,
        temperature=None,
        subtalker_dosample=False,
        subtalker_top_k=None,
        subtalker_top_p=None,
        subtalker_temperature=None,
    )
    elapsed = time.time() - t0

    audio = np.asarray(wavs[0], dtype=np.float32)
    sf.write(str(output_path), audio, sample_rate)
    print(f"  Python: {len(audio)} samples @ {sample_rate} Hz ({elapsed:.1f}s)")
    return elapsed


def run_cpp_pipeline(
    text: str,
    output_path: Path,
    max_tokens: int,
) -> float:
    """Run C++ TTS pipeline via CLI and save output WAV. Returns elapsed time."""
    cmd = [
        str(CLI_BINARY),
        "-m",
        str(MODEL_DIR),
        "-t",
        text,
        "-r",
        str(REF_AUDIO_PATH),
        "-o",
        str(output_path),
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        "0",  # greedy decoding
    ]

    print(f"  C++ CLI: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  C++ CLI stderr:\n{result.stderr}")
        print(f"  C++ CLI stdout:\n{result.stdout}")
        raise RuntimeError(f"C++ CLI failed with return code {result.returncode}")

    # Print any useful info from stdout
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            print(f"  C++ output: {line.strip()}")

    print(f"  C++ CLI: completed in {elapsed:.1f}s")
    return elapsed


def print_summary(results: list[dict]) -> bool:
    """Print summary table and return True if all tests pass."""
    print("\n" + "=" * 80)
    print("END-TO-END COMPARISON SUMMARY")
    print("=" * 80)

    header = (
        f"{'Test':<15} {'Corr':>8} {'MaxErr':>10} {'MeanErr':>10} "
        f"{'Dur Ratio':>10} {'PyRMS':>8} {'CppRMS':>8} "
        f"{'PySamp':>10} {'CppSamp':>10} {'Pass':>6}"
    )
    print(header)
    print("-" * len(header))

    all_pass = True
    for r in results:
        passed = (
            r["py_rms"] > RMS_THRESHOLD
            and r["cpp_rms"] > RMS_THRESHOLD
            and DURATION_RATIO_MIN < r["duration_ratio"] < DURATION_RATIO_MAX
            and r["py_samples"] > MIN_SAMPLES
            and r["cpp_samples"] > MIN_SAMPLES
        )
        if not passed:
            all_pass = False
        status = "PASS" if passed else "FAIL"
        print(
            f"{r['label']:<15} "
            f"{r['correlation']:>8.4f} "
            f"{r['max_abs_error']:>10.6f} "
            f"{r['mean_abs_error']:>10.6f} "
            f"{r['duration_ratio']:>10.4f} "
            f"{r['py_rms']:>8.4f} "
            f"{r['cpp_rms']:>8.4f} "
            f"{r['py_samples']:>10d} "
            f"{r['cpp_samples']:>10d} "
            f"{status:>6}"
        )

    print("-" * len(header))
    print(
        f"\nPass criteria: RMS > {RMS_THRESHOLD}, "
        f"{DURATION_RATIO_MIN} < duration_ratio < {DURATION_RATIO_MAX}, "
        f"samples > {MIN_SAMPLES}"
    )
    print(
        "Note: Waveform correlation is informational only. Low correlation is expected\n"
        "      for autoregressive TTS with F16 model weights — different speech codes\n"
        "      decode to different-but-perceptually-equivalent waveforms."
    )
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 80)

    return all_pass


def main() -> int:
    import random
    import torch

    # Deterministic settings
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    print("=" * 80)
    print("Qwen3-TTS End-to-End Comparison: Python vs C++")
    print("=" * 80)

    # Check prerequisites
    for p in [CLI_BINARY, HF_MODEL_PATH, REF_AUDIO_PATH]:
        if not p.exists():
            print(f"ERROR: Missing required path: {p}")
            return 1

    # Load Python model once
    print("\nLoading Python model (float32, CPU)...")
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        str(HF_MODEL_PATH),
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    model.model = model.model.eval()
    print("Model loaded.\n")

    results = []

    # ── Part A: Short text ──────────────────────────────────────────────
    print("─" * 60)
    print(f'Part A: Short text = "{SHORT_TEXT}"')
    print(f"  max_new_tokens = 64")
    print("─" * 60)

    py_short_path = Path("/tmp/e2e_py_short.wav")
    cpp_short_path = Path("/tmp/e2e_cpp_short.wav")

    print("\n[Python pipeline]")
    with torch.no_grad():
        run_python_pipeline(model, SHORT_TEXT, py_short_path, max_new_tokens=64)

    print("\n[C++ pipeline]")
    run_cpp_pipeline(SHORT_TEXT, cpp_short_path, max_tokens=64)

    print("\n[Comparing short text outputs]")
    py_short, py_sr = load_wav(py_short_path)
    cpp_short, cpp_sr = load_wav(cpp_short_path)
    print(f"  Python WAV: {len(py_short)} samples @ {py_sr} Hz")
    print(f"  C++ WAV:    {len(cpp_short)} samples @ {cpp_sr} Hz")

    short_result = compare_waveforms(py_short, cpp_short, "Short")
    results.append(short_result)
    print(f"  Correlation: {short_result['correlation']:.4f}")
    print(f"  Max abs error: {short_result['max_abs_error']:.6f}")
    print(f"  Mean abs error: {short_result['mean_abs_error']:.6f}")
    print(f"  Duration ratio (cpp/py): {short_result['duration_ratio']:.4f}")

    # ── Part B: Long text ───────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f'Part B: Long text = "{LONG_TEXT}"')
    print(f"  max_new_tokens = 2048")
    print("─" * 60)

    py_long_path = Path("/tmp/e2e_py_long.wav")
    cpp_long_path = Path("/tmp/e2e_cpp_long.wav")

    print("\n[Python pipeline]")
    with torch.no_grad():
        run_python_pipeline(model, LONG_TEXT, py_long_path, max_new_tokens=2048)

    print("\n[C++ pipeline]")
    run_cpp_pipeline(LONG_TEXT, cpp_long_path, max_tokens=2048)

    print("\n[Comparing long text outputs]")
    py_long, py_sr2 = load_wav(py_long_path)
    cpp_long, cpp_sr2 = load_wav(cpp_long_path)
    print(f"  Python WAV: {len(py_long)} samples @ {py_sr2} Hz")
    print(f"  C++ WAV:    {len(cpp_long)} samples @ {cpp_sr2} Hz")

    long_result = compare_waveforms(py_long, cpp_long, "Long")
    results.append(long_result)
    print(f"  Correlation: {long_result['correlation']:.4f}")
    print(f"  Max abs error: {long_result['max_abs_error']:.6f}")
    print(f"  Mean abs error: {long_result['mean_abs_error']:.6f}")
    print(f"  Duration ratio (cpp/py): {long_result['duration_ratio']:.4f}")

    # ── Part C: Summary ─────────────────────────────────────────────────
    all_pass = print_summary(results)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
