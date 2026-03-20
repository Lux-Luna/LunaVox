#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime as ort


def eprint(msg: str) -> None:
    print(msg)


def _load_session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 4
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def detect_codec_grid(codec_path: Path) -> Tuple[int, int, List[int]]:
    sess = _load_session(codec_path)
    inp = sess.get_inputs()[0].name

    ok_lengths: List[int] = []
    # Start from 19200 so common valid points like 24000/25920 are sampled.
    for n in range(19200, 42001, 320):
        x = np.zeros((1, n), dtype=np.float32)
        try:
            y = sess.run(None, {inp: x})[0]
            if y.ndim != 3 or y.shape[0] != 1:
                raise RuntimeError(f"Unexpected codec output shape at n={n}: {y.shape}")
            ok_lengths.append(n)
            if len(ok_lengths) >= 8:
                break
        except Exception:
            continue

    if len(ok_lengths) < 2:
        raise RuntimeError(
            "Codec ONNX runtime self-check failed: cannot find >=2 valid input lengths in probe range"
        )

    diffs = [ok_lengths[i + 1] - ok_lengths[i] for i in range(len(ok_lengths) - 1)]
    stride = diffs[0]
    if any(d != stride for d in diffs[1:]):
        raise RuntimeError(
            f"Codec ONNX runtime self-check failed: non-uniform valid-length stride, lengths={ok_lengths}"
        )

    base = ok_lengths[0]

    # Long-length sanity check (clone references are often long).
    n_long = base + stride * 64
    y_long = sess.run(None, {inp: np.zeros((1, n_long), dtype=np.float32)})[0]
    if y_long.ndim != 3 or y_long.shape[0] != 1:
        raise RuntimeError(f"Codec long-length output shape invalid: {y_long.shape}")

    return base, stride, ok_lengths


def validate_speaker(speaker_path: Path) -> None:
    sess = _load_session(speaker_path)
    inp = sess.get_inputs()[0].name
    x = np.zeros((1, 100, 128), dtype=np.float32)
    y = sess.run(None, {inp: x})[0]
    if y.ndim < 2 or y.shape[0] != 1:
        raise RuntimeError(f"Speaker ONNX output shape invalid: {y.shape}")


def validate_decoder_io(decoder_path: Path) -> None:
    sess = _load_session(decoder_path)
    input_names = {x.name for x in sess.get_inputs()}
    output_names = {x.name for x in sess.get_outputs()}

    required_inputs = {
        "audio_codes",
        "pre_conv_history",
        "latent_buffer",
        "conv_history",
        "is_last",
    }
    required_outputs = {
        "final_wav",
        "valid_samples",
        "next_pre_conv_history",
        "next_latent_buffer",
        "next_conv_history",
    }

    missing_in = sorted(required_inputs - input_names)
    missing_out = sorted(required_outputs - output_names)
    if missing_in:
        raise RuntimeError(f"Decoder ONNX missing required inputs: {missing_in}")
    if missing_out:
        raise RuntimeError(f"Decoder ONNX missing required outputs: {missing_out}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate local ONNX artifacts for lunavox runtime")
    ap.add_argument("--models-dir", required=True, help="models directory containing exported ONNX files")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    models_dir = Path(args.models_dir).resolve()

    codec = models_dir / "qwen3_tts_codec_encoder.fp16.onnx"
    speaker = models_dir / "qwen3_tts_speaker_encoder.fp16.onnx"
    decoder = models_dir / "qwen3_tts_decoder.fp16.onnx"

    missing = [p for p in (codec, speaker, decoder) if not p.exists()]
    if missing:
        raise RuntimeError("Missing ONNX files for validation:\n" + "\n".join(str(p) for p in missing))

    base, stride, ok_lengths = detect_codec_grid(codec)
    validate_speaker(speaker)
    validate_decoder_io(decoder)

    eprint("[ok] ONNX runtime validation passed")
    eprint(f"[ok] codec valid-length probe: base={base}, stride={stride}, samples={ok_lengths}")
    eprint("[ok] speaker output shape check passed")
    eprint("[ok] decoder I/O signature check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
