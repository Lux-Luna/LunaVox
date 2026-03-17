#!/usr/bin/env python3
"""
Run reproducible Lunavox end-to-end performance benchmarks for CPU/GPU backends.

This script does not rebuild anything. It only runs existing binaries, parses timing
output, and writes logs + machine-readable summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CaseResult:
    case: str
    backend_env: str
    backend_speaker: str
    backend_transformer: str
    backend_talker: str
    backend_code_predictor: str
    backend_decoder: str
    exe: str
    return_code: int
    log_path: str
    wav_path: str
    frames: Optional[int] = None
    talker_ms: Optional[float] = None
    code_pred_ms: Optional[float] = None
    code_generation_ms: Optional[int] = None
    vocoder_decode_ms: Optional[int] = None
    total_ms: Optional[int] = None
    audio_duration_s: Optional[float] = None
    rtf: Optional[float] = None
    stream_chunks: Optional[int] = None
    stream_batches: Optional[int] = None
    stream_decode_wall_ms: Optional[int] = None
    stream_overlap_ms: Optional[int] = None
    stream_overlap_ratio: Optional[float] = None
    stream_pipeline_saved_ms: Optional[int] = None


def parse_metrics(log_text: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "frames": None,
        "talker_ms": None,
        "code_pred_ms": None,
        "code_generation_ms": None,
        "vocoder_decode_ms": None,
        "total_ms": None,
        "audio_duration_s": None,
        "rtf": None,
        "stream_chunks": None,
        "stream_batches": None,
        "stream_decode_wall_ms": None,
        "stream_overlap_ms": None,
        "stream_overlap_ratio": None,
        "stream_pipeline_saved_ms": None,
    }

    m = re.search(r"Detailed Generation Timing \((\d+) frames\)", log_text)
    if m:
        out["frames"] = int(m.group(1))

    m = re.search(r"Talker forward_step .*?\n\s*Total:\s*([0-9.]+)\s*ms", log_text, re.S)
    if m:
        out["talker_ms"] = float(m.group(1))

    m = re.search(r"Code predictor \(total / per-frame\):.*?\n\s*Total:\s*([0-9.]+)\s*ms", log_text, re.S)
    if m:
        out["code_pred_ms"] = float(m.group(1))

    m = re.search(r"Code generation:\s*(\d+)\s*ms", log_text)
    if m:
        out["code_generation_ms"] = int(m.group(1))

    m = re.search(r"Vocoder decode:\s*(\d+)\s*ms", log_text)
    if m:
        out["vocoder_decode_ms"] = int(m.group(1))

    totals = re.findall(r"\n\s*Total:\s*(\d+)\s*ms", log_text)
    if totals:
        out["total_ms"] = int(totals[-1])

    m = re.search(r"Audio duration:\s*([0-9.]+)\s*s", log_text)
    if m:
        out["audio_duration_s"] = float(m.group(1))

    m = re.search(r"RTF=([0-9.]+)", log_text)
    if m:
        out["rtf"] = float(m.group(1))

    # New format with explicit batch count.
    m = re.search(
        r"chunks=(\d+),\s*batches=(\d+),\s*decode-wall=(\d+)\s*ms,\s*overlap=(\d+)\s*ms\s*\(ratio=([0-9.]+)\),\s*pipeline-saved=(\d+)\s*ms",
        log_text,
    )
    if m:
        out["stream_chunks"] = int(m.group(1))
        out["stream_batches"] = int(m.group(2))
        out["stream_decode_wall_ms"] = int(m.group(3))
        out["stream_overlap_ms"] = int(m.group(4))
        out["stream_overlap_ratio"] = float(m.group(5))
        out["stream_pipeline_saved_ms"] = int(m.group(6))
        return out

    # Backward-compatible parser for older logs without "batches=...".
    m = re.search(
        r"chunks=(\d+),\s*decode-wall=(\d+)\s*ms,\s*overlap=(\d+)\s*ms\s*\(ratio=([0-9.]+)\),\s*pipeline-saved=(\d+)\s*ms",
        log_text,
    )
    if m:
        out["stream_chunks"] = int(m.group(1))
        out["stream_decode_wall_ms"] = int(m.group(2))
        out["stream_overlap_ms"] = int(m.group(3))
        out["stream_overlap_ratio"] = float(m.group(4))
        out["stream_pipeline_saved_ms"] = int(m.group(5))

    return out


def run_case(
    case: str,
    exe: Path,
    backend_env: str,
    backend_speaker: str,
    backend_transformer: str,
    backend_talker: str,
    backend_code_predictor: str,
    backend_decoder: str,
    model_dir: Path,
    text: str,
    max_tokens: int,
    threads: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    streaming_decode: bool,
    decode_chunk_frames: int,
    streaming_max_queued_chunks: int,
    streaming_decode_batch_chunks: int,
    out_dir: Path,
) -> CaseResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{case}.wav"
    log_path = out_dir / f"{case}.log"

    cmd: List[str] = [
        str(exe),
        "-m",
        str(model_dir),
        "-t",
        text,
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(temperature),
        "--top-k",
        str(top_k),
        "--repetition-penalty",
        str(repetition_penalty),
        "-o",
        str(wav_path),
    ]
    if threads > 0:
        cmd += ["--threads", str(threads)]
    if streaming_decode:
        cmd += ["--streaming-decode", "--decode-chunk-frames", str(decode_chunk_frames)]
    if streaming_decode_batch_chunks > 0:
        cmd += ["--streaming-decode-batch-chunks", str(streaming_decode_batch_chunks)]

    env = os.environ.copy()
    env["QWEN3_TTS_BACKEND"] = backend_env
    if backend_speaker:
        env["QWEN3_TTS_BACKEND_SPEAKER_ENCODER"] = backend_speaker
    if backend_transformer:
        env["QWEN3_TTS_BACKEND_TRANSFORMER"] = backend_transformer
    if backend_talker:
        env["QWEN3_TTS_BACKEND_TALKER"] = backend_talker
    if backend_code_predictor:
        env["QWEN3_TTS_BACKEND_CODE_PREDICTOR"] = backend_code_predictor
    if backend_decoder:
        env["QWEN3_TTS_BACKEND_CODEC_DECODER"] = backend_decoder
    if streaming_max_queued_chunks > 0:
        env["QWEN3_TTS_STREAMING_MAX_QUEUED_CHUNKS"] = str(streaming_max_queued_chunks)

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    log_text = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(log_text, encoding="utf-8", errors="replace")

    metrics = parse_metrics(log_text)
    return CaseResult(
        case=case,
        backend_env=backend_env,
        backend_speaker=backend_speaker,
        backend_transformer=backend_transformer,
        backend_talker=backend_talker,
        backend_code_predictor=backend_code_predictor,
        backend_decoder=backend_decoder,
        exe=str(exe),
        return_code=proc.returncode,
        log_path=str(log_path),
        wav_path=str(wav_path),
        frames=metrics["frames"],  # type: ignore[arg-type]
        talker_ms=metrics["talker_ms"],  # type: ignore[arg-type]
        code_pred_ms=metrics["code_pred_ms"],  # type: ignore[arg-type]
        code_generation_ms=metrics["code_generation_ms"],  # type: ignore[arg-type]
        vocoder_decode_ms=metrics["vocoder_decode_ms"],  # type: ignore[arg-type]
        total_ms=metrics["total_ms"],  # type: ignore[arg-type]
        audio_duration_s=metrics["audio_duration_s"],  # type: ignore[arg-type]
        rtf=metrics["rtf"],  # type: ignore[arg-type]
        stream_chunks=metrics["stream_chunks"],  # type: ignore[arg-type]
        stream_batches=metrics["stream_batches"],  # type: ignore[arg-type]
        stream_decode_wall_ms=metrics["stream_decode_wall_ms"],  # type: ignore[arg-type]
        stream_overlap_ms=metrics["stream_overlap_ms"],  # type: ignore[arg-type]
        stream_overlap_ratio=metrics["stream_overlap_ratio"],  # type: ignore[arg-type]
        stream_pipeline_saved_ms=metrics["stream_pipeline_saved_ms"],  # type: ignore[arg-type]
    )


def write_markdown(results: List[CaseResult], path: Path) -> None:
    lines: List[str] = []
    lines.append("# Lunavox Benchmark Summary")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("| case | backend | talker | code-pred | decoder | return | frames | gen(ms) | decode(ms) | total(ms) | audio(s) | RTF | overlap(r) | saved(ms) | chunks/batches |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r.case} | {r.backend_env} | "
            f"{r.backend_talker or r.backend_transformer or '-'} | "
            f"{r.backend_code_predictor or r.backend_transformer or '-'} | "
            f"{r.backend_decoder or '-'} | "
            f"{r.return_code} | "
            f"{'' if r.frames is None else r.frames} | "
            f"{'' if r.code_generation_ms is None else r.code_generation_ms} | "
            f"{'' if r.vocoder_decode_ms is None else r.vocoder_decode_ms} | "
            f"{'' if r.total_ms is None else r.total_ms} | "
            f"{'' if r.audio_duration_s is None else f'{r.audio_duration_s:.2f}'} | "
            f"{'' if r.rtf is None else f'{r.rtf:.3f}'} | "
            f"{'' if r.stream_overlap_ratio is None else f'{r.stream_overlap_ratio:.2f}'} | "
            f"{'' if r.stream_pipeline_saved_ms is None else r.stream_pipeline_saved_ms} | "
            f"{'' if r.stream_chunks is None else r.stream_chunks}/"
            f"{'' if r.stream_batches is None else r.stream_batches} |"
        )
    lines.append("")
    lines.append("## Logs")
    lines.append("")
    for r in results:
        lines.append(f"- `{r.case}`: `{r.log_path}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--cpu-exe", default="build-cpu-timing/qwen3-tts-cli.exe")
    parser.add_argument("--gpu-exe", default="build-cuda-timing/qwen3-tts-cli.exe")
    parser.add_argument("--text", default="Performance regression run to validate streaming overlap.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--threads", type=int, default=0, help="0 means backend default")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--backend-speaker", default="", help="override Speaker Encoder backend")
    parser.add_argument("--backend-transformer", default="", help="override Talker/Code Predictor backend")
    parser.add_argument("--backend-talker", default="", help="override Talker backend")
    parser.add_argument("--backend-code-predictor", default="", help="override Code Predictor backend")
    parser.add_argument("--backend-decoder", default="", help="override Codec Decoder backend")
    parser.add_argument("--streaming-decode", action="store_true", help="enable streaming decode overlap")
    parser.add_argument("--decode-chunk-frames", type=int, default=32, help="frames per decoder chunk")
    parser.add_argument("--streaming-max-queued-chunks", type=int, default=0, help="0 means runtime default")
    parser.add_argument("--streaming-decode-batch-chunks", type=int, default=1, help="decode worker batch size in chunks")
    parser.add_argument("--out-dir", default="perf")
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    model_dir = (root / args.model_dir).resolve()
    out_dir = (root / args.out_dir).resolve()

    results: List[CaseResult] = []

    cpu_exe = (root / args.cpu_exe).resolve()
    if cpu_exe.exists():
        results.append(
            run_case(
                case="cpu",
                exe=cpu_exe,
                backend_env="cpu",
                backend_speaker=args.backend_speaker,
                backend_transformer=args.backend_transformer,
                backend_talker=args.backend_talker,
                backend_code_predictor=args.backend_code_predictor,
                backend_decoder=args.backend_decoder,
                model_dir=model_dir,
                text=args.text,
                max_tokens=args.max_tokens,
                threads=args.threads,
                temperature=args.temperature,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                streaming_decode=args.streaming_decode,
                decode_chunk_frames=args.decode_chunk_frames,
                streaming_max_queued_chunks=args.streaming_max_queued_chunks,
                streaming_decode_batch_chunks=args.streaming_decode_batch_chunks,
                out_dir=out_dir,
            )
        )
    else:
        print(f"[warn] CPU exe not found: {cpu_exe}")

    gpu_exe = (root / args.gpu_exe).resolve()
    if not args.skip_gpu:
        if gpu_exe.exists():
            results.append(
                run_case(
                    case="gpu",
                    exe=gpu_exe,
                    backend_env="gpu",
                    backend_speaker=args.backend_speaker,
                    backend_transformer=args.backend_transformer,
                    backend_talker=args.backend_talker,
                    backend_code_predictor=args.backend_code_predictor,
                    backend_decoder=args.backend_decoder,
                    model_dir=model_dir,
                    text=args.text,
                    max_tokens=args.max_tokens,
                    threads=args.threads,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    streaming_decode=args.streaming_decode,
                    decode_chunk_frames=args.decode_chunk_frames,
                    streaming_max_queued_chunks=args.streaming_max_queued_chunks,
                    streaming_decode_batch_chunks=args.streaming_decode_batch_chunks,
                    out_dir=out_dir,
                )
            )
        else:
            print(f"[warn] GPU exe not found: {gpu_exe}")

    if not results:
        print("[error] no runnable benchmark cases")
        return 2

    json_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(results, md_path)

    for r in results:
        print(
            f"[{r.case}] rc={r.return_code} "
            f"gen={r.code_generation_ms}ms decode={r.vocoder_decode_ms}ms "
            f"total={r.total_ms}ms rtf={r.rtf} "
            f"overlap={r.stream_overlap_ratio} saved={r.stream_pipeline_saved_ms}ms "
            f"chunks={r.stream_chunks} batches={r.stream_batches}"
        )

    return 0 if all(r.return_code == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
