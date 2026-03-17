#!/usr/bin/env python3
"""
Parameter sweep for Lunavox streaming pipeline.

Scans thread count + decode chunk size + streaming queue depth + decode batch size
and aggregates CPU/GPU RTF plus overlap metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class SweepRow:
    backend: str
    threads: int
    chunk_frames: int
    max_queued_chunks: int
    decode_batch_chunks: int
    total_ms: Optional[int]
    rtf: Optional[float]
    overlap_ratio: Optional[float]
    pipeline_saved_ms: Optional[int]
    summary_json: str
    log_path: str


def parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def run_once(
    root: Path,
    out_dir: Path,
    cpu_exe: str,
    gpu_exe: str,
    model_dir: str,
    text: str,
    max_tokens: int,
    threads: int,
    chunk_frames: int,
    max_queued_chunks: int,
    decode_batch_chunks: int,
    skip_gpu: bool,
    backend_talker: str,
    backend_code_predictor: str,
    backend_decoder: str,
) -> Path:
    cmd = [
        "python",
        str(root / "tools" / "perf_benchmark.py"),
        "--cpu-exe",
        cpu_exe,
        "--gpu-exe",
        gpu_exe,
        "--model-dir",
        model_dir,
        "--text",
        text,
        "--max-tokens",
        str(max_tokens),
        "--threads",
        str(threads),
        "--temperature",
        "0",
        "--top-k",
        "0",
        "--repetition-penalty",
        "1.0",
        "--streaming-decode",
        "--decode-chunk-frames",
        str(chunk_frames),
        "--streaming-max-queued-chunks",
        str(max_queued_chunks),
        "--streaming-decode-batch-chunks",
        str(decode_batch_chunks),
        "--out-dir",
        str(out_dir),
    ]
    if skip_gpu:
        cmd.append("--skip-gpu")
    if backend_talker:
        cmd += ["--backend-talker", backend_talker]
    if backend_code_predictor:
        cmd += ["--backend-code-predictor", backend_code_predictor]
    if backend_decoder:
        cmd += ["--backend-decoder", backend_decoder]

    subprocess.run(cmd, check=False)
    return out_dir / "summary.json"


def load_rows(
    summary_json: Path,
    threads: int,
    chunk_frames: int,
    max_queued_chunks: int,
    decode_batch_chunks: int,
) -> List[SweepRow]:
    if not summary_json.exists():
        return []
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    rows: List[SweepRow] = []
    for item in data:
        rows.append(
            SweepRow(
                backend=item.get("case", "unknown"),
                threads=threads,
                chunk_frames=chunk_frames,
                max_queued_chunks=max_queued_chunks,
                decode_batch_chunks=decode_batch_chunks,
                total_ms=item.get("total_ms"),
                rtf=item.get("rtf"),
                overlap_ratio=item.get("stream_overlap_ratio"),
                pipeline_saved_ms=item.get("stream_pipeline_saved_ms"),
                summary_json=str(summary_json),
                log_path=item.get("log_path", ""),
            )
        )
    return rows


def write_report(rows: List[SweepRow], out_md: Path) -> None:
    rows_sorted = sorted(
        rows,
        key=lambda r: (r.backend, 9999.0 if r.rtf is None else r.rtf),
    )
    lines = [
        "# Lunavox Streaming Sweep",
        "",
        "| backend | threads | chunk | queue | batch | total(ms) | RTF | overlap(r) | saved(ms) | log |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows_sorted:
        lines.append(
            f"| {r.backend} | {r.threads} | {r.chunk_frames} | {r.max_queued_chunks} | {r.decode_batch_chunks} | "
            f"{'' if r.total_ms is None else r.total_ms} | "
            f"{'' if r.rtf is None else f'{r.rtf:.3f}'} | "
            f"{'' if r.overlap_ratio is None else f'{r.overlap_ratio:.2f}'} | "
            f"{'' if r.pipeline_saved_ms is None else r.pipeline_saved_ms} | "
            f"`{r.log_path}` |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--cpu-exe", default="build-cpu-timing/qwen3-tts-cli.exe")
    parser.add_argument("--gpu-exe", default="build-cuda-timing/qwen3-tts-cli.exe")
    parser.add_argument("--text", default="Sweep streaming overlap parameters for stable throughput.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--threads", default="0,8,12,16")
    parser.add_argument("--chunks", default="16,32,64,96")
    parser.add_argument("--queues", default="2,4,8")
    parser.add_argument("--batches", default="1,2,4")
    parser.add_argument("--out-root", default="perf/phaseD_sweep")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--backend-talker", default="")
    parser.add_argument("--backend-code-predictor", default="")
    parser.add_argument("--backend-decoder", default="")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_root = (root / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    thread_list = parse_int_list(args.threads)
    chunk_list = parse_int_list(args.chunks)
    queue_list = parse_int_list(args.queues)
    batch_list = parse_int_list(args.batches)

    rows: List[SweepRow] = []
    for threads in thread_list:
        for chunk in chunk_list:
            for queue in queue_list:
                for batch in batch_list:
                    run_name = f"t{threads}_c{chunk}_q{queue}_b{batch}"
                    out_dir = out_root / run_name
                    summary_json = run_once(
                        root=root,
                        out_dir=out_dir,
                        cpu_exe=args.cpu_exe,
                        gpu_exe=args.gpu_exe,
                        model_dir=args.model_dir,
                        text=args.text,
                        max_tokens=args.max_tokens,
                        threads=threads,
                        chunk_frames=chunk,
                        max_queued_chunks=queue,
                        decode_batch_chunks=batch,
                        skip_gpu=args.skip_gpu,
                        backend_talker=args.backend_talker,
                        backend_code_predictor=args.backend_code_predictor,
                        backend_decoder=args.backend_decoder,
                    )
                    rows.extend(load_rows(summary_json, threads, chunk, queue, batch))

    summary_json = out_root / "summary.json"
    summary_md = out_root / "summary.md"
    summary_json.write_text(
        json.dumps([row.__dict__ for row in rows], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(rows, summary_md)

    print(f"[done] rows={len(rows)}")
    print(f"[done] json={summary_json}")
    print(f"[done] md={summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
