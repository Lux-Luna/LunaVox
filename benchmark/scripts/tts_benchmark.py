#!/usr/bin/env python
"""
LunaVox TTS performance benchmark.

This script measures latency and resource consumption when synthesising
'This is LunaVox speaking English.' with a selected pretrained ONNX character.

Metrics:
    - Model disk size (original ONNX footprint)
    - Runtime memory footprint (process RSS delta after load)
    - Runtime GPU memory footprint (NVML, if available)
    - Per-iteration statistics:
        * First packet latency
        * End-to-end latency
        * CPU time
        * Process RSS delta
        * GPU memory delta (if available)
        * Audio duration
        * Real-time factor
        * Throughput (iterations per second)
    - Aggregate statistics over N iterations (defaults to 100)

Results are written to JSON (and optionally CSV) files inside benchmark/results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SAMPLE_RATE = 32000
REFERENCE_TEXT = "This is LunaVox speaking English."
REFERENCE_AUDIO_TEXT = "私は天使なんかじゃないわ。病院なんてないわよ。誰も病まないから。みんな死んでるから。"
REFERENCE_AUDIO_FILENAME = "私は天使なんかじゃないわ。病院なんてないわよ。誰も病まないから。みんな死んでるから。.wav"
CHARACTER_NAME = "pretrained"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
TUTORIAL_DIR = REPO_ROOT / "Tutorial"

MODEL_DIRECTORIES: Dict[str, Path] = {
    "v2": REPO_ROOT / "Data" / "character_model" / "v2" / "pretrained",
    "v2_pro_plus": REPO_ROOT / "Data" / "character_model" / "v2_pro_plus" / "pretrained",
}

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

import psutil  # type: ignore

try:
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover
    pynvml = None

import data_setup  # type: ignore
import lunavox_tts as lunavox  # type: ignore


def bytes_to_mb(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return value / (1024 * 1024)


def compute_directory_size(path: Path) -> int:
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


class GPUMonitor:
    def __init__(self, device_index: int = 0):
        self.available = False
        self.device_index = device_index
        self._handle = None
        self.device_name: Optional[str] = None
        self.error: Optional[str] = None
        if pynvml is None:
            self.error = "pynvml not installed"
            return
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.device_name = pynvml.nvmlDeviceGetName(self._handle).decode("utf-8")  # type: ignore[arg-type]
            self.available = True
        except Exception as exc:  # pragma: no cover
            self.error = str(exc)
            self.available = False

    def memory_info(self) -> Optional[Tuple[int, int, int]]:
        if not self.available or self._handle is None:
            return None
        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return info.total, info.used, info.free  # type: ignore[attr-defined]

    def shutdown(self) -> None:
        if self.available:
            try:  # pragma: no cover
                pynvml.nvmlShutdown()
            except Exception:
                pass


@dataclass
class IterationMetrics:
    iteration: int
    timestamp: float
    first_packet_latency_ms: float
    total_latency_ms: float
    cpu_time_s: float
    rss_before_mb: float
    rss_after_mb: float
    rss_delta_mb: float
    gpu_before_mb: Optional[float]
    gpu_after_mb: Optional[float]
    gpu_delta_mb: Optional[float]
    audio_duration_s: float
    real_time_factor: Optional[float]
    iterations_per_second: float


def format_duration_ms(value: float) -> float:
    return round(value * 1000, 3)


async def run_tts_once(language: str) -> Tuple[float, float, bytearray]:
    start_time = time.perf_counter()
    first_packet_latency: Optional[float] = None
    audio_buffer = bytearray()
    async for chunk in lunavox.tts_async(
        character_name=CHARACTER_NAME,
        text=REFERENCE_TEXT,
        play=False,
        split_sentence=False,
        language=language,
    ):
        now = time.perf_counter()
        if first_packet_latency is None:
            first_packet_latency = now - start_time
        audio_buffer.extend(chunk)
    total_latency = time.perf_counter() - start_time
    if first_packet_latency is None:
        first_packet_latency = total_latency
    return first_packet_latency, total_latency, audio_buffer


def measure_iteration(
    iteration_index: int,
    process: psutil.Process,
    gpu_monitor: GPUMonitor,
    language: str,
) -> IterationMetrics:
    timestamp = time.time()
    rss_before = process.memory_info().rss
    cpu_times_before = process.cpu_times()
    if gpu_monitor.available:
        gpu_mem_before_tuple = gpu_monitor.memory_info()
        gpu_used_before = gpu_mem_before_tuple[1] if gpu_mem_before_tuple else None
    else:
        gpu_used_before = None

    first_packet_latency, total_latency, audio_buffer = asyncio.run(run_tts_once(language))

    cpu_times_after = process.cpu_times()
    rss_after = process.memory_info().rss
    if gpu_monitor.available:
        gpu_mem_after_tuple = gpu_monitor.memory_info()
        gpu_used_after = gpu_mem_after_tuple[1] if gpu_mem_after_tuple else None
    else:
        gpu_used_after = None

    cpu_time = (cpu_times_after.user + cpu_times_after.system) - (cpu_times_before.user + cpu_times_before.system)

    audio_samples = len(audio_buffer) // 2  # int16 -> 2 bytes
    audio_duration = audio_samples / SAMPLE_RATE if audio_samples else 0.0
    real_time_factor = total_latency / audio_duration if audio_duration > 0 else None

    iteration_metrics = IterationMetrics(
        iteration=iteration_index,
        timestamp=timestamp,
        first_packet_latency_ms=format_duration_ms(first_packet_latency),
        total_latency_ms=format_duration_ms(total_latency),
        cpu_time_s=round(cpu_time, 6),
        rss_before_mb=bytes_to_mb(rss_before) or 0.0,
        rss_after_mb=bytes_to_mb(rss_after) or 0.0,
        rss_delta_mb=bytes_to_mb(rss_after - rss_before) or 0.0,
        gpu_before_mb=bytes_to_mb(gpu_used_before),
        gpu_after_mb=bytes_to_mb(gpu_used_after),
        gpu_delta_mb=bytes_to_mb((gpu_used_after - gpu_used_before) if (gpu_used_after is not None and gpu_used_before is not None) else None),
        audio_duration_s=round(audio_duration, 6),
        real_time_factor=round(real_time_factor, 6) if real_time_factor is not None else None,
        iterations_per_second=round(1.0 / total_latency if total_latency > 0 else 0.0, 6),
    )
    return iteration_metrics


def summarise_metrics(metrics: List[IterationMetrics]) -> Dict[str, Any]:
    def maybe_mean(values: Iterable[float]) -> Optional[float]:
        values = list(values)
        if not values:
            return None
        return statistics.mean(values)

    def maybe_stdev(values: Iterable[float]) -> Optional[float]:
        values = list(values)
        if len(values) < 2:
            return None
        return statistics.stdev(values)

    def collect(field: str) -> List[float]:
        return [getattr(m, field) for m in metrics if getattr(m, field) is not None]

    summary = {
        "iterations": len(metrics),
        "avg_first_packet_latency_ms": round(maybe_mean(collect("first_packet_latency_ms")) or 0.0, 3),
        "std_first_packet_latency_ms": round(maybe_stdev(collect("first_packet_latency_ms")) or 0.0, 3)
        if len(metrics) > 1
        else None,
        "avg_total_latency_ms": round(maybe_mean(collect("total_latency_ms")) or 0.0, 3),
        "std_total_latency_ms": round(maybe_stdev(collect("total_latency_ms")) or 0.0, 3)
        if len(metrics) > 1
        else None,
        "avg_cpu_time_s": round(maybe_mean(collect("cpu_time_s")) or 0.0, 6),
        "avg_rss_after_mb": round(maybe_mean(collect("rss_after_mb")) or 0.0, 3),
        "peak_rss_after_mb": round(max(collect("rss_after_mb")) if collect("rss_after_mb") else 0.0, 3),
        "avg_rss_delta_mb": round(maybe_mean(collect("rss_delta_mb")) or 0.0, 3),
        "avg_gpu_after_mb": round(maybe_mean(collect("gpu_after_mb")) or 0.0, 3) if collect("gpu_after_mb") else None,
        "peak_gpu_after_mb": round(max(collect("gpu_after_mb")), 3) if collect("gpu_after_mb") else None,
        "avg_gpu_delta_mb": round(maybe_mean(collect("gpu_delta_mb")) or 0.0, 3) if collect("gpu_delta_mb") else None,
        "avg_audio_duration_s": round(maybe_mean(collect("audio_duration_s")) or 0.0, 6),
        "avg_real_time_factor": round(maybe_mean(collect("real_time_factor")) or 0.0, 6) if collect("real_time_factor") else None,
        "avg_iterations_per_second": round(maybe_mean(collect("iterations_per_second")) or 0.0, 6),
    }
    return summary


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def write_csv(path: Path, metrics: List[IterationMetrics]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(metrics[0]).keys()) if metrics else []
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            writer.writerow(asdict(item))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LunaVox TTS benchmark runner")
    parser.add_argument("--iterations", type=int, default=100, help="Number of measured synthesis iterations.")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warm-up iterations to run before measurement.")
    parser.add_argument("--language", type=str, default="en", help="Language code passed to lunavox.tts_async.")
    parser.add_argument(
        "--model-version",
        type=str,
        choices=sorted(MODEL_DIRECTORIES.keys()),
        default="v2",
        help="Character model version to benchmark (determines ONNX directory).",
    )
    parser.add_argument("--device-index", type=int, default=0, help="GPU device index for NVML monitoring.")
    parser.add_argument("--output-json", type=Path, default=None, help="Path to store aggregate JSON results.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Path to store per-iteration CSV results.")
    parser.add_argument("--skip-csv", action="store_true", help="Do not emit CSV iteration details.")
    parser.add_argument("--verbose", action="store_true", help="Print extra diagnostic information during the run.")
    return parser.parse_args()


def setup_environment(model_version: str) -> Dict[str, Any]:
    os.environ.setdefault("HUBERT_MODEL_PATH", str(REPO_ROOT / "Data" / "chinese-hubert-base.onnx"))
    os.environ.setdefault("OPEN_JTALK_DICT_DIR", str(REPO_ROOT / "Data" / "open_jtalk_dic_utf_8-1.11"))

    data_setup.ensure_data_from_hf()

    if model_version not in MODEL_DIRECTORIES:
        raise ValueError(f"Unsupported model_version '{model_version}'. Available: {sorted(MODEL_DIRECTORIES)}")

    model_dir = MODEL_DIRECTORIES[model_version]

    load_start = time.perf_counter()
    lunavox.load_character(CHARACTER_NAME, str(model_dir))
    load_end = time.perf_counter()

    audio_path = REPO_ROOT / "Data" / "audio_resources" / "pretrained" / REFERENCE_AUDIO_FILENAME
    lunavox.set_reference_audio(
        CHARACTER_NAME,
        str(audio_path),
        REFERENCE_AUDIO_TEXT,
        audio_language="ja",
    )

    disk_size_bytes = compute_directory_size(model_dir)
    return {
        "model_dir": str(model_dir),
        "model_disk_size_bytes": disk_size_bytes,
        "load_time_seconds": load_end - load_start,
        "reference_audio": str(audio_path),
    }


def main() -> None:
    args = parse_args()
    process = psutil.Process(os.getpid())
    gpu_monitor = GPUMonitor(device_index=args.device_index)

    if args.output_json is None:
        args.output_json = REPO_ROOT / "benchmark" / "results" / f"{args.model_version}_tts_benchmark_results.json"
    if args.output_csv is None:
        args.output_csv = REPO_ROOT / "benchmark" / "results" / f"{args.model_version}_tts_benchmark_iterations.csv"

    baseline_rss = process.memory_info().rss
    baseline_gpu_used: Optional[int] = None
    if gpu_monitor.available:
        gpu_info = gpu_monitor.memory_info()
        baseline_gpu_used = gpu_info[1] if gpu_info else None

    setup_info = setup_environment(args.model_version)

    rss_after_load = process.memory_info().rss
    gpu_after_load: Optional[int] = None
    if gpu_monitor.available:
        gpu_info = gpu_monitor.memory_info()
        gpu_after_load = gpu_info[1] if gpu_info else None

    runtime_rss_delta_mb = bytes_to_mb(rss_after_load - baseline_rss) or 0.0
    runtime_gpu_delta_mb = (
        bytes_to_mb((gpu_after_load - baseline_gpu_used) if (gpu_after_load is not None and baseline_gpu_used is not None) else None)
        if gpu_monitor.available
        else None
    )

    if args.verbose:
        print(f"Model loaded from: {setup_info['model_dir']}")
        print(f"Model disk size: {bytes_to_mb(setup_info['model_disk_size_bytes']):.2f} MB")
        print(f"Load time: {setup_info['load_time_seconds']:.3f} s")
        print(f"Runtime RSS delta after load: {runtime_rss_delta_mb:.2f} MB")
        if runtime_gpu_delta_mb is not None:
            print(f"Runtime GPU memory delta after load: {runtime_gpu_delta_mb:.2f} MB")
        if gpu_monitor.error:
            print(f"GPU monitoring disabled: {gpu_monitor.error}")

    warmup_iterations = max(0, args.warmup)
    for i in range(warmup_iterations):
        asyncio.run(run_tts_once(args.language))
        if args.verbose:
            print(f"Warm-up iteration {i + 1}/{warmup_iterations} completed.")

    metrics: List[IterationMetrics] = []
    failures: List[str] = []

    for iteration in range(1, args.iterations + 1):
        try:
            metrics.append(measure_iteration(iteration, process, gpu_monitor, args.language))
            if args.verbose:
                print(
                    f"[{iteration}/{args.iterations}] "
                    f"first_packet={metrics[-1].first_packet_latency_ms:.2f} ms, "
                    f"total={metrics[-1].total_latency_ms:.2f} ms, "
                    f"rss_after={metrics[-1].rss_after_mb:.2f} MB"
                )
        except Exception as exc:  # pragma: no cover
            message = f"Iteration {iteration} failed: {exc}"
            failures.append(message)
            if args.verbose:
                print(message)

    summary = summarise_metrics(metrics)

    payload: Dict[str, Any] = {
        "timestamp": time.time(),
        "iterations_requested": args.iterations,
        "warmup_iterations": warmup_iterations,
        "language": args.language,
        "model_version": args.model_version,
        "first_packet_latency_ms_avg": summary["avg_first_packet_latency_ms"],
        "total_latency_ms_avg": summary["avg_total_latency_ms"],
        "cpu_time_s_avg": summary["avg_cpu_time_s"],
        "real_time_factor_avg": summary.get("avg_real_time_factor"),
        "iterations_per_second_avg": summary["avg_iterations_per_second"],
        "model_disk_size_mb": round(bytes_to_mb(setup_info["model_disk_size_bytes"]) or 0.0, 3),
        "runtime_rss_delta_after_load_mb": round(runtime_rss_delta_mb, 3),
        "runtime_gpu_delta_after_load_mb": round(runtime_gpu_delta_mb, 3) if runtime_gpu_delta_mb is not None else None,
        "gpu_name": gpu_monitor.device_name if gpu_monitor.available else None,
        "gpu_monitor_error": gpu_monitor.error,
        "summary": summary,
        "iterations": [asdict(item) for item in metrics],
        "failures": failures,
    }

    write_json(args.output_json, payload)
    if metrics and not args.skip_csv:
        write_csv(args.output_csv, metrics)

    if args.verbose:
        print(f"Results saved to: {args.output_json}")
        if metrics and not args.skip_csv:
            print(f"Iteration CSV saved to: {args.output_csv}")

    gpu_monitor.shutdown()


if __name__ == "__main__":
    main()

