#!/usr/bin/env python
"""
Summarise LunaVox benchmark result JSON files and optionally emit a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def percentile(values: List[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    k = (len(values) - 1) * fraction
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] * (c - k) + values[c] * (k - f)


def round_or_none(value: Optional[float], digits: int = 3) -> Optional[float]:
    if value is None:
        return None
    return round(value, digits)


def load_metrics(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    iterations = data.get("iterations", [])

    totals = [item["total_latency_ms"] for item in iterations]
    first_packets = [item["first_packet_latency_ms"] for item in iterations]
    cpu_times = [item["cpu_time_s"] for item in iterations]
    rss_after = [item["rss_after_mb"] for item in iterations]
    audio_duration = [item["audio_duration_s"] for item in iterations]
    throughput = [item["iterations_per_second"] for item in iterations]

    summary = data.get("summary", {})

    result = {
        "model_version": data.get("model_version"),
        "iterations": len(iterations),
        "model_disk_size_mb": data.get("model_disk_size_mb"),
        "runtime_rss_delta_after_load_mb": data.get("runtime_rss_delta_after_load_mb"),
        "peak_rss_after_mb": summary.get("peak_rss_after_mb"),
        "avg_first_packet_latency_ms": summary.get("avg_first_packet_latency_ms"),
        "avg_total_latency_ms": summary.get("avg_total_latency_ms"),
        "avg_cpu_time_s": summary.get("avg_cpu_time_s"),
        "avg_real_time_factor": summary.get("avg_real_time_factor"),
        "avg_iterations_per_second": summary.get("avg_iterations_per_second"),
        "avg_audio_duration_s": summary.get("avg_audio_duration_s"),
        "first_packet_latency_ms_p95": round_or_none(percentile(first_packets, 0.95)),
        "total_latency_ms_p95": round_or_none(percentile(totals, 0.95)),
        "cpu_time_s_p95": round_or_none(percentile(cpu_times, 0.95), digits=6),
        "rss_after_mb_avg": summary.get("avg_rss_after_mb"),
        "rss_after_mb_p95": round_or_none(percentile(rss_after, 0.95)),
        "iterations_per_second_p95": round_or_none(percentile(throughput, 0.95), digits=6),
        "audio_duration_s_p95": round_or_none(percentile(audio_duration, 0.95), digits=6),
        "failures": len(data.get("failures", [])),
        "input_path": str(path),
    }
    return result


def emit_markdown_table(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "模型版本",
        "模型大小 (MB)",
        "加载后RSS增量 (MB)",
        "平均首包延迟 (ms)",
        "P95首包延迟 (ms)",
        "平均全句延迟 (ms)",
        "P95全句延迟 (ms)",
        "平均CPU时间 (s)",
        "平均实时因子",
        "平均吞吐 (iter/s)",
    ]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]

    for row in rows:
        lines.append(
            "|"
            + "|".join(
                [
                    str(row["model_version"]),
                    f"{row['model_disk_size_mb']:.2f}" if row["model_disk_size_mb"] is not None else "N/A",
                    f"{row['runtime_rss_delta_after_load_mb']:.2f}" if row["runtime_rss_delta_after_load_mb"] is not None else "N/A",
                    f"{row['avg_first_packet_latency_ms']:.2f}" if row["avg_first_packet_latency_ms"] is not None else "N/A",
                    f"{row['first_packet_latency_ms_p95']:.2f}" if row["first_packet_latency_ms_p95"] is not None else "N/A",
                    f"{row['avg_total_latency_ms']:.2f}" if row["avg_total_latency_ms"] is not None else "N/A",
                    f"{row['total_latency_ms_p95']:.2f}" if row["total_latency_ms_p95"] is not None else "N/A",
                    f"{row['avg_cpu_time_s']:.2f}" if row["avg_cpu_time_s"] is not None else "N/A",
                    f"{row['avg_real_time_factor']:.3f}" if row["avg_real_time_factor"] is not None else "N/A",
                    f"{row['avg_iterations_per_second']:.3f}" if row["avg_iterations_per_second"] is not None else "N/A",
                ]
            )
            + "|"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise benchmark result JSON files.")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[
            Path("benchmark/results/v2_results.json"),
            Path("benchmark/results/v2_pro_plus_results.json"),
        ],
        help="Paths to benchmark result JSON files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmark/results/summary.json"),
        help="Path to write aggregated summary JSON.",
    )
    parser.add_argument(
        "--print-markdown",
        action="store_true",
        help="Print a Markdown table summarising key metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries: List[Dict[str, Any]] = []
    for path in args.inputs:
        if not path.exists():
            raise FileNotFoundError(f"Benchmark result not found: {path}")
        summaries.append(load_metrics(path))

    combined = {"models": summaries}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.print_markdown:
        print(emit_markdown_table(summaries))
    else:
        print(json.dumps(combined, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

