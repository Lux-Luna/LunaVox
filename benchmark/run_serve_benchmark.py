"""Concurrent-request benchmark for ``lunavox serve``.

Spins up a local uvicorn server bound to ``127.0.0.1``, then fires
``N`` concurrent ``POST /v1/synth`` requests in parallel, repeats a
configurable number of rounds, and reports:

* Per-request latency percentiles (p50 / p95 / p99)
* Aggregate throughput (requests/second)
* Speedup vs. sequential baseline
* Average server-reported RTF

Phase 5B target: ``concurrency=4`` throughput ≥ ``2.5×`` the
``concurrency=1`` baseline (with ``--batch-size 4``). The ratio is
bounded by the context-pool design — perfect scaling would be 4×,
but the shared CPU bookkeeping and ONNX decoder serialisation
prevent that. 2.5× is the empirical floor observed for similar
pool designs in llama.cpp server.

Usage::

    python benchmark/run_serve_benchmark.py \
        --model base_small \
        --batch-size 4 \
        --concurrency 1 2 4 \
        --rounds 5

The script assumes the model is already pulled (``lunavox model
pull --model base_small``) and the C++ engine is built (``lunavox
build``). It does NOT start / stop the server — spin up
``lunavox serve --batch-size 4 --port 8765`` in another terminal
first, then run this against ``http://127.0.0.1:8765``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import httpx
except ImportError:
    print("httpx is required. Install it via `pip install httpx`.", file=sys.stderr)
    raise SystemExit(1) from None


DEFAULT_TEXT = (
    "This is a LunaVox serve benchmark sentence, roughly twenty five words long "
    "so the synthesis path exercises a representative talker and predictor workload."
)


@dataclass
class RoundResult:
    """One round of ``concurrency`` parallel requests."""

    concurrency: int
    latencies_ms: list[float] = field(default_factory=list)
    wall_ms: float = 0.0
    rtfs: list[float] = field(default_factory=list)


async def _single_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
) -> tuple[float, float]:
    """Fire one synthesis request and return (latency_ms, server_rtf)."""
    t0 = time.perf_counter()
    resp = await client.post(url, json=payload, timeout=120.0)
    resp.raise_for_status()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    stats_header = resp.headers.get("x-lunavox-stats", "{}")
    server_rtf = 0.0
    try:
        parsed = json.loads(stats_header)
        server_rtf = float(parsed.get("stats", {}).get("rtf", 0.0))
    except (ValueError, KeyError, TypeError):
        pass
    return latency_ms, server_rtf


async def _run_round(
    base_url: str,
    text: str,
    concurrency: int,
) -> RoundResult:
    payload = {"text": text, "voice": "base"}
    url = f"{base_url.rstrip('/')}/v1/synth"
    result = RoundResult(concurrency=concurrency)

    async with httpx.AsyncClient() as client:
        wall_start = time.perf_counter()
        tasks = [_single_request(client, url, payload) for _ in range(concurrency)]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        result.wall_ms = (time.perf_counter() - wall_start) * 1000.0

    for outcome in outcomes:
        if isinstance(outcome, Exception):
            print(f"  request failed: {outcome}", file=sys.stderr)
            continue
        latency, rtf = outcome
        result.latencies_ms.append(latency)
        result.rtfs.append(rtf)

    return result


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = max(0, min(len(values_sorted) - 1, int(round(pct / 100.0 * (len(values_sorted) - 1)))))
    return values_sorted[idx]


def _report(rounds: list[RoundResult]) -> None:
    by_c: dict[int, list[RoundResult]] = {}
    for r in rounds:
        by_c.setdefault(r.concurrency, []).append(r)

    print()
    print("  concurrency | p50 latency | p95 latency | p99 latency | throughput (req/s) | avg RTF")
    print("  ----------- + ----------- + ----------- + ----------- + ------------------ + -------")

    baseline_tp: float = 0.0
    for concurrency in sorted(by_c.keys()):
        lats: list[float] = []
        rtfs: list[float] = []
        walls: list[float] = []
        req_counts: list[int] = []
        for r in by_c[concurrency]:
            lats.extend(r.latencies_ms)
            rtfs.extend(r.rtfs)
            walls.append(r.wall_ms)
            req_counts.append(len(r.latencies_ms))

        if not lats:
            continue

        throughput = sum(req_counts) / (sum(walls) / 1000.0) if sum(walls) > 0 else 0.0
        p50 = _percentile(lats, 50)
        p95 = _percentile(lats, 95)
        p99 = _percentile(lats, 99)
        avg_rtf = statistics.mean(rtfs) if rtfs else 0.0

        if concurrency == 1:
            baseline_tp = throughput

        print(
            f"  {concurrency:^11d} | "
            f"{p50:>9.1f}ms | "
            f"{p95:>9.1f}ms | "
            f"{p99:>9.1f}ms | "
            f"{throughput:>16.2f} | "
            f"{avg_rtf:>5.3f}"
        )

    if baseline_tp > 0:
        print()
        for concurrency in sorted(by_c.keys()):
            if concurrency == 1:
                continue
            merged_wall = sum(r.wall_ms for r in by_c[concurrency])
            merged_reqs = sum(len(r.latencies_ms) for r in by_c[concurrency])
            if merged_wall <= 0:
                continue
            tp = merged_reqs / (merged_wall / 1000.0)
            print(f"  speedup @ concurrency={concurrency}: {tp / baseline_tp:.2f}×")


async def _main_async(args: argparse.Namespace) -> None:
    # Warm-up round at concurrency 1 so JIT / page-in costs don't
    # contaminate the first measurement.
    print("Warming up…")
    await _run_round(args.base_url, args.text, concurrency=1)

    rounds: list[RoundResult] = []
    for concurrency in args.concurrency:
        for round_idx in range(args.rounds):
            print(
                f"Round {round_idx + 1}/{args.rounds} @ concurrency={concurrency}…",
                flush=True,
            )
            result = await _run_round(args.base_url, args.text, concurrency)
            rounds.append(result)

    _report(rounds)


def main() -> int:
    parser = argparse.ArgumentParser(description="LunaVox serve concurrent benchmark")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL for the running `lunavox serve` instance",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Concurrency levels to test (default: 1 2 4)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Measurement rounds per concurrency level (default: 3)",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Text to synthesize (default: 25-word English sentence)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
