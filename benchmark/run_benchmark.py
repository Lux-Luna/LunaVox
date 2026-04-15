"""LunaVox 0.6B 100-run benchmark driver.

Usage:
    # Per-backend runs (do one per conda env + rebuild)
    python benchmark/run_benchmark.py --backend vulkan
    python benchmark/run_benchmark.py --backend cuda
    python benchmark/run_benchmark.py --backend cpu

    # After all three are done, aggregate into report.md
    python benchmark/run_benchmark.py --aggregate

Layout:
    benchmark/results/stats_<backend>.json    - raw --stats-json output
    benchmark/results/summary_<backend>.json  - aggregated metrics
    benchmark/results/meta_<backend>.json     - git/env/host/timestamp
    benchmark/report.md                       - final 3-backend report
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmark" / "results"
REPORT_PATH = REPO_ROOT / "benchmark" / "report.md"

# Fixed benchmark config — do NOT override from CLI. Keeping the parameters
# as constants makes every run directly comparable across backends.
MODEL_DIR = REPO_ROOT / "models" / "base_small"
REF_PATH = REPO_ROOT / "ref" / "ref_0.6B.json"
CLI_BINARY = REPO_ROOT / "build" / ("lunavox-cli.exe" if os.name == "nt" else "lunavox-cli")
BENCH_TEXT = (
    "LunaVox is a lightweight on-device text-to-speech engine built on "
    "Qwen3, optimized for low-latency streaming synthesis on consumer GPUs."
)
WARMUP_RUNS = 5
REPEAT_RUNS = 100

BACKENDS = ("vulkan", "cuda", "cpu")


# ---------------------------------------------------------------------------
# VRAM sampler
# ---------------------------------------------------------------------------


class VramSampler:
    """Polls NVML `used` memory every 100 ms while a subprocess is running.

    Records (timestamp, used_bytes) samples. After join, exposes `peak_bytes`
    (max used - baseline), `steady_mb` (mean of the last 90% of samples minus
    baseline), and `baseline_bytes` (initial reading before subprocess start).
    """

    def __init__(self, interval_s: float = 0.1):
        self.interval_s = interval_s
        self.samples: list[tuple[float, int]] = []
        self.baseline_bytes: int = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._handle = None
        self._pynvml = None
        self.available = False
        self.skip_reason: str | None = None

        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count == 0:
                self.skip_reason = "no NVIDIA device"
                return
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self.baseline_bytes = int(info.used)
            self.available = True
        except Exception as e:  # noqa: BLE001 — pynvml missing or init failure
            self.skip_reason = f"pynvml unavailable: {type(e).__name__}: {e}"

    def start(self) -> None:
        if not self.available:
            return
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.available:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        try:
            self._pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass

    def _poll(self) -> None:
        assert self._pynvml is not None and self._handle is not None
        while not self._stop.is_set():
            try:
                info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self.samples.append((time.monotonic(), int(info.used)))
            except Exception:  # noqa: BLE001
                break
            self._stop.wait(self.interval_s)

    def metrics(self) -> dict:
        if not self.available:
            return {
                "vram_peak_mb": None,
                "vram_steady_mb": None,
                "vram_baseline_mb": None,
                "vram_n_samples": 0,
                "vram_skipped_reason": self.skip_reason,
            }
        if not self.samples:
            return {
                "vram_peak_mb": 0.0,
                "vram_steady_mb": 0.0,
                "vram_baseline_mb": round(self.baseline_bytes / 1024 / 1024, 2),
                "vram_n_samples": 0,
                "vram_skipped_reason": "no samples captured",
            }
        used_values = [s[1] for s in self.samples]
        peak = max(used_values)
        # Last 90% window for steady-state (skip initial ramp)
        tail_start = max(1, int(len(used_values) * 0.10))
        tail = used_values[tail_start:]
        steady = sum(tail) / len(tail) if tail else peak
        peak_delta = max(0, peak - self.baseline_bytes)
        steady_delta = max(0.0, steady - self.baseline_bytes)
        return {
            "vram_peak_mb": round(peak_delta / 1024 / 1024, 2),
            "vram_steady_mb": round(steady_delta / 1024 / 1024, 2),
            "vram_baseline_mb": round(self.baseline_bytes / 1024 / 1024, 2),
            "vram_peak_absolute_mb": round(peak / 1024 / 1024, 2),
            "vram_n_samples": len(self.samples),
            "vram_skipped_reason": None,
        }


# ---------------------------------------------------------------------------
# Subprocess invocation
# ---------------------------------------------------------------------------


def run_cli(stats_path: Path) -> tuple[int, float]:
    if not CLI_BINARY.exists():
        sys.exit(f"ERROR: CLI binary not found at {CLI_BINARY}. Run `lunavox build` first.")
    if not MODEL_DIR.exists():
        sys.exit(f"ERROR: model dir not found at {MODEL_DIR}.")
    if not REF_PATH.exists():
        sys.exit(f"ERROR: ref file not found at {REF_PATH}.")

    cmd = [
        str(CLI_BINARY),
        "-m",
        str(MODEL_DIR),
        "-r",
        str(REF_PATH),
        "-t",
        BENCH_TEXT,
        "--warmup",
        str(WARMUP_RUNS),
        "--repeat",
        str(REPEAT_RUNS),
        "--stats-json",
        str(stats_path),
    ]
    print(f"[bench] launching: {' '.join(cmd)}", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    wall = time.monotonic() - t0
    return proc.returncode, wall


# ---------------------------------------------------------------------------
# Stats aggregation
# ---------------------------------------------------------------------------


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _stats_block(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        "stddev": round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0,
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "p50": round(_percentile(values, 0.50), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "p99": round(_percentile(values, 0.99), 3),
    }


def summarize(stats_path: Path, vram: dict, wall_elapsed_s: float) -> dict:
    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = data.get("runs", [])
    if len(runs) < REPEAT_RUNS:
        print(
            f"[bench] WARNING: expected {REPEAT_RUNS} runs in stats, got {len(runs)}",
            flush=True,
        )

    totals = [float(r["timing_ms"]["total"]) for r in runs]
    tokenizes = [float(r["timing_ms"].get("tokenize", 0)) for r in runs]
    encodes = [float(r["timing_ms"].get("encode", 0)) for r in runs]
    generates = [float(r["timing_ms"].get("generate", 0)) for r in runs]
    decodes = [float(r["timing_ms"].get("decode", 0)) for r in runs]
    ttfbs = [float(r.get("stream", {}).get("t_first_audio_ms", r["timing_ms"].get("first_audio", 0))) for r in runs]
    rtfs = [float(r["rtf"]) for r in runs]
    audio_durs = [float(r.get("audio_sec", r.get("audio_duration_s", 0))) for r in runs]
    rss_peaks = [int(r["mem"].get("rss_peak", 0)) for r in runs]
    phys_peaks = [int(r["mem"].get("phys_peak", 0)) for r in runs]

    rss_peak_max_mb = max(rss_peaks) / 1024 / 1024 if rss_peaks else 0.0
    phys_peak_max_mb = max(phys_peaks) / 1024 / 1024 if phys_peaks else 0.0

    chars_per_sec = [len(BENCH_TEXT) / (t / 1000.0) for t in totals if t > 0]

    first_chunk_frames = runs[0].get("stream", {}).get("first_chunk_frames") if runs else None
    sample_rate = runs[0].get("sample_rate") if runs else None

    return {
        "config": {
            "text": BENCH_TEXT,
            "text_length_chars": len(BENCH_TEXT),
            "warmup_runs": WARMUP_RUNS,
            "repeat_runs": REPEAT_RUNS,
            "model_dir": str(MODEL_DIR.relative_to(REPO_ROOT)).replace("\\", "/"),
            "ref": str(REF_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "first_chunk_frames": first_chunk_frames,
            "sample_rate": sample_rate,
        },
        "load": {
            "t_load_ms": int(data.get("t_load_ms", 0)),
            "t_warmup_ms": int(data.get("t_warmup_ms", 0)),
        },
        "wall_elapsed_s": round(wall_elapsed_s, 2),
        "total_ms": _stats_block(totals),
        "ttfb_ms": _stats_block(ttfbs),
        "rtf": _stats_block(rtfs),
        "realtime_multiplier": round(1.0 / statistics.fmean(rtfs), 3) if rtfs else None,
        "stage_mean_ms": {
            "tokenize": round(statistics.fmean(tokenizes), 3) if tokenizes else 0.0,
            "encode": round(statistics.fmean(encodes), 3) if encodes else 0.0,
            "generate": round(statistics.fmean(generates), 3) if generates else 0.0,
            "decode": round(statistics.fmean(decodes), 3) if decodes else 0.0,
        },
        "audio_duration_s": _stats_block(audio_durs),
        "throughput_chars_per_sec": _stats_block(chars_per_sec),
        "memory": {
            "rss_peak_max_mb": round(rss_peak_max_mb, 2),
            "rss_peak_mean_mb": round(statistics.fmean(rss_peaks) / 1024 / 1024, 2) if rss_peaks else 0.0,
            "phys_peak_max_mb": round(phys_peak_max_mb, 2),
            **vram,
        },
    }


# ---------------------------------------------------------------------------
# Meta capture
# ---------------------------------------------------------------------------


def capture_meta(backend: str) -> dict:
    def _run(cmd: list[str]) -> str:
        try:
            out = subprocess.check_output(cmd, cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL)
            return out.decode("utf-8", errors="replace").strip()
        except Exception:  # noqa: BLE001
            return ""

    uname = platform.uname()
    return {
        "backend": backend,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python_version": sys.version.split()[0],
        "platform": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor,
            "node": uname.node,
        },
        "cli_binary": str(CLI_BINARY.relative_to(REPO_ROOT)).replace("\\", "/"),
        "cli_binary_mtime": datetime.fromtimestamp(CLI_BINARY.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds") if CLI_BINARY.exists() else None,
    }


# ---------------------------------------------------------------------------
# Per-backend run
# ---------------------------------------------------------------------------


def run_backend(backend: str) -> None:
    if backend not in BACKENDS:
        sys.exit(f"ERROR: backend must be one of {BACKENDS}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stats_path = RESULTS_DIR / f"stats_{backend}.json"
    summary_path = RESULTS_DIR / f"summary_{backend}.json"
    meta_path = RESULTS_DIR / f"meta_{backend}.json"

    meta = capture_meta(backend)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[bench] wrote {meta_path.relative_to(REPO_ROOT)}", flush=True)

    sampler = VramSampler()
    if sampler.available:
        print(f"[bench] VRAM sampler active (baseline={sampler.baseline_bytes/1024/1024:.1f} MB)", flush=True)
    else:
        print(f"[bench] VRAM sampler SKIPPED ({sampler.skip_reason})", flush=True)

    sampler.start()
    rc, wall = run_cli(stats_path)
    sampler.stop()

    if rc != 0:
        sys.exit(f"ERROR: lunavox-cli exited with code {rc}")

    vram_metrics = sampler.metrics()
    summary = summarize(stats_path, vram_metrics, wall)
    summary["meta"] = meta

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[bench] wrote {summary_path.relative_to(REPO_ROOT)}", flush=True)

    _print_single_summary(backend, summary)


def _print_single_summary(backend: str, s: dict) -> None:
    tot = s["total_ms"]
    ttfb = s["ttfb_ms"]
    rtf = s["rtf"]
    mem = s["memory"]
    print("", flush=True)
    print(f"=== {backend.upper()} — {s['config']['repeat_runs']} runs ===", flush=True)
    print(f"  total_ms   mean={tot['mean']:.1f}  p50={tot['p50']:.1f}  p95={tot['p95']:.1f}  p99={tot['p99']:.1f}", flush=True)
    print(f"  ttfb_ms    mean={ttfb['mean']:.1f}  p50={ttfb['p50']:.1f}  p95={ttfb['p95']:.1f}  p99={ttfb['p99']:.1f}", flush=True)
    print(f"  rtf        mean={rtf['mean']:.4f}  (×{s['realtime_multiplier']} realtime)", flush=True)
    print(f"  rss_peak   max={mem['rss_peak_max_mb']:.1f} MB", flush=True)
    vram_str = f"{mem['vram_peak_mb']:.1f} MB" if mem.get("vram_peak_mb") is not None else f"n/a ({mem.get('vram_skipped_reason')})"
    print(f"  vram_peak  {vram_str}", flush=True)
    print(f"  load_ms={s['load']['t_load_ms']}  warmup_ms={s['load']['t_warmup_ms']}", flush=True)


# ---------------------------------------------------------------------------
# Aggregate + report rendering
# ---------------------------------------------------------------------------


def aggregate() -> None:
    summaries: dict[str, dict] = {}
    missing: list[str] = []
    for b in BACKENDS:
        p = RESULTS_DIR / f"summary_{b}.json"
        if not p.exists():
            missing.append(b)
            continue
        with open(p, "r", encoding="utf-8") as f:
            summaries[b] = json.load(f)

    if missing:
        print(
            f"[aggregate] WARNING: missing summaries for {missing} — report will include only: {list(summaries)}",
            flush=True,
        )
    if not summaries:
        sys.exit("ERROR: no summaries found. Run --backend <name> first.")

    # Sanity check: warn if git hashes differ across backends
    hashes = {b: s.get("meta", {}).get("git_commit", "") for b, s in summaries.items()}
    unique_hashes = {h for h in hashes.values() if h}
    if len(unique_hashes) > 1:
        print(f"[aggregate] WARNING: git commit differs across backends: {hashes}", flush=True)

    report = _render_report(summaries)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[aggregate] wrote {REPORT_PATH.relative_to(REPO_ROOT)}", flush=True)


def _render_report(summaries: dict[str, dict]) -> str:
    # Meta is the same shape across backends; take the first one for header
    any_backend = next(iter(summaries.values()))
    meta = any_backend.get("meta", {})
    cfg = any_backend["config"]

    host = meta.get("platform", {})
    out: list[str] = []
    out.append("# LunaVox 0.6B — 100-Run Benchmark Report")
    out.append("")
    out.append(f"Generated: {meta.get('timestamp_utc', '')}  \n"
               f"Git commit: `{meta.get('git_commit', '')[:12]}`  \n"
               f"Host: {host.get('system', '')} {host.get('release', '')} ({host.get('machine', '')})  \n"
               f"CPU: {host.get('processor', '')}  \n"
               f"Python: {meta.get('python_version', '')}")
    out.append("")
    out.append("## Configuration")
    out.append("")
    out.append(f"- **Model**: `{cfg['model_dir']}` (Qwen3-TTS-12Hz-0.6B-Base)")
    out.append(f"- **Voice reference**: `{cfg['ref']}` (pre-encoded codes)")
    out.append(f"- **Sample rate**: {cfg.get('sample_rate', 'n/a')} Hz")
    out.append(f"- **Warm-up**: {cfg['warmup_runs']} runs (not included in stats)")
    out.append(f"- **Repeat**: {cfg['repeat_runs']} runs per backend")
    out.append(f"- **First chunk frames**: {cfg.get('first_chunk_frames', 'n/a')}")
    out.append("- **Text** (25 words, fixed across backends):")
    out.append("")
    out.append(f"  > {cfg['text']}")
    out.append("")
    out.append("## 1. Overview")
    out.append("")
    out.append("| Backend | Total mean (ms) | TTFB mean (ms) | RTF mean | ×Realtime | Peak RSS (MB) | Peak VRAM (MB) |")
    out.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for b in BACKENDS:
        if b not in summaries:
            continue
        s = summaries[b]
        m = s["memory"]
        vram = f"{m['vram_peak_mb']:.1f}" if m.get("vram_peak_mb") is not None else "n/a"
        out.append(
            f"| **{_backend_label(b)}** | "
            f"{s['total_ms']['mean']:.1f} | "
            f"{s['ttfb_ms']['mean']:.1f} | "
            f"{s['rtf']['mean']:.4f} | "
            f"{s['realtime_multiplier']:.2f}× | "
            f"{m['rss_peak_max_mb']:.1f} | "
            f"{vram} |"
        )
    out.append("")

    out.append("## 2. Latency Distribution (total wall time per synth, ms)")
    out.append("")
    out.append("| Backend | mean | p50 | p95 | p99 | min | max | stddev |")
    out.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for b in BACKENDS:
        if b not in summaries:
            continue
        t = summaries[b]["total_ms"]
        out.append(
            f"| **{_backend_label(b)}** | {t['mean']:.1f} | {t['p50']:.1f} | {t['p95']:.1f} | "
            f"{t['p99']:.1f} | {t['min']:.1f} | {t['max']:.1f} | {t['stddev']:.1f} |"
        )
    out.append("")

    out.append("## 3. Time-to-First-Byte (first audio chunk, ms)")
    out.append("")
    out.append("TTFB = wall-clock delay from synth start to the first PCM sample becoming available via the streaming decoder.")
    out.append("")
    out.append("| Backend | mean | p50 | p95 | p99 | min | max | stddev |")
    out.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for b in BACKENDS:
        if b not in summaries:
            continue
        t = summaries[b]["ttfb_ms"]
        out.append(
            f"| **{_backend_label(b)}** | {t['mean']:.1f} | {t['p50']:.1f} | {t['p95']:.1f} | "
            f"{t['p99']:.1f} | {t['min']:.1f} | {t['max']:.1f} | {t['stddev']:.1f} |"
        )
    out.append("")

    out.append("## 4. Real-Time Factor (RTF = synth_wall / audio_duration)")
    out.append("")
    out.append("| Backend | mean | p50 | p95 | ×Realtime |")
    out.append("| :--- | ---: | ---: | ---: | ---: |")
    for b in BACKENDS:
        if b not in summaries:
            continue
        r = summaries[b]["rtf"]
        out.append(
            f"| **{_backend_label(b)}** | {r['mean']:.4f} | {r['p50']:.4f} | {r['p95']:.4f} | "
            f"{summaries[b]['realtime_multiplier']:.2f}× |"
        )
    out.append("")

    out.append("## 5. Stage Breakdown (per-run mean, ms)")
    out.append("")
    out.append("| Backend | Tokenize | Encode | Generate (LLM) | Decode (codec) |")
    out.append("| :--- | ---: | ---: | ---: | ---: |")
    for b in BACKENDS:
        if b not in summaries:
            continue
        st = summaries[b]["stage_mean_ms"]
        out.append(
            f"| **{_backend_label(b)}** | {st['tokenize']:.1f} | {st['encode']:.1f} | "
            f"{st['generate']:.1f} | {st['decode']:.1f} |"
        )
    out.append("")

    out.append("## 6. Memory Footprint")
    out.append("")
    out.append("| Backend | Peak RSS max (MB) | Peak RSS mean (MB) | Peak physical (MB) | Peak VRAM (MB) | Steady VRAM (MB) |")
    out.append("| :--- | ---: | ---: | ---: | ---: | ---: |")
    for b in BACKENDS:
        if b not in summaries:
            continue
        m = summaries[b]["memory"]
        vram_peak = f"{m['vram_peak_mb']:.1f}" if m.get("vram_peak_mb") is not None else "n/a"
        vram_steady = f"{m['vram_steady_mb']:.1f}" if m.get("vram_steady_mb") is not None else "n/a"
        out.append(
            f"| **{_backend_label(b)}** | {m['rss_peak_max_mb']:.1f} | {m['rss_peak_mean_mb']:.1f} | "
            f"{m['phys_peak_max_mb']:.1f} | {vram_peak} | {vram_steady} |"
        )
    out.append("")

    out.append("## 7. Throughput")
    out.append("")
    out.append("| Backend | Audio duration (s) | Chars/sec mean | Chars/sec p95 | Load time (ms) | In-load warmup (ms) |")
    out.append("| :--- | ---: | ---: | ---: | ---: | ---: |")
    for b in BACKENDS:
        if b not in summaries:
            continue
        s = summaries[b]
        cps = s["throughput_chars_per_sec"]
        out.append(
            f"| **{_backend_label(b)}** | {s['audio_duration_s']['mean']:.2f} | "
            f"{cps['mean']:.1f} | {cps['p95']:.1f} | "
            f"{s['load']['t_load_ms']} | {s['load']['t_warmup_ms']} |"
        )
    out.append("")

    out.append("## 8. Per-Backend Metadata")
    out.append("")
    for b in BACKENDS:
        if b not in summaries:
            continue
        mm = summaries[b].get("meta", {})
        out.append(f"### {_backend_label(b)}")
        out.append(f"- Git commit: `{mm.get('git_commit', '')[:12]}`")
        out.append(f"- Conda env: `{mm.get('conda_env', '')}`")
        out.append(f"- CLI binary mtime: {mm.get('cli_binary_mtime', '')}")
        out.append(f"- Wall elapsed: {summaries[b].get('wall_elapsed_s', 0)} s")
        out.append("")

    out.append("## 9. Notes")
    out.append("")
    out.append("- **RTF** = synth wall time / generated audio duration. Lower is better; <1.0 means faster than realtime.")
    out.append("- **TTFB** reflects the streaming pipeline (threaded decoder overlapped with talker+predictor). A batch caller that waits for `result.audio` still observes `total_ms`; a streaming caller starts consuming PCM at `t_first_audio_ms`.")
    out.append("- **VRAM** is sampled externally via pynvml at 100 ms intervals; values are delta above the pre-run baseline, so other GPU workloads on the same device do not skew the numbers.")
    out.append("- Raw per-run records are in `benchmark/results/stats_<backend>.json`; aggregated metrics are in `benchmark/results/summary_<backend>.json`.")
    out.append("")
    return "\n".join(out)


def _backend_label(b: str) -> str:
    return {
        "vulkan": "Universal GPU (Vulkan + DirectML)",
        "cuda": "CUDA 13",
        "cpu": "Full CPU",
    }.get(b, b)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--backend", choices=BACKENDS, help="Run 100-run benchmark for this backend")
    group.add_argument("--aggregate", action="store_true", help="Aggregate existing per-backend summaries into report.md")
    args = ap.parse_args()

    if args.aggregate:
        aggregate()
    else:
        run_backend(args.backend)


if __name__ == "__main__":
    main()
