"""LunaVox baseline benchmark driver.

Runs lunavox-cli across a matrix of {model, mode} configurations and
captures:

    - Timing from --stats-json (load / tokenize / encode / generate / decode)
    - Process RSS sampled from psutil at 100 ms intervals (wall timeline)
    - NVIDIA VRAM sampled from pynvml at 100 ms intervals (if available)
    - First-run vs second-run split (cold vs warm synthesis)

Output:

    tests/results/baseline_<timestamp>/
        raw_<tag>.json        # one per (model, mode) configuration
        summary.json          # aggregated matrix

The script is deliberately stand-alone: it only depends on psutil and,
optionally, pynvml. No LunaVox internals are imported — it drives the CLI
as a black box so results reflect what an end user would actually see.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import psutil
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "psutil is required: pip install psutil"
    ) from exc

try:
    import pynvml  # type: ignore
    pynvml.nvmlInit()
    _HAS_NVML = True
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _HAS_NVML = False
    _NVML_HANDLE = None


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_BIN = REPO_ROOT / "build" / ("lunavox-cli.exe" if os.name == "nt" else "lunavox-cli")
OUTPUT_DIR = REPO_ROOT / "tests" / "results"
REF_AUDIO = REPO_ROOT / "ref" / "ref.wav"

DEFAULT_TEXT = (
    "LunaVox runs the Qwen3-TTS pipeline on a lightweight C++ runtime. "
    "This benchmark measures cold load, warm synthesis, and memory footprint "
    "across the base and clone modes."
)

# (tag, model_dir, mode, extra_args)
DEFAULT_MATRIX: list[tuple[str, str, str, list[str]]] = [
    ("base_small__base",  "models/base_small", "base",  []),
    ("base_small__clone", "models/base_small", "clone", ["-r", str(REF_AUDIO)]),
    ("base__base",        "models/base",       "base",  []),
    ("base__clone",       "models/base",       "clone", ["-r", str(REF_AUDIO)]),
]


@dataclass
class Sample:
    t_ms: float
    rss_bytes: int
    vram_bytes: int


@dataclass
class RunResult:
    tag: str
    model_dir: str
    mode: str
    cmd: list[str]
    wall_ms: float
    exit_code: int
    stdout_tail: str
    stats: dict[str, Any]
    samples: list[Sample] = field(default_factory=list)
    rss_peak_bytes: int = 0
    vram_peak_bytes: int = 0
    vram_baseline_bytes: int = 0


class ResourceSampler(threading.Thread):
    def __init__(self, proc: subprocess.Popen, interval_s: float = 0.1) -> None:
        super().__init__(daemon=True)
        self.proc = proc
        self.interval = interval_s
        self.samples: list[Sample] = []
        self.rss_peak = 0
        self.vram_peak = 0
        self._start_ts = time.perf_counter()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            psu = psutil.Process(self.proc.pid)
        except psutil.NoSuchProcess:
            return
        while not self._stop.is_set() and self.proc.poll() is None:
            try:
                rss = psu.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            vram = 0
            if _HAS_NVML:
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
                    vram = int(info.used)
                except Exception:
                    vram = 0
            t_ms = (time.perf_counter() - self._start_ts) * 1000.0
            self.samples.append(Sample(t_ms=t_ms, rss_bytes=rss, vram_bytes=vram))
            if rss > self.rss_peak:
                self.rss_peak = rss
            if vram > self.vram_peak:
                self.vram_peak = vram
            if self._stop.wait(self.interval):
                break


def read_vram_baseline() -> int:
    if not _HAS_NVML:
        return 0
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
        return int(info.used)
    except Exception:
        return 0


def run_one(
    tag: str,
    model_dir: str,
    mode: str,
    extra_args: list[str],
    text: str,
    repeat: int,
    out_dir: Path,
) -> RunResult:
    stats_file = out_dir / f"stats_{tag}.json"
    wav_file = out_dir / f"out_{tag}.wav"

    cmd: list[str] = [
        str(CLI_BIN),
        "-m", model_dir,
        "-t", text,
        "-o", str(wav_file),
        "--stats-json", str(stats_file),
        "--warmup", "0",
        "--repeat", str(repeat),
        "--mode", mode,
    ] + extra_args

    vram_base = read_vram_baseline()
    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    sampler = ResourceSampler(proc, interval_s=0.1)
    sampler.start()

    try:
        stdout, _ = proc.communicate(timeout=900)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate()

    sampler.stop()
    sampler.join(timeout=2.0)
    wall_ms = (time.perf_counter() - start) * 1000.0

    stats: dict[str, Any] = {}
    if stats_file.exists():
        with contextlib.suppress(Exception):
            stats = json.loads(stats_file.read_text(encoding="utf-8"))

    tail = "\n".join(stdout.splitlines()[-25:]) if stdout else ""

    return RunResult(
        tag=tag,
        model_dir=model_dir,
        mode=mode,
        cmd=cmd,
        wall_ms=wall_ms,
        exit_code=proc.returncode,
        stdout_tail=tail,
        stats=stats,
        samples=sampler.samples,
        rss_peak_bytes=sampler.rss_peak,
        vram_peak_bytes=sampler.vram_peak,
        vram_baseline_bytes=vram_base,
    )


def summarize(result: RunResult) -> dict[str, Any]:
    runs = result.stats.get("runs", []) if isinstance(result.stats, dict) else []
    load_ms = result.stats.get("t_load_ms") if isinstance(result.stats, dict) else None

    first = runs[0] if runs else {}
    second = runs[1] if len(runs) > 1 else {}

    def ms(block: dict[str, Any], key: str) -> Optional[int]:
        try:
            return block.get("timing_ms", {}).get(key)
        except Exception:
            return None

    vram_used = max(0, result.vram_peak_bytes - result.vram_baseline_bytes)
    warmup_ms = result.stats.get("t_warmup_ms") if isinstance(result.stats, dict) else None
    return {
        "tag": result.tag,
        "model_dir": result.model_dir,
        "mode": result.mode,
        "exit_code": result.exit_code,
        "wall_ms": round(result.wall_ms, 1),
        "t_load_ms": load_ms,
        "t_warmup_ms": warmup_ms,
        "run1_tokenize_ms": ms(first, "tokenize"),
        "run1_encode_ms": ms(first, "encode"),
        "run1_generate_ms": ms(first, "generate"),
        "run1_llama_prefill_ms": ms(first, "llama_prefill"),
        "run1_llama_decode_loop_ms": ms(first, "llama_decode_loop"),
        "run1_talker_post_ms": ms(first, "talker_post"),
        "run1_predictor_sample_ms": ms(first, "predictor_sample"),
        "run1_decode_ms": ms(first, "decode"),
        "run1_ort_decoder_run_ms": ms(first, "ort_decoder_run"),
        "run1_pcm_gather_ms": ms(first, "pcm_gather"),
        "run1_total_ms": ms(first, "total"),
        "run2_total_ms": ms(second, "total") if second else None,
        "run2_generate_ms": ms(second, "generate") if second else None,
        "run2_llama_prefill_ms": ms(second, "llama_prefill") if second else None,
        "run2_llama_decode_loop_ms": ms(second, "llama_decode_loop") if second else None,
        "run2_talker_post_ms": ms(second, "talker_post") if second else None,
        "run2_predictor_sample_ms": ms(second, "predictor_sample") if second else None,
        "run2_decode_ms": ms(second, "decode") if second else None,
        "run2_ort_decoder_run_ms": ms(second, "ort_decoder_run") if second else None,
        "run1_rtf": first.get("rtf"),
        "run2_rtf": second.get("rtf") if second else None,
        "rss_peak_mb": round(result.rss_peak_bytes / 1024 / 1024, 1),
        "stats_rss_peak_mb": round(
            first.get("mem", {}).get("rss_peak", 0) / 1024 / 1024, 1
        ) if first else None,
        "vram_delta_mb": round(vram_used / 1024 / 1024, 1) if _HAS_NVML else None,
        "vram_baseline_mb": round(result.vram_baseline_bytes / 1024 / 1024, 1) if _HAS_NVML else None,
        "audio_sec": first.get("audio_sec") if first else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--repeat", type=int, default=2, help="runs per configuration (>=1)")
    ap.add_argument("--only", help="comma-separated tags to restrict the matrix")
    ap.add_argument("--save-samples", action="store_true", help="store per-100ms samples in raw files")
    args = ap.parse_args()

    if not CLI_BIN.exists():
        print(f"ERROR: CLI binary not found: {CLI_BIN}", file=sys.stderr)
        return 2

    matrix = DEFAULT_MATRIX
    if args.only:
        wanted = {t.strip() for t in args.only.split(",") if t.strip()}
        matrix = [row for row in DEFAULT_MATRIX if row[0] in wanted]
        if not matrix:
            print(f"ERROR: --only left an empty matrix", file=sys.stderr)
            return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / f"baseline_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bench] results -> {out_dir}")
    print(f"[bench] nvml={'on' if _HAS_NVML else 'off'}, cli={CLI_BIN}")

    summaries: list[dict[str, Any]] = []
    for tag, model_dir, mode, extra_args in matrix:
        print(f"[bench] running: {tag}  ({model_dir}, mode={mode})")
        if not (REPO_ROOT / model_dir).exists():
            print(f"[bench]   SKIP: model dir not found")
            continue
        result = run_one(tag, model_dir, mode, extra_args, args.text, args.repeat, out_dir)
        summary = summarize(result)
        summaries.append(summary)

        raw = asdict(result)
        if not args.save_samples:
            raw.pop("samples", None)
        (out_dir / f"raw_{tag}.json").write_text(
            json.dumps(raw, indent=2), encoding="utf-8"
        )
        print(f"[bench]   wall={summary['wall_ms']} ms  load={summary['t_load_ms']} ms"
              f"  run1_total={summary['run1_total_ms']} ms"
              f"  rss_peak={summary['rss_peak_mb']} MB"
              f"  vram_delta={summary['vram_delta_mb']} MB")

    (out_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    print()
    print(f"{'tag':<22}{'load_ms':>10}{'run1_ms':>10}{'run2_ms':>10}"
          f"{'rtf1':>8}{'rss_mb':>10}{'vram_mb':>10}")
    for s in summaries:
        print(f"{s['tag']:<22}"
              f"{str(s['t_load_ms']):>10}"
              f"{str(s['run1_total_ms']):>10}"
              f"{str(s['run2_total_ms']):>10}"
              f"{str(s['run1_rtf']):>8}"
              f"{str(s['rss_peak_mb']):>10}"
              f"{str(s['vram_delta_mb']):>10}")
    print()
    print(f"[bench] summary -> {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
