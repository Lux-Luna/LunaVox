"""LunaVox 0.6B benchmark — single-build, single-report driver.

Detects the active backend combination (llama.cpp backend + ONNX Runtime
execution provider) from ``build/metadata.json`` plus the host CPU/GPU,
then runs the standard warm-up-5 + repeat-100 benchmark against
``build/lunavox-cli.exe`` exactly as before.

No CLI flags — whatever is currently in ``build/`` is what gets measured.
Output filenames embed a tag derived from the active combo, e.g.

    benchmark/results/stats__llama-vulkan__ort-dml__rtx-3090.json
    benchmark/results/summary__llama-vulkan__ort-dml__rtx-3090.json
    benchmark/results/meta__llama-vulkan__ort-dml__rtx-3090.json
    benchmark/report__llama-vulkan__ort-dml__rtx-3090.md

The Markdown report records the llama.cpp backend + version, the ORT
execution provider + version, and the host CPU/GPU it ran on.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmark" / "results"
REPORT_DIR = REPO_ROOT / "benchmark"
OUTPUT_DIR = REPO_ROOT / "benchmark" / "output"

# Fixed benchmark config — held as constants so every run, across builds,
# is directly comparable.
BUILD_DIR = REPO_ROOT / "build"
CLI_BINARY = BUILD_DIR / ("lunavox-cli.exe" if os.name == "nt" else "lunavox-cli")
BUILD_META_PATH = BUILD_DIR / "metadata.json"
MODEL_DIR = REPO_ROOT / "models" / "base_small"
REF_PATH = REPO_ROOT / "ref" / "ref_0.6B.json"
BENCH_TEXT = (
    "LunaVox is a lightweight on-device text-to-speech engine built on "
    "Qwen3, optimized for low-latency streaming synthesis on consumer GPUs."
)
WARMUP_RUNS = 5
REPEAT_RUNS = 100

ORT_PROVIDER_SHORT = {
    "DmlExecutionProvider": "dml",
    "CUDAExecutionProvider": "cuda",
    "TensorrtExecutionProvider": "trt",
    "OpenVINOExecutionProvider": "openvino",
    "CoreMLExecutionProvider": "coreml",
    "ROCMExecutionProvider": "rocm",
    "CPUExecutionProvider": "cpu",
}

ORT_PROVIDER_LABEL = {
    "DmlExecutionProvider": "DirectML",
    "CUDAExecutionProvider": "CUDA",
    "TensorrtExecutionProvider": "TensorRT",
    "OpenVINOExecutionProvider": "OpenVINO",
    "CoreMLExecutionProvider": "CoreML",
    "ROCMExecutionProvider": "ROCm",
    "CPUExecutionProvider": "CPU",
}

GPU_BACKENDS_LLAMA = {
    "vulkan", "cuda", "opencl", "rocm", "hip", "metal", "kompute", "sycl",
}
GPU_PROVIDERS_ORT = {
    "DmlExecutionProvider", "CUDAExecutionProvider",
    "TensorrtExecutionProvider", "ROCMExecutionProvider",
    "OpenVINOExecutionProvider", "CoreMLExecutionProvider",
}


# ---------------------------------------------------------------------------
# Backend + hardware detection
# ---------------------------------------------------------------------------


def load_build_metadata() -> dict:
    if not BUILD_META_PATH.exists():
        sys.exit(
            f"ERROR: {BUILD_META_PATH} not found — current build did not record "
            "backend metadata. Re-run `lunavox build` to populate it."
        )
    with open(BUILD_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _powershell(cmd: str) -> str:
    try:
        out = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command", cmd],
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return out.decode("utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001
        return ""


def detect_cpu_name() -> str:
    if os.name == "nt":
        out = _powershell(
            "Get-CimInstance -ClassName Win32_Processor | "
            "Select-Object -ExpandProperty Name"
        )
        if out:
            return out.splitlines()[0].strip()
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or "unknown-cpu"


def detect_gpu_name() -> str | None:
    # Prefer NVML — exact name of the active CUDA device, no virtual adapters.
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                return name.strip()
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        pass

    if os.name == "nt":
        out = _powershell(
            "Get-CimInstance -ClassName Win32_VideoController | "
            "Where-Object { $_.AdapterCompatibility -notmatch 'Microsoft' } | "
            "Select-Object -ExpandProperty Name"
        )
        if out:
            return out.splitlines()[0].strip()
    return None


def _slug(s: str) -> str:
    s = re.sub(r"[®™()\[\]]", "", s)
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")
    return s.lower()


def gpu_short_tag(name: str) -> str:
    m = re.search(
        r"\b(RTX|GTX|RX|Arc|Radeon|Iris|UHD)\s*([A-Za-z0-9]+(?:\s?Ti|\s?Super|\s?XT)?)",
        name,
        re.IGNORECASE,
    )
    if m:
        family = m.group(1).lower()
        model = re.sub(r"\s+", "-", m.group(2).strip()).lower()
        return f"{family}-{model}"
    return _slug(name)[:32]


def cpu_short_tag(name: str) -> str:
    m = re.search(
        r"\b(i[3579]-\w+|Ryzen[\s-]\w+(?:[\s-]\w+)?|EPYC[\s-]\w+|Xeon[\s-]\w+)\b",
        name,
    )
    if m:
        return _slug(m.group(0))
    return _slug(name)[:32]


def derive_run_tag(
    build_meta: dict, gpu_name: str | None, cpu_name: str
) -> tuple[str, str]:
    """Return ``(filename_tag, human_label)``.

    ``filename_tag`` looks like ``llama-vulkan__ort-dml__rtx-3090`` for a
    GPU-using combo, or ``llama-cpu__ort-cpu__i9-12900k`` for pure CPU.
    """
    llama_backend = (build_meta.get("llama") or {}).get("backend", "unknown").lower()
    ort_provider = (build_meta.get("onnx") or {}).get("provider", "unknown")

    ort_short = ORT_PROVIDER_SHORT.get(
        ort_provider, _slug(ort_provider.replace("ExecutionProvider", "")) or "unknown"
    )
    llama_short = _slug(llama_backend) or "unknown"

    uses_gpu = (
        llama_backend in GPU_BACKENDS_LLAMA or ort_provider in GPU_PROVIDERS_ORT
    )
    if uses_gpu and gpu_name:
        hw_short = gpu_short_tag(gpu_name)
        hw_label = gpu_name
    else:
        hw_short = cpu_short_tag(cpu_name)
        hw_label = cpu_name

    tag = f"llama-{llama_short}__ort-{ort_short}__{hw_short}"
    label = (
        f"llama.cpp/{llama_backend} + ORT/"
        f"{ORT_PROVIDER_LABEL.get(ort_provider, ort_provider)} on {hw_label}"
    )
    return tag, label


# ---------------------------------------------------------------------------
# VRAM sampler
# ---------------------------------------------------------------------------


class VramSampler:
    """Polls NVML ``used`` memory every 100 ms while a subprocess is running.

    Records ``(timestamp, used_bytes)`` samples. After ``stop()``, exposes
    ``peak_bytes`` (max used minus baseline), ``steady_mb`` (mean of the last
    90 % of samples minus baseline), and ``baseline_bytes`` (initial reading).
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
        except Exception as e:  # noqa: BLE001
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
            assert self._pynvml is not None
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = OUTPUT_DIR / "output.wav"

    cmd = [
        str(CLI_BINARY),
        "-m", str(MODEL_DIR),
        "-r", str(REF_PATH),
        "-t", BENCH_TEXT,
        "-o", str(output_stem),
        "--warmup", str(WARMUP_RUNS),
        "--repeat", str(REPEAT_RUNS),
        "--stats-json", str(stats_path),
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
    ttfbs = [
        float(r.get("stream", {}).get("t_first_audio_ms", r["timing_ms"].get("first_audio", 0)))
        for r in runs
    ]
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


def capture_meta(
    build_meta: dict, gpu_name: str | None, cpu_name: str, run_tag: str, run_label: str
) -> dict:
    def _run(cmd: list[str]) -> str:
        try:
            out = subprocess.check_output(cmd, cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL)
            return out.decode("utf-8", errors="replace").strip()
        except Exception:  # noqa: BLE001
            return ""

    uname = platform.uname()
    return {
        "run_tag": run_tag,
        "run_label": run_label,
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
        "cpu_name": cpu_name,
        "gpu_name": gpu_name,
        "build": build_meta,
        "cli_binary": str(CLI_BINARY.relative_to(REPO_ROOT)).replace("\\", "/"),
        "cli_binary_mtime": (
            datetime.fromtimestamp(CLI_BINARY.stat().st_mtime, tz=timezone.utc)
            .isoformat(timespec="seconds")
            if CLI_BINARY.exists() else None
        ),
    }


# ---------------------------------------------------------------------------
# Report rendering (single-config)
# ---------------------------------------------------------------------------


def render_report(summary: dict, run_label: str) -> str:
    meta = summary["meta"]
    cfg = summary["config"]
    build = meta.get("build", {})
    host = meta.get("platform", {})
    llama_info = build.get("llama", {}) or {}
    ort_info = build.get("onnx", {}) or {}
    ort_provider = ort_info.get("provider", "?")
    ort_label = ORT_PROVIDER_LABEL.get(ort_provider, ort_provider)

    out: list[str] = []
    out.append(f"# LunaVox 0.6B Benchmark — {run_label}")
    out.append("")
    out.append(f"Generated: {meta.get('timestamp_utc', '')}  ")
    out.append(f"Git commit: `{meta.get('git_commit', '')[:12]}`  ")
    out.append(f"Run tag: `{meta.get('run_tag', '')}`")
    out.append("")

    out.append("## Active Backend Combo")
    out.append("")
    out.append("| Component | Backend | Version | Provider / EP | Build platform |")
    out.append("| :--- | :--- | :--- | :--- | :--- |")
    out.append(
        f"| llama.cpp (talker + predictor GGUF) | `{llama_info.get('backend', '?')}` | "
        f"`{llama_info.get('version', '?')}` | — | {llama_info.get('platform', '?')} |"
    )
    out.append(
        f"| ONNX Runtime (codec encoder + decoder) | onnxruntime | "
        f"`{ort_info.get('version', '?')}` | `{ort_provider}` ({ort_label}) | "
        f"{ort_info.get('platform', '?')} |"
    )
    out.append("")

    out.append("## Host")
    out.append("")
    out.append(f"- **CPU**: {meta.get('cpu_name', 'unknown')}")
    out.append(f"- **GPU**: {meta.get('gpu_name') or 'n/a (no discrete GPU detected)'}")
    out.append(
        f"- **OS**: {host.get('system', '')} {host.get('release', '')} "
        f"({host.get('machine', '')})"
    )
    out.append(f"- **Python**: {meta.get('python_version', '')}")
    out.append(f"- **Conda env**: `{meta.get('conda_env', '')}`")
    out.append(f"- **CLI binary**: `{meta.get('cli_binary', '')}` (mtime: {meta.get('cli_binary_mtime', '')})")
    out.append("")

    out.append("## Configuration")
    out.append("")
    out.append(f"- **Model**: `{cfg['model_dir']}` (Qwen3-TTS-12Hz-0.6B-Base)")
    out.append(f"- **Voice reference**: `{cfg['ref']}` (pre-encoded codes)")
    out.append(f"- **Sample rate**: {cfg.get('sample_rate', 'n/a')} Hz")
    out.append(f"- **Warm-up**: {cfg['warmup_runs']} runs (excluded from stats)")
    out.append(f"- **Repeat**: {cfg['repeat_runs']} runs")
    out.append(f"- **First chunk frames**: {cfg.get('first_chunk_frames', 'n/a')}")
    out.append("- **Text** (25 words):")
    out.append("")
    out.append(f"  > {cfg['text']}")
    out.append("")

    out.append("## 1. Latency (total wall time per synth, ms)")
    out.append("")
    out.append("| Metric | mean | p50 | p95 | p99 | min | max | stddev |")
    out.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    t = summary["total_ms"]
    out.append(
        f"| total_ms | {t['mean']:.1f} | {t['p50']:.1f} | {t['p95']:.1f} | {t['p99']:.1f} | "
        f"{t['min']:.1f} | {t['max']:.1f} | {t['stddev']:.1f} |"
    )
    tt = summary["ttfb_ms"]
    out.append(
        f"| ttfb_ms | {tt['mean']:.1f} | {tt['p50']:.1f} | {tt['p95']:.1f} | {tt['p99']:.1f} | "
        f"{tt['min']:.1f} | {tt['max']:.1f} | {tt['stddev']:.1f} |"
    )
    out.append("")

    rtf = summary["rtf"]
    out.append("## 2. Real-Time Factor")
    out.append("")
    out.append(
        f"- **Mean RTF**: {rtf['mean']:.4f} → **{summary['realtime_multiplier']:.2f}× realtime**"
    )
    out.append(
        f"- **p50 / p95 / p99**: {rtf['p50']:.4f} / {rtf['p95']:.4f} / {rtf['p99']:.4f}"
    )
    out.append("")

    st = summary["stage_mean_ms"]
    out.append("## 3. Stage Breakdown (per-run mean, ms)")
    out.append("")
    out.append("| Tokenize | Encode | Generate (LLM) | Decode (codec) |")
    out.append("| ---: | ---: | ---: | ---: |")
    out.append(
        f"| {st['tokenize']:.1f} | {st['encode']:.1f} | {st['generate']:.1f} | {st['decode']:.1f} |"
    )
    out.append("")

    m = summary["memory"]
    out.append("## 4. Memory Footprint")
    out.append("")
    out.append(
        "| Peak RSS max (MB) | Peak RSS mean (MB) | Peak physical (MB) | "
        "Peak VRAM Δ (MB) | Steady VRAM Δ (MB) |"
    )
    out.append("| ---: | ---: | ---: | ---: | ---: |")
    vram_peak = f"{m['vram_peak_mb']:.1f}" if m.get("vram_peak_mb") is not None else "n/a"
    vram_steady = f"{m['vram_steady_mb']:.1f}" if m.get("vram_steady_mb") is not None else "n/a"
    out.append(
        f"| {m['rss_peak_max_mb']:.1f} | {m['rss_peak_mean_mb']:.1f} | "
        f"{m['phys_peak_max_mb']:.1f} | {vram_peak} | {vram_steady} |"
    )
    if m.get("vram_skipped_reason"):
        out.append("")
        out.append(f"VRAM sampler note: {m['vram_skipped_reason']}")
    out.append("")

    cps = summary["throughput_chars_per_sec"]
    out.append("## 5. Throughput")
    out.append("")
    out.append(f"- Audio duration mean: **{summary['audio_duration_s']['mean']:.2f} s**")
    out.append(f"- Chars/sec mean / p95: **{cps['mean']:.1f} / {cps['p95']:.1f}**")
    out.append(
        f"- Load time: {summary['load']['t_load_ms']} ms "
        f"(in-load warm-up: {summary['load']['t_warmup_ms']} ms)"
    )
    out.append(
        f"- Wall elapsed (load + warmup + repeats): {summary.get('wall_elapsed_s', 0)} s"
    )
    out.append("")

    out.append("## Notes")
    out.append("")
    out.append("- **RTF** = synth wall time / generated audio duration. Lower is better; <1.0 means faster than realtime.")
    out.append("- **TTFB** is the wall-clock delay from synth start to the first PCM sample becoming available via the streaming decoder.")
    out.append("- **VRAM Δ** is sampled externally via NVML at 100 ms intervals, minus the pre-run baseline; only meaningful when an NVIDIA GPU is present.")
    out.append(f"- Raw per-run records: `benchmark/results/stats__{meta.get('run_tag', '')}.json`")
    out.append(f"- Aggregated metrics: `benchmark/results/summary__{meta.get('run_tag', '')}.json`")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def _print_single_summary(label: str, s: dict) -> None:
    tot = s["total_ms"]
    ttfb = s["ttfb_ms"]
    rtf = s["rtf"]
    mem = s["memory"]
    print("", flush=True)
    print(f"=== {label} — {s['config']['repeat_runs']} runs ===", flush=True)
    print(
        f"  total_ms  mean={tot['mean']:.1f}  p50={tot['p50']:.1f}  "
        f"p95={tot['p95']:.1f}  p99={tot['p99']:.1f}",
        flush=True,
    )
    print(
        f"  ttfb_ms   mean={ttfb['mean']:.1f}  p50={ttfb['p50']:.1f}  "
        f"p95={ttfb['p95']:.1f}  p99={ttfb['p99']:.1f}",
        flush=True,
    )
    print(
        f"  rtf       mean={rtf['mean']:.4f}  (×{s['realtime_multiplier']} realtime)",
        flush=True,
    )
    print(f"  rss_peak  max={mem['rss_peak_max_mb']:.1f} MB", flush=True)
    vram_str = (
        f"{mem['vram_peak_mb']:.1f} MB"
        if mem.get("vram_peak_mb") is not None
        else f"n/a ({mem.get('vram_skipped_reason')})"
    )
    print(f"  vram_peak {vram_str}", flush=True)
    print(
        f"  load_ms={s['load']['t_load_ms']}  warmup_ms={s['load']['t_warmup_ms']}",
        flush=True,
    )


def main() -> None:
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()

    build_meta = load_build_metadata()
    cpu_name = detect_cpu_name()
    gpu_name = detect_gpu_name()
    run_tag, run_label = derive_run_tag(build_meta, gpu_name, cpu_name)

    print(f"[bench] active combo: {run_label}", flush=True)
    print(f"[bench] run tag:      {run_tag}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = RESULTS_DIR / f"stats__{run_tag}.json"
    summary_path = RESULTS_DIR / f"summary__{run_tag}.json"
    meta_path = RESULTS_DIR / f"meta__{run_tag}.json"
    report_path = REPORT_DIR / f"report__{run_tag}.md"

    meta = capture_meta(build_meta, gpu_name, cpu_name, run_tag, run_label)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[bench] wrote {meta_path.relative_to(REPO_ROOT)}", flush=True)

    sampler = VramSampler()
    if sampler.available:
        print(
            f"[bench] VRAM sampler active (baseline="
            f"{sampler.baseline_bytes / 1024 / 1024:.1f} MB)",
            flush=True,
        )
    else:
        print(f"[bench] VRAM sampler SKIPPED ({sampler.skip_reason})", flush=True)

    sampler.start()
    rc, wall = run_cli(stats_path)
    sampler.stop()
    if rc != 0:
        sys.exit(f"ERROR: lunavox-cli exited with code {rc}")

    summary = summarize(stats_path, sampler.metrics(), wall)
    summary["meta"] = meta
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[bench] wrote {summary_path.relative_to(REPO_ROOT)}", flush=True)

    report = render_report(summary, run_label)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[bench] wrote {report_path.relative_to(REPO_ROOT)}", flush=True)

    _print_single_summary(run_label, summary)


if __name__ == "__main__":
    main()
