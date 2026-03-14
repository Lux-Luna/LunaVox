#!/usr/bin/env python3
"""
Benchmark PyTorch Qwen3-TTS pipeline vs qwen3-tts.cpp CLI on macOS.

Runs two scenarios for both pipelines:
1) basic TTS (no reference audio)
2) voice cloning (x-vector only)

For each run, captures:
- wall clock time (`/usr/bin/time -l`: real seconds)
- peak RSS (`/usr/bin/time -l`: maximum resident set size)

Outputs:
- docs/benchmark_pytorch_vs_cpp.json
- docs/benchmark_pytorch_vs_cpp.png
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = PROJECT_ROOT / "models"
HF_MODEL_DIR = MODELS_DIR / "Qwen3-TTS-12Hz-0.6B-Base"
CPP_CLI = PROJECT_ROOT / "build" / "qwen3-tts-cli"
REF_AUDIO = PROJECT_ROOT / "examples" / "readme_clone_input.wav"

# Keep benchmark prompts aligned with the known-good quickstart examples.
BASIC_TEXT = "Hello from qwen3-tts.cpp running on macOS with CoreML by default."
CLONE_TEXT = "This is a voice cloning example generated from the sample audio file in this directory."


def _parse_time_l(stderr: str) -> tuple[float, int]:
    real_match = re.search(r"^\s*([0-9]+(?:\.[0-9]+)?)\s+real\b", stderr, re.MULTILINE)
    rss_match = re.search(r"^\s*([0-9]+)\s+maximum resident set size\b", stderr, re.MULTILINE)
    if not real_match:
        raise RuntimeError("Failed to parse '/usr/bin/time -l' real time")
    if not rss_match:
        raise RuntimeError("Failed to parse '/usr/bin/time -l' maximum resident set size")
    return float(real_match.group(1)), int(rss_match.group(1))


def _run_with_time_l(cmd: list[str], env: dict[str, str] | None = None) -> dict[str, Any]:
    wrapped = ["/usr/bin/time", "-l"] + cmd
    proc = subprocess.run(
        wrapped,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + f"\nreturncode={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    real_s, max_rss_bytes = _parse_time_l(proc.stderr)
    return {
        "cmd": cmd,
        "real_s": real_s,
        "max_rss_bytes": max_rss_bytes,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _write_wav(path: Path, audio, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    arr = np.asarray(audio, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), arr, sample_rate)


def _run_pytorch_worker(mode: str, output: Path, max_tokens: int) -> None:
    import torch
    from qwen_tts import Qwen3TTSModel

    if not HF_MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing HF model directory: {HF_MODEL_DIR}")
    if mode == "voice_clone" and not REF_AUDIO.exists():
        raise FileNotFoundError(f"Missing reference audio: {REF_AUDIO}")

    model = Qwen3TTSModel.from_pretrained(
        str(HF_MODEL_DIR),
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    model.model = model.model.eval()

    if mode == "voice_clone":
        wavs, sr = model.generate_voice_clone(
            text=CLONE_TEXT,
            language="English",
            ref_audio=str(REF_AUDIO),
            x_vector_only_mode=True,
            non_streaming_mode=False,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            max_new_tokens=max_tokens,
        )
        _write_wav(output, wavs[0], sr)
    elif mode == "basic":
        input_ids = model._tokenize_texts([model._build_assistant_text(BASIC_TEXT)])
        gen_kwargs = model._merge_generate_kwargs(
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            max_new_tokens=max_tokens,
        )
        talker_codes_list, _ = model.model.generate(
            input_ids=input_ids,
            languages=["English"],
            speakers=[None],
            non_streaming_mode=False,
            **gen_kwargs,
        )
        wavs, sr = model.model.speech_tokenizer.decode([{"audio_codes": talker_codes_list[0]}])
        _write_wav(output, wavs[0], sr)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _build_plot(report: dict[str, Any], out_png: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    cases = ["basic", "voice_clone"]
    case_titles = {"basic": "Basic TTS", "voice_clone": "Voice Cloning"}

    py_times = [report["results"][c]["pytorch"]["real_s"] for c in cases]
    cpp_times = [report["results"][c]["cpp"]["real_s"] for c in cases]
    py_mem = [report["results"][c]["pytorch"]["max_rss_gb"] for c in cases]
    cpp_mem = [report["results"][c]["cpp"]["max_rss_gb"] for c in cases]

    labels = [case_titles[c] for c in cases]
    x = np.arange(len(cases))
    w = 0.36

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(13, 6), dpi=180)

    b1 = ax_t.bar(x - w / 2, py_times, w, label="PyTorch", color="#3B82F6")
    b2 = ax_t.bar(x + w / 2, cpp_times, w, label="qwen3-tts.cpp", color="#16A34A")
    ax_t.set_title("End-to-End Latency (lower is better)")
    ax_t.set_ylabel("Seconds")
    ax_t.set_xticks(x, labels)
    ax_t.legend(loc="upper right")

    for i, (py, cpp) in enumerate(zip(py_times, cpp_times)):
        speedup = py / cpp if cpp > 0 else 0.0
        ax_t.text(i, max(py, cpp) * 1.03, f"{speedup:.2f}x faster", ha="center", va="bottom", fontsize=9)

    m1 = ax_m.bar(x - w / 2, py_mem, w, label="PyTorch", color="#60A5FA")
    m2 = ax_m.bar(x + w / 2, cpp_mem, w, label="qwen3-tts.cpp", color="#22C55E")
    ax_m.set_title("Peak RSS (lower is better)")
    ax_m.set_ylabel("GB")
    ax_m.set_xticks(x, labels)
    ax_m.legend(loc="upper right")

    for i, (py, cpp) in enumerate(zip(py_mem, cpp_mem)):
        if py > 0:
            delta_pct = (1.0 - (cpp / py)) * 100.0
            ax_m.text(
                i,
                max(py, cpp) * 1.03,
                f"{delta_pct:+.1f}% vs PyTorch",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    for bars in (b1, b2, m1, m2):
        for b in bars:
            h = b.get_height()
            b.axes.text(
                b.get_x() + b.get_width() / 2,
                h * 1.005,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    hw = report.get("hardware", "unknown")
    fig.suptitle(f"Qwen3-TTS Benchmark: PyTorch vs qwen3-tts.cpp ({hw})", fontsize=13, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Measured with /usr/bin/time -l; values include model load + synthesis in a fresh process.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _human_hw() -> str:
    sysname = platform.system()
    machine = platform.machine()
    cpu = platform.processor() or machine
    return f"{sysname} / {cpu}"


def _benchmark_all(python_exe: Path, out_json: Path, out_png: Path) -> dict[str, Any]:
    out_dir = PROJECT_ROOT / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        "basic": {
            "max_tokens": 320,
            "cpp_out": out_dir / "cpp_basic.wav",
            "py_out": out_dir / "py_basic.wav",
        },
        "voice_clone": {
            "max_tokens": 320,
            "cpp_out": out_dir / "cpp_clone.wav",
            "py_out": out_dir / "py_clone.wav",
        },
    }

    results: dict[str, Any] = {}

    for case, cfg in cases.items():
        max_tokens = int(cfg["max_tokens"])

        py_cmd = [
            str(python_exe),
            str(Path(__file__).resolve()),
            "--worker",
            "--mode",
            case,
            "--output",
            str(cfg["py_out"]),
            "--max-tokens",
            str(max_tokens),
        ]
        py_res = _run_with_time_l(py_cmd)

        cpp_cmd = [
            str(CPP_CLI),
            "-m",
            "models",
            "-t",
            BASIC_TEXT if case == "basic" else CLONE_TEXT,
            "-o",
            str(cfg["cpp_out"]),
            "--max-tokens",
            str(max_tokens),
        ]
        if case == "voice_clone":
            cpp_cmd.extend(["-r", str(REF_AUDIO)])

        cpp_env = dict(os.environ)
        cpp_env.pop("QWEN3_TTS_USE_COREML", None)
        cpp_env.pop("QWEN3_TTS_LOW_MEM", None)
        cpp_res = _run_with_time_l(cpp_cmd, env=cpp_env)

        py_time = py_res["real_s"]
        cpp_time = cpp_res["real_s"]
        py_rss = py_res["max_rss_bytes"]
        cpp_rss = cpp_res["max_rss_bytes"]

        results[case] = {
            "pytorch": {
                "real_s": py_time,
                "max_rss_bytes": py_rss,
                "max_rss_gb": py_rss / (1024.0 ** 3),
                "output_wav": str(cfg["py_out"]),
            },
            "cpp": {
                "real_s": cpp_time,
                "max_rss_bytes": cpp_rss,
                "max_rss_gb": cpp_rss / (1024.0 ** 3),
                "output_wav": str(cfg["cpp_out"]),
            },
            "speedup_cpp_vs_pytorch": (py_time / cpp_time) if cpp_time > 0 else 0.0,
            "memory_delta_cpp_vs_pytorch_pct": ((1.0 - (cpp_rss / py_rss)) * 100.0) if py_rss > 0 else 0.0,
        }

    report: dict[str, Any] = {
        "hardware": _human_hw(),
        "python_exe": str(python_exe),
        "cpp_cli": str(CPP_CLI),
        "reference_audio": str(REF_AUDIO),
        "results": results,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    _build_plot(report, out_png)
    return report


def _update_readme(readme_path: Path, fig_path: Path, report: dict[str, Any]) -> None:
    data = readme_path.read_text()
    rel_fig = fig_path.relative_to(PROJECT_ROOT).as_posix()

    basic = report["results"]["basic"]
    clone = report["results"]["voice_clone"]
    summary = (
        f"**Benchmark Snapshot (PyTorch vs qwen3-tts.cpp):** "
        f"Basic {basic['speedup_cpp_vs_pytorch']:.2f}x faster, "
        f"Clone {clone['speedup_cpp_vs_pytorch']:.2f}x faster. "
        f"Peak RSS delta: Basic {basic['memory_delta_cpp_vs_pytorch_pct']:+.1f}%, "
        f"Clone {clone['memory_delta_cpp_vs_pytorch_pct']:+.1f}%.\n"
    )

    bench_block = (
        f"![PyTorch vs qwen3-tts.cpp benchmark](./{rel_fig})\n\n"
        + summary
        + "\n"
    )

    if data.startswith("# qwen3-tts.cpp\n\n"):
        prefix = "# qwen3-tts.cpp\n\n"
        rest = data[len(prefix):]
        if rest.startswith("![PyTorch vs qwen3-tts.cpp benchmark]"):
            # Replace existing benchmark block at top.
            parts = rest.split("\n\n", 2)
            if len(parts) >= 3:
                rest = parts[2]
            else:
                rest = "\n".join(parts[1:]) if len(parts) > 1 else ""
        readme_path.write_text(prefix + bench_block + rest)
    else:
        readme_path.write_text("# qwen3-tts.cpp\n\n" + bench_block + data)


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark PyTorch vs qwen3-tts.cpp and generate comparison graph")
    p.add_argument("--worker", action="store_true", help="Internal: run one PyTorch worker case")
    p.add_argument("--mode", choices=["basic", "voice_clone"], help="Worker mode")
    p.add_argument("--output", type=Path, help="Worker output wav path")
    p.add_argument("--max-tokens", type=int, default=200, help="Max generated tokens")
    p.add_argument("--out-json", type=Path, default=DOCS_DIR / "benchmark_pytorch_vs_cpp.json")
    p.add_argument("--out-png", type=Path, default=DOCS_DIR / "benchmark_pytorch_vs_cpp.png")
    p.add_argument("--update-readme", action="store_true", help="Insert graph + summary at the top of README")
    args = p.parse_args()

    if args.worker:
        if not args.mode or not args.output:
            raise SystemExit("--worker requires --mode and --output")
        _run_pytorch_worker(args.mode, args.output, args.max_tokens)
        return 0

    python_exe = PROJECT_ROOT / ".venv" / "bin" / "python"
    if not python_exe.exists():
        raise SystemExit(f"Missing venv python: {python_exe}")
    if not CPP_CLI.exists():
        raise SystemExit(f"Missing C++ CLI binary: {CPP_CLI}")
    if not REF_AUDIO.exists():
        raise SystemExit(f"Missing reference audio: {REF_AUDIO}")

    report = _benchmark_all(python_exe=python_exe, out_json=args.out_json, out_png=args.out_png)
    print(json.dumps(report, indent=2))

    if args.update_readme:
        _update_readme(PROJECT_ROOT / "README.md", args.out_png, report)
        print(f"Updated README.md with benchmark figure: {args.out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
