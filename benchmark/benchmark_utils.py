"""
LunaVox Benchmark Utilities
Enhanced utilities for multi-dimension benchmark reporting and output management.
"""

import statistics
import time
import platform
import sys
import subprocess
import json
import wave
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional


def get_device_info(pynvml_module=None, run_mode: str = "cpu") -> str:
    """Get CPU and GPU names for the report filename."""
    cpu_name = platform.processor()
    if sys.platform == "win32":
        try:
            output = subprocess.check_output(["wmic", "cpu", "get", "name"]).decode().split('\n')
            if len(output) > 1:
                cpu_name = output[1].strip()
        except Exception:
            pass
    elif sys.platform == "darwin":
        try:
            cpu_name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        except Exception:
            pass

    gpu_name = None
    if pynvml_module and run_mode == "gpu":
        try:
            pynvml_module.nvmlInit()
            handle = pynvml_module.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml_module.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode()
        except Exception:
            pass

    def clean_name(name: str) -> str:
        return "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()

    cpu_clean = clean_name(cpu_name if cpu_name else "Unknown_CPU")

    if gpu_name:
        gpu_clean = clean_name(gpu_name)
        return f"{cpu_clean}+{gpu_clean}"
    return cpu_clean


def format_stats(data_list: List[float], round_to: int = 2) -> Dict[str, float]:
    """Calculate statistics from a list of numbers."""
    if not data_list:
        return {"avg": 0, "min": 0, "max": 0, "std": 0}
    return {
        "avg": round(statistics.mean(data_list), round_to),
        "min": round(min(data_list), round_to),
        "max": round(max(data_list), round_to),
        "std": round(statistics.stdev(data_list), round_to) if len(data_list) > 1 else 0
    }


def get_timestamp_str() -> str:
    """Get consistent timestamp string for filenames and reports."""
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def get_audio_duration(path: Path) -> float:
    """Get duration of WAV file in seconds."""
    try:
        with wave.open(str(path), 'rb') as f:
            return f.getnframes() / float(f.getframerate())
    except Exception:
        return 0.0


def save_warmup_audio(source_path: Path, output_dir: Path) -> Optional[Path]:
    """Save first warmup audio to the designated output directory."""
    if not source_path.exists():
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_path = output_dir / "warmup.wav"
    shutil.copy2(source_path, dest_path)
    return dest_path


def save_json_result(result: Dict[str, Any], output_dir: Path) -> Path:
    """Save benchmark result as JSON to the designated output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    return json_path


def generate_markdown_report(
    all_results: List[Dict[str, Any]],
    report_path: Path,
    device_info: str,
    timestamp: str
) -> None:
    """Generate a comprehensive markdown report with multi-dimension comparison."""
    lines = [
        "# LunaVox Benchmark Report",
        f"\n- **Generated**: {timestamp}",
        f"- **Device**: {device_info}",
        "",
        "## Summary Table",
        "",
        "| Env | Mode | Version | Lang | Latency(avg) | RTF(avg) | RAM(peak) | VRAM(peak) |",
        "|:---:|:----:|:-------:|:----:|-------------:|---------:|----------:|-----------:|"
    ]

    for res in all_results:
        s = res['statistics']
        env = res['environment'].upper()
        mode = res['mode'][:3].upper()  # PER or REF
        ver = res['version']
        lang = res['language'].upper()
        lat_avg = f"{s['latency']['avg']:.1f} ms"
        rtf_avg = f"{s['rtf']['avg']:.4f}"
        ram_peak = f"{s['ram']['max']:.1f} MB"
        vram_peak = f"{s['vram']['max']:.1f} MB" if res['environment'] == "gpu" else "N/A"

        lines.append(f"| {env} | {mode} | {ver} | {lang} | {lat_avg} | {rtf_avg} | {ram_peak} | {vram_peak} |")

    # Environment Comparison Section (if mixed CPU/GPU results exist)
    cpu_results = [r for r in all_results if r['environment'] == 'cpu']
    gpu_results = [r for r in all_results if r['environment'] == 'gpu']

    if cpu_results and gpu_results:
        lines.extend([
            "",
            "## CPU vs GPU Comparison",
            "",
            "| Mode | Version | Lang | CPU Latency | GPU Latency | Speedup |",
            "|:----:|:-------:|:----:|------------:|------------:|--------:|"
        ])

        for cpu_res in cpu_results:
            # Find matching GPU result
            gpu_match = next(
                (g for g in gpu_results 
                 if g['mode'] == cpu_res['mode'] 
                 and g['version'] == cpu_res['version'] 
                 and g['language'] == cpu_res['language']),
                None
            )
            if gpu_match:
                cpu_lat = cpu_res['statistics']['latency']['avg']
                gpu_lat = gpu_match['statistics']['latency']['avg']
                speedup = cpu_lat / gpu_lat if gpu_lat > 0 else 0

                lines.append(
                    f"| {cpu_res['mode'][:3].upper()} | {cpu_res['version']} | "
                    f"{cpu_res['language'].upper()} | {cpu_lat:.1f} ms | "
                    f"{gpu_lat:.1f} ms | {speedup:.2f}x |"
                )

    # Component Latency Breakdown
    lines.extend([
        "",
        "## Component Latency Breakdown (avg)",
        "",
        "| Env | Mode | Ver | Lang | Frontend | T2S | VITS | Vocoder |",
        "|:---:|:----:|:---:|:----:|---------:|----:|-----:|--------:|"
    ])

    for res in all_results:
        comp = res['statistics'].get('components', {})
        env = res['environment'].upper()
        mode = res['mode'][:3].upper()
        ver = res['version']
        lang = res['language'].upper()
        f_ms = f"{comp.get('Frontend', 0):.1f} ms"
        t_ms = f"{comp.get('T2S Inference', 0):.1f} ms"
        v_ms = f"{comp.get('VITS Inference', 0):.1f} ms"
        k_ms = f"{comp.get('Vocoder Kernel', 0):.1f} ms"

        lines.append(f"| {env} | {mode} | {ver} | {lang} | {f_ms} | {t_ms} | {v_ms} | {k_ms} |")

    # Test Configuration Summary
    if all_results:
        sample = all_results[0]
        lines.extend([
            "",
            "## Test Configuration",
            "",
            f"- **Warmup Rounds**: {sample.get('warmup_rounds', 'N/A')}",
            f"- **Test Rounds**: {sample.get('test_rounds', 'N/A')}",
            f"- **Environments Tested**: {', '.join(sorted(set(r['environment'] for r in all_results)))}",
            f"- **Modes Tested**: {', '.join(sorted(set(r['mode'] for r in all_results)))}",
            f"- **Versions Tested**: {', '.join(sorted(set(r['version'] for r in all_results)))}",
            f"- **Languages Tested**: {', '.join(sorted(set(r['language'] for r in all_results)))}"
        ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_summary_terminal(all_results: List[Dict[str, Any]]) -> None:
    """Print a clean summary table to terminal."""
    if not all_results:
        return

    print("\n" + "=" * 140)
    print(f"{'LUNAVOX BENCHMARK SUMMARY':^140}")
    print("-" * 140)
    header = (
        f"{'Env':<5} | {'Mode':<9} | {'Ver':<6} | {'Lang':<5} | "
        f"{'Latency(avg)':<14} | {'RTF(avg)':<10} | "
        f"{'RAM(peak)':<12} | {'VRAM(peak)':<12}"
    )
    print(header)
    print("-" * 140)

    for res in all_results:
        s = res['statistics']
        env = res['environment'].upper()
        mode = res['mode']
        ver = res['version']
        lang = res['language'].upper()
        lat_avg = f"{s['latency']['avg']:.1f} ms"
        rtf_avg = f"{s['rtf']['avg']:.4f}"
        ram_peak = f"{s['ram']['max']:.1f} MB"
        vram_peak = f"{s['vram']['max']:.1f} MB" if res['environment'] == "gpu" else "N/A"

        print(
            f"{env:<5} | {mode:<9} | {ver:<6} | {lang:<5} | "
            f"{lat_avg:<14} | {rtf_avg:<10} | "
            f"{ram_peak:<12} | {vram_peak:<12}"
        )

    print("=" * 140 + "\n")
