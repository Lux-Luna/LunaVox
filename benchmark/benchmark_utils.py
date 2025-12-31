import statistics
import time
import platform
import sys
import subprocess
from pathlib import Path
import json

def get_device_info(pynvml_module=None, run_mode="cpu"):
    """Get CPU and GPU names for the report filename."""
    cpu_name = platform.processor()
    if sys.platform == "win32":
        try:
            output = subprocess.check_output(["wmic", "cpu", "get", "name"]).decode().split('\n')
            if len(output) > 1:
                cpu_name = output[1].strip()
        except:
            pass
    elif sys.platform == "darwin":
        try:
            cpu_name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        except:
            pass
            
    gpu_name = None
    if pynvml_module:
        try:
            pynvml_module.nvmlInit()
            handle = pynvml_module.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml_module.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode()
        except:
            pass
            
    # Clean names for filename compatibility
    def clean_name(name):
        return "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()

    cpu_clean = clean_name(cpu_name if cpu_name else "Unknown_CPU")
    
    if gpu_name:
        gpu_clean = clean_name(gpu_name)
        return f"{cpu_clean}+{gpu_clean}"
    return cpu_clean

def format_stats(data_list, round_to=2):
    """Calculate statistics from a list of numbers."""
    if not data_list:
        return {"avg": 0, "min": 0, "max": 0, "std": 0}
    return {
        "avg": round(statistics.mean(data_list), round_to),
        "min": round(min(data_list), round_to),
        "max": round(max(data_list), round_to),
        "std": round(statistics.stdev(data_list), round_to) if len(data_list) > 1 else 0
    }

def generate_markdown_report(all_results, report_path, device_info, run_mode, rounds, warmup):
    """Generate a visual-friendly markdown report."""
    lines = [
        f"# LunaVox Benchmark Report ({run_mode.upper()})",
        f"\n- **Device**: {device_info}",
        f"- **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Rounds**: {rounds}",
        f"- **Warmup**: {warmup}",
        "\n## Summary Table",
        "\n| Version | Lang | Latency(avg) | RTF(avg) | RAM(avg) | RAM(peak) | VRAM(peak) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    for res in all_results:
        s = res['statistics']
        v = res['version']
        l = res['language'].upper()
        t_avg = f"{s['latency']['avg']:.1f} ms"
        rtf_avg = f"{s['rtf']['avg']:.4f}"
        ram_avg = f"{s['ram']['avg']:.1f} MB"
        ram_peak = f"{s['ram']['max']:.1f} MB"
        v_peak = f"{s['vram']['max']:.1f} MB" if run_mode == "gpu" else "N/A"
        
        lines.append(f"| {v} | {l} | {t_avg} | {rtf_avg} | {ram_avg} | {ram_peak} | {v_peak} |")
    
    # Detail Breakdown
    lines.append("\n## Pipeline Component Latency (avg)")
    lines.append("\n| Version | Lang | Frontend | T2S | VITS | Vocoder Kernel |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for res in all_results:
        v = res['version']
        l = res['language'].upper()
        comp = res['statistics'].get('components', {})
        f_ms = f"{comp.get('Frontend', 0):.2f} ms"
        t_ms = f"{comp.get('T2S Inference', 0):.2f} ms"
        v_ms = f"{comp.get('VITS Inference', 0):.2f} ms"
        k_ms = f"{comp.get('Vocoder Kernel', 0):.2f} ms"
        lines.append(f"| {v} | {l} | {f_ms} | {t_ms} | {v_ms} | {k_ms} |")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def print_summary_terminal(all_results, run_mode):
    """Print a clean summary table to terminal."""
    if not all_results: return
    print("\n" + "=" * 130)
    print(f"{'LUNAVOX BENCHMARK SUMMARY':^130}")
    print("-" * 130)
    header = f"{'Ver':<8} | {'Lang':<5} | {'Latency(avg)':<15} | {'RTF(avg)':<10} | {'RAM(peak)':<12} | {'VRAM(peak)':<12}"
    print(header)
    print("-" * 130)

    for res in all_results:
        s = res['statistics']
        v = res['version']
        l = res['language'].upper()
        t_avg = f"{s['latency']['avg']:.1f} ms"
        rtf_avg = f"{s['rtf']['avg']:.4f}"
        r_peak = f"{s['ram']['max']:.1f} MB"
        v_peak = f"{s['vram']['max']:.1f} MB" if run_mode == "gpu" else "N/A"
        print(f"{v:<8} | {l:<5} | {t_avg:<15} | {rtf_avg:<10} | {r_peak:<12} | {v_peak:<12}")
    print("=" * 130 + "\n")
