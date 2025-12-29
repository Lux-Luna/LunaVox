import sys
import os
import re
import subprocess
from pathlib import Path
import csv
from datetime import datetime

# Setup paths
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
BENCHMARK_RESULTS_FILE = REPO_ROOT / "benchmark_history.csv"

def run_benchmark(lang='en'):
    print(f"Running benchmark ({lang})...")
    
    script = "Tutorial/v2_quick_tryout/quick_tryout_en.py" if lang == 'en' else "scripts/test_zh.py"
    
    cmd = [
        sys.executable, script
    ]
    
    try:
        # Run process and capture output
        # In this project, logs are typically written to stderr
        result = subprocess.run(
            cmd, 
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True
        )
        output = result.stderr + "\n" + result.stdout
        
    except subprocess.CalledProcessError as e:
        print(f"Benchmark run ({lang}) failed with exit code {e.returncode}")
        print(e.stderr)
        return

    # Parse logs for timing information
    frontend_match = re.search(r"Frontend \(\w+\) took: ([\d\.]+)ms", output)
    t2s_match = re.search(r"T2S Inference took: ([\d\.]+)ms", output)
    vits_match = re.search(r"VITS Inference took: ([\d\.]+)ms", output)
    total_match = re.search(r"Total TTS Latency: ([\d\.]+)ms", output)
    
    frontend_time = float(frontend_match.group(1)) if frontend_match else 0.0
    t2s_time = float(t2s_match.group(1)) if t2s_match else 0.0
    vits_time = float(vits_match.group(1)) if vits_match else 0.0
    total_time = float(total_match.group(1)) if total_match else 0.0
    
    print(f"Benchmark Results ({lang}):")
    print(f"  Frontend: {frontend_time:.2f}ms")
    print(f"  T2S:      {t2s_time:.2f}ms")
    print(f"  VITS:     {vits_time:.2f}ms")
    print(f"  Total:    {total_time:.2f}ms")
    
    # Save to CSV
    # Check if header is needed (file exists and has content)
    file_exists = BENCHMARK_RESULTS_FILE.exists() and BENCHMARK_RESULTS_FILE.stat().st_size > 0
    
    with open(BENCHMARK_RESULTS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Lang", "Frontend_ms", "T2S_ms", "VITS_ms", "Total_ms", "Notes"])
        
        writer.writerow([
            datetime.now().isoformat(),
            lang,
            f"{frontend_time:.2f}",
            f"{t2s_time:.2f}",
            f"{vits_time:.2f}",
            f"{total_time:.2f}",
            "Optimization Step"
        ])
    print(f"Results saved to {BENCHMARK_RESULTS_FILE}")

if __name__ == "__main__":
    run_benchmark('en')
    run_benchmark('zh')

