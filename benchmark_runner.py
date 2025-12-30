import time
import psutil
import subprocess
import os
import sys
import logging
from statistics import mean

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Benchmark")

def run_benchmark(iterations=5, output_file="benchmark_baseline.txt"):
    script_path = os.path.join("Tutorial", "v2_quick_tryout", "quick_tryout_en.py")
    
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        return

    latencies = []
    peak_memories = []

    logger.info(f"Starting benchmark: {iterations} iterations of {script_path}")

    for i in range(iterations):
        logger.info(f"Running iteration {i+1}/{iterations}...")
        
        start_time = time.time()
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Monitor memory usage
        max_mem = 0
        try:
            p = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    mem_info = p.memory_info()
                    max_mem = max(max_mem, mem_info.rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                time.sleep(0.1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        stdout, stderr = process.communicate()
        end_time = time.time()
        
        duration = end_time - start_time
        latencies.append(duration)
        peak_memories.append(max_mem / (1024 * 1024)) # MB

        logger.info(f"Iteration {i+1}: Time={duration:.2f}s, Peak Memory={max_mem / (1024 * 1024):.2f} MB")
        
        if process.returncode != 0:
            logger.error(f"Iteration {i+1} failed with return code {process.returncode}")
            logger.error(f"Stderr: {stderr}")

    avg_latency = mean(latencies)
    avg_memory = mean(peak_memories)

    result_text = (
        f"Benchmark Results ({iterations} iterations)\n"
        f"----------------------------------------\n"
        f"Script: {script_path}\n"
        f"Average Latency: {avg_latency:.2f} seconds\n"
        f"Average Peak Memory: {avg_memory:.2f} MB\n"
        f"Latencies: {[f'{x:.2f}' for x in latencies]}\n"
        f"Memories: {[f'{x:.2f}' for x in peak_memories]}\n"
    )

    print("\n" + result_text)
    
    with open(output_file, "w") as f:
        f.write(result_text)

if __name__ == "__main__":
    # Ensure we are in the LunaVox root directory
    if os.path.basename(os.getcwd()) != "LunaVox" and os.path.exists("LunaVox"):
        os.chdir("LunaVox")
        
    run_benchmark(iterations=5, output_file="benchmark_final.txt")

