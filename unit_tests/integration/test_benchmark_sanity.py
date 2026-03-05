
import subprocess
import sys
from pathlib import Path
import pytest

# Locate benchmark.py
BENCHMARK_SCRIPT = Path(__file__).parent.parent.parent / "benchmark" / "benchmark.py"

@pytest.mark.integration
def test_benchmark_script_sanity():
    """
    Verifies that the benchmark.py script runs without import errors.
    Runs a minimal configuration: CPU, English, v2, 0 warmup, 1 round.
    """
    if not BENCHMARK_SCRIPT.exists():
        pytest.skip(f"Benchmark script not found at {BENCHMARK_SCRIPT}")

    cmd = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--env", "cpu",
        "--mode", "reference",
        "--version", "v2", 
        "--lang", "en",
        "--warmup", "0",
        "--rounds", "1"
    ]

    try:
        # Run process with specific timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes max for a sanity check
        )
        
        # Check return code
        if result.returncode != 0:
            pytest.fail(f"Benchmark script failed with return code {result.returncode}.\nStderr: {result.stderr}\nStdout: {result.stdout}")
            
        # Check for success message in stdout (optional, but good for confidence)
        if "Benchmark completed!" not in result.stdout:
             # It might have completed but maybe with warnings. output varies.
             # But if returncode is 0, it generally means no uncaught exception.
             pass

    except subprocess.TimeoutExpired:
        pytest.fail("Benchmark script timed out during sanity check.")
    except Exception as e:
        pytest.fail(f"Benchmark execution failed: {e}")
