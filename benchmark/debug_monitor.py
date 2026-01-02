import os
import sys
import time
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lunavox_tts.Utils.PerformanceMonitor import monitor
from lunavox_tts.Utils.EnvManager import env_manager

def test_monitor():
    env_manager.set_mode("gpu")
    env_manager.set_developer_mode(True)
    
    print("Initial state...")
    monitor._ensure_baselines()
    print(f"Base RSS: {monitor.base_rss / 1024 / 1024:.2f} MB")
    print(f"Base VRAM: {monitor.base_vram / 1024 / 1024:.2f} MB")
    
    with monitor.measure("Test Task", category="USER_PERCEIVED"):
        print("Running dummy task...")
        # Simulate some work and potential allocation
        time.sleep(0.5)
        
    buffer = monitor.get_buffer()
    # If not buffering, it prints to logger. Let's enable buffering.
    monitor.set_buffering(True)
    
    with monitor.measure("Test Task Buffered", category="USER_PERCEIVED"):
        time.sleep(0.5)
        
    data = monitor.get_buffer()
    print("Results:")
    for entry in data:
        print(entry)

if __name__ == "__main__":
    test_monitor()
