import time
import logging
import os
from contextlib import contextmanager
from typing import Optional, Any
from .EnvManager import env_manager

logger = logging.getLogger(__name__)

# Optional dependency
try:
    import psutil
except ImportError:
    psutil = None

class PerformanceMonitor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PerformanceMonitor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Initialized once due to singleton pattern check
        self.process = None
        if psutil:
             self.process = psutil.Process(os.getpid())

    @property
    def is_enabled(self) -> bool:
        # If enabled but psutil missing, warn once and maybe auto-disable or just skip mem stats?
        # We'll just skip mem stats if psutil is missing, but developer_mode is enabled.
        enabled = env_manager.get_developer_mode()
        if enabled and psutil is None:
             # Just a one-time warning could be annoying in loop, maybe log once at init?
             # For now, we handle it in measure()
             pass
        return enabled

    @contextmanager
    def measure(self, task_name: str):
        """
        Context manager to measure execution time and memory usage of a block.
        Only logs if developer mode is enabled.
        """
        if not self.is_enabled:
            yield
            return

        # Check dependency if not already loaded (though we try import at top)
        if psutil is None:
             # Warning is handled in EnvManager when enabling developer mode
             pass
        
        start_time = time.perf_counter()
        start_mem = 0
        if self.process:
             start_mem = self.process.memory_info().rss

        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            mem_str = ""
            if self.process:
                end_mem = self.process.memory_info().rss
                mem_delta_mb = (end_mem - start_mem) / (1024 * 1024)
                # Format memory string based on sign
                mem_str = f" | RSS: {mem_delta_mb:+.2f}MB" if abs(mem_delta_mb) > 0.1 else " | RSS: unchanged"
            
            logger.info(f"[Perf] {task_name} took: {duration_ms:.2f}ms{mem_str}")

    def log_data(self, name: str, data: Any, level: int = logging.DEBUG):
        """
        Log detailed data information (shape, stats) if developer mode is enabled.
        """
        if not self.is_enabled:
            return

        # Lazy import numpy to avoid overhead if not needed globally (though likely already imported)
        import numpy as np
        
        if isinstance(data, np.ndarray):
            # Calculate stats safely
            try:
                min_val = float(np.min(data))
                max_val = float(np.max(data))
                mean_val = float(np.mean(data))
                info = f"shape={data.shape}, dtype={data.dtype}, range=[{min_val:.4f}, {max_val:.4f}], mean={mean_val:.4f}"
            except Exception:
                 info = f"shape={data.shape}, dtype={data.dtype} (stats failed)"
            
            logger.log(level, f"[Data] {name}: {info}")
        else:
             logger.log(level, f"[Data] {name}: {data}")

    def log_metric(self, name: str, value: Any, unit: str = ""):
        """
        Log specific metric value if developer mode is enabled.
        """
        if not self.is_enabled:
            return
        
        logger.info(f"[Metric] {name}: {value}{unit}")

monitor = PerformanceMonitor()

