import time
import logging
import os
import sys
from contextlib import contextmanager
from typing import Optional, Any, Literal
from .EnvManager import env_manager

logger = logging.getLogger(__name__)

# Lazy initialization of dependencies to avoid any overhead at startup
_psutil = None
_pynvml = None
_gpu_handle = None
_gpu_initialized = False

def _init_gpu():
    global _pynvml, _gpu_handle, _gpu_initialized
    if _gpu_initialized:
        return
    try:
        # nvidia-ml-py is compatible with pynvml import name
        import pynvml
        pynvml.nvmlInit()
        _pynvml = pynvml
        _gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except:
        _pynvml = None
        _gpu_handle = None
    _gpu_initialized = True

def _get_psutil():
    global _psutil
    if _psutil is None:
        try:
            import psutil
            _psutil = psutil
        except ImportError:
            logger.warning("[Monitor] 'psutil' is missing. RAM tracking will be unavailable. 'pip install psutil' to enable.")
    return _psutil

class PerformanceMonitor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PerformanceMonitor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self.process = None
        # We record baselines at the start of the FIRST measurement to capture "idle" state
        self.base_rss = 0
        self.base_vram = 0
        self._baselines_set = False
        
        self._buffer = []
        self._buffering_enabled = False
        self._initialized = True

    def _ensure_baselines(self):
        """Capture background noise once to filter it from future measurements."""
        if self._baselines_set:
            return
        
        ps = _get_psutil()
        if ps:
            self.process = ps.Process(os.getpid())
            try:
                self.base_rss = self.process.memory_info().rss
            except: pass
        
        _init_gpu()
        if _gpu_handle:
            try:
                self.base_vram = _pynvml.nvmlDeviceGetMemoryInfo(_gpu_handle).used
            except: pass
        
        self._baselines_set = True

    @property
    def is_enabled(self) -> bool:
        return env_manager.get_developer_mode()

    @contextmanager
    def measure(self, task_name: str, category: Literal["USER_PERCEIVED", "LINK_DETAIL"] = "LINK_DETAIL"):
        """
        High-performance context manager for measuring task duration and memory.
        Calculations are performed AFTER the task to avoid blocking the TTS chain.
        """
        if not self.is_enabled:
            yield
            return

        # Ensure baselines are set before any measurement (only happens once)
        self._ensure_baselines()
        
        # Start timing (extremely lightweight)
        start_time = time.perf_counter()

        try:
            yield
        finally:
            # End timing
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Post-task analysis: All heavy syscalls happen here, after the task is finished.
            # This ensures they don't affect the duration_ms or the TTS latency.
            rss_mb = 0
            if self.process:
                try:
                    # Report absolute RSS of the process for better visibility
                    current_rss = self.process.memory_info().rss
                    rss_mb = current_rss / (1024 * 1024)
                except: pass
            
            vram_mb = 0
            if _gpu_handle:
                try:
                    # Attempt precise PID-based VRAM tracking
                    pid = os.getpid()
                    used_vram = 0
                    pid_found = False
                    try:
                        procs = (_pynvml.nvmlDeviceGetComputeRunningProcesses(_gpu_handle) or []) + \
                                (_pynvml.nvmlDeviceGetGraphicsRunningProcesses(_gpu_handle) or [])
                        for p in procs:
                            if p.pid == pid:
                                used_vram = p.usedGpuMemory
                                pid_found = True
                                break
                    except: pass
                    
                    if pid_found:
                        vram_mb = used_vram / (1024 * 1024)
                    else:
                        # Fallback to absolute system VRAM usage if PID tracking fails
                        vram_mb = _pynvml.nvmlDeviceGetMemoryInfo(_gpu_handle).used / (1024 * 1024)
                except: pass
            
            if self._buffering_enabled:
                self._buffer.append({
                    "type": "perf",
                    "task": task_name,
                    "category": category,
                    "duration_ms": duration_ms,
                    "mem_rss_mb": rss_mb,
                    "vram_mb": vram_mb,
                    "timestamp": time.time()
                })
            else:
                logger.info(f"[Perf][{category}] {task_name} took: {duration_ms:.2f}ms | RAM+: {rss_mb:.2f}MB | VRAM+: {vram_mb:.2f}MB")

    def set_buffering(self, enabled: bool):
        """Enable or disable buffering of metrics."""
        self._buffering_enabled = enabled
        if not enabled:
            self._buffer = []

    def get_buffer(self) -> list:
        """Retrieve the current buffer and clear it."""
        data = self._buffer
        self._buffer = []
        return data

    def log_data(self, name: str, data: Any, level: int = logging.DEBUG):
        if not self.is_enabled:
            return
        import numpy as np
        if isinstance(data, np.ndarray):
            try:
                info = f"shape={data.shape}, dtype={data.dtype}, range=[{np.min(data):.4f}, {np.max(data):.4f}], mean={np.mean(data):.4f}"
            except:
                 info = f"shape={data.shape}, dtype={data.dtype} (stats failed)"
            logger.log(level, f"[Data] {name}: {info}")
        else:
             logger.log(level, f"[Data] {name}: {data}")

    def log_metric(self, name: str, value: Any, unit: str = ""):
        if not self.is_enabled:
            return
        if self._buffering_enabled:
            self._buffer.append({
                "type": "metric", "name": name, "value": value, "unit": unit, "timestamp": time.time()
            })
        else:
            logger.info(f"[Metric] {name}: {value}{unit}")

monitor = PerformanceMonitor()

