import os
import sys
import logging
import json
from pathlib import Path
from enum import Enum
from typing import Optional

from .CudaSetup import setup_portable_cuda_paths
from .Diagnostics import print_gpu_instruction, ensure_developer_dependencies

logger = logging.getLogger(__name__)


class EnvironmentStatus(Enum):
    """Runtime environment detection status."""
    CPU_ONLY = "cpu_only"           # Only onnxruntime installed (no GPU support)
    GPU_READY = "gpu_ready"         # onnxruntime-gpu installed and CUDA available
    GPU_DEPS_MISSING = "gpu_deps_missing"  # GPU package installed but CUDA DLLs missing


class EnvManager:
    def __init__(self):
        # Determine the Data directory relative to the package root.
        try:
            current_file = Path(__file__).resolve()
            repo_root = current_file.parents[3]
            self.repo_root = repo_root
            self.data_root = repo_root / "lunavoxData"
            self.config_dir = self.data_root / "TTSData"
        except Exception:
            self.repo_root = Path(".")
            self.data_root = self.repo_root / "lunavoxData"
            self.config_dir = self.data_root / "TTSData"

        # Allow override via env var
        data_dir_env = os.environ.get("LUNAVOX_DATA_DIR")
        if data_dir_env:
            self.config_dir = Path(data_dir_env)
            
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "env_config.json"
        self._config = self._load_config()
        self._mode_override: Optional[str] = None
        self._cached_env_status: Optional[EnvironmentStatus] = None
        
        # Setup portable CUDA paths if on Windows and GPU mode is active
        if sys.platform == "win32" and self.get_mode() == "gpu":
            setup_portable_cuda_paths()

    def _load_config(self):
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load env config: {e}")
        return {"mode": "cpu", "developer_mode": False}

    def _save_config(self):
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save env config: {e}")

    def get_mode(self) -> str:
        """Returns the configured mode ('cpu' or 'gpu'), respecting overrides."""
        if self._mode_override:
            return self._mode_override
        return self._config.get("mode", "cpu")

    def get_developer_mode(self) -> bool:
        """Returns True if developer mode is enabled."""
        return self._config.get("developer_mode", False)

    def set_developer_mode(self, enabled: bool):
        """Sets the developer mode and saves configuration."""
        if enabled:
            ensure_developer_dependencies()
        
        self._config["developer_mode"] = enabled
        self._save_config()
        logger.info(f"Developer mode set to: {enabled}")

    def set_mode(self, mode: str):
        """Sets the desired mode and saves configuration."""
        if mode not in ["cpu", "gpu"]:
            raise ValueError("Mode must be 'cpu' or 'gpu'")
        
        if mode == "gpu" and self.get_environment_status() == EnvironmentStatus.CPU_ONLY:
            print_gpu_instruction()
            logger.warning("GPU mode set, but environment is CPU-only.")
             
        self._config["mode"] = mode
        self._save_config()
        logger.info(f"LunaVox mode set to: {mode}")

    def is_gpu_installed(self) -> bool:
        """Checks if onnxruntime-gpu is currently installed and functional."""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" not in providers:
                return False
            return True
        except Exception:
            return False

    def get_environment_status(self) -> EnvironmentStatus:
        """Detect the current runtime environment status."""
        if self._cached_env_status is not None:
            return self._cached_env_status
        
        try:
            import onnxruntime as ort
            available_providers = set(ort.get_available_providers())
            
            if "CUDAExecutionProvider" in available_providers:
                self._cached_env_status = EnvironmentStatus.GPU_READY
                return self._cached_env_status
            
            try:
                from importlib.metadata import distribution
                dist = distribution('onnxruntime-gpu')
                if dist:
                    self._cached_env_status = EnvironmentStatus.GPU_DEPS_MISSING
                    return self._cached_env_status
            except Exception:
                pass
            
            self._cached_env_status = EnvironmentStatus.CPU_ONLY
            return self._cached_env_status
            
        except ImportError:
            self._cached_env_status = EnvironmentStatus.CPU_ONLY
            return self._cached_env_status

    def invalidate_cache(self) -> None:
        """Clear cached environment status."""
        self._cached_env_status = None

    def temporary_mode(self, mode: str):
        """Context manager to temporarily override mode without saving."""
        from contextlib import contextmanager

        @contextmanager
        def _override():
            old_override = self._mode_override
            self._mode_override = mode
            try:
                yield
            finally:
                self._mode_override = old_override
        
        return _override()

    def ensure_environment(self) -> bool:
        """Validates environment against requested mode."""
        target_mode = self.get_mode()
        status = self.get_environment_status()

        if target_mode == "gpu":
            if status == EnvironmentStatus.CPU_ONLY:
                print_gpu_instruction()
                from ..Core.Model.ExecutionPolicy import EnvironmentMismatchError
                raise EnvironmentMismatchError(
                    "GPU mode requested but only CPU runtime is installed."
                )
            elif status == EnvironmentStatus.GPU_DEPS_MISSING:
                logger.warning("GPU package installed but CUDA deps missing.")
        
        return True


env_manager = EnvManager()
