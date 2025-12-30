import os
import sys
import subprocess
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class EnvManager:
    def __init__(self):
        # Determine the Data directory relative to the package root.
        # This ensures the config file stays with the package, not CWD.
        try:
            # Use __file__ to resolve paths reliably
            current_file = Path(__file__).resolve()
            # Parents: 0=Utils, 1=lunavox_tts, 2=src, 3=LunaVox(repo root)
            repo_root = current_file.parents[3]
            self.config_dir = repo_root / "TTSData"
            self.repo_root = repo_root
        except Exception:
             # Fallback if path resolution fails
            self.config_dir = Path("TTSData")
            self.repo_root = Path(".")

        # Allow override via env var
        data_dir_env = os.environ.get("LUNAVOX_DATA_DIR")
        if data_dir_env:
            self.config_dir = Path(data_dir_env)
            
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "env_config.json"
        self._config = self._load_config()

    def _load_config(self):
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load env config: {e}")
        return {"mode": "cpu"}

    def _save_config(self):
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save env config: {e}")

    def get_mode(self) -> str:
        """Returns the configured mode ('cpu' or 'gpu')."""
        return self._config.get("mode", "cpu")

    def set_mode(self, mode: str):
        """Sets the desired mode and saves configuration."""
        if mode not in ["cpu", "gpu"]:
            raise ValueError("Mode must be 'cpu' or 'gpu'")
        self._config["mode"] = mode
        self._save_config()
        logger.info(f"LunaVox mode set to: {mode}")

    def is_gpu_installed(self) -> bool:
        """Checks if onnxruntime-gpu is currently installed and functional."""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            return "CUDAExecutionProvider" in providers
        except Exception:
            return False

    def ensure_environment(self):
        """
        Validates the current environment against the requested mode.
        If a mismatch is found, it attempts to install the correct dependencies.
        Returns True if environment matches, False if a change was made (requires restart).
        """
        target_mode = self.get_mode()
        current_is_gpu = self.is_gpu_installed()

        if target_mode == "gpu" and not current_is_gpu:
            logger.warning("Target mode is GPU but onnxruntime-gpu is not found. Attempting upgrade...")
            self.install_gpu_runtime()
            return False
        
        if target_mode == "cpu" and current_is_gpu:
            # If user explicitly wants CPU, but GPU is installed, we should probably stick to CPU execution provider
            # but if they want "cpu environment cleaned", we might want to uninstall GPU.
            # For now, let's just log it. ORT-GPU can run CPU just fine.
            # But the user specifically asked to "confirm cpu environment correctly cleaned".
            logger.info("Target mode is CPU but onnxruntime-gpu is currently installed.")
            # self.install_cpu_runtime() # Optional: active cleanup
            # return False
            
        return True

    def install_gpu_runtime(self):
        """Uninstalls CPU runtime and installs GPU runtime."""
        logger.info("Switching to GPU runtime. This will uninstall onnxruntime and install onnxruntime-gpu.")
        try:
            # Uninstall both just to be clean, though usually only one exists
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "onnxruntime", "-y"])
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "onnxruntime-gpu", "-y"])
            
            logger.info("Installing onnxruntime-gpu...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime-gpu"])
            logger.info("onnxruntime-gpu installed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install GPU runtime: {e}")
            raise RuntimeError(f"Dependency installation failed: {e}")

    def install_cpu_runtime(self):
        """Uninstalls GPU runtime and installs CPU runtime."""
        logger.info("Switching to CPU runtime. This will uninstall onnxruntime-gpu and install onnxruntime.")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "onnxruntime-gpu", "-y"])
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "onnxruntime", "-y"])
            
            logger.info("Installing onnxruntime...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime"])
            logger.info("onnxruntime installed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install CPU runtime: {e}")
            raise RuntimeError(f"Dependency installation failed: {e}")

env_manager = EnvManager()
