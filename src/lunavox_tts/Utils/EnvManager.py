import os
import sys
import logging
import json
from pathlib import Path
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class EnvironmentStatus(Enum):
    """Runtime environment detection status."""
    CPU_ONLY = "cpu_only"           # Only onnxruntime installed (no GPU support)
    GPU_READY = "gpu_ready"         # onnxruntime-gpu installed and CUDA available
    GPU_DEPS_MISSING = "gpu_deps_missing"  # GPU package installed but CUDA DLLs missing



class EnvManager:
    def __init__(self):
        # Determine the Data directory relative to the package root.
        # This ensures the config file stays with the package, not CWD.
        try:
            # Use __file__ to resolve paths reliably
            current_file = Path(__file__).resolve()
            # Parents: 0=Utils, 1=lunavox_tts, 2=src, 3=LunaVox(repo root)
            repo_root = current_file.parents[3]
            self.repo_root = repo_root
            self.data_root = repo_root / "lunavoxData"
            self.config_dir = self.data_root / "TTSData"
        except Exception:
             # Fallback if path resolution fails
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
            self._setup_portable_cuda_paths()

    def _setup_portable_cuda_paths(self):
        """
        Search for portable CUDA DLLs in the current Python environment's site-packages
        (installed via nvidia-*-cu12 pip packages) and add them to the DLL search path.
        This is critical for Windows users who don't have a system-wide CUDA Toolkit installed.
        """
        if sys.platform != "win32":
            return

        try:
            import site
            # Aggressively find all possible site-packages locations
            search_paths = site.getsitepackages()
            if hasattr(site, 'getusersitepackages'):
                search_paths.append(site.getusersitepackages())
            
            # Add current sys.path entries that look like site-packages
            for p in sys.path:
                if "site-packages" in p and p not in search_paths:
                    search_paths.append(p)
            
            added_paths = []
            for sp_str in search_paths:
                sp = Path(sp_str)
                nvidia_base = sp / "nvidia"
                if not nvidia_base.exists():
                    continue
                
                # Find all 'bin' directories under nvidia base
                for bin_folder in nvidia_base.glob("**/bin"):
                    if bin_folder.is_dir():
                        bin_path_str = str(bin_folder.absolute())
                        if bin_path_str not in added_paths:
                            os.add_dll_directory(bin_path_str)
                            # Also add to PATH for some older or stubborn loaders
                            os.environ["PATH"] = bin_path_str + os.pathsep + os.environ["PATH"]
                            added_paths.append(bin_path_str)
            
            if added_paths:
                logger.info(f"Added portable CUDA DLL paths to search path: {len(added_paths)} paths found.")
                for p in added_paths:
                    logger.debug(f"  - {p}")
        except Exception as e:
            logger.warning(f"Failed to setup portable CUDA paths: {e}")
            pass

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
        val = self._config.get("developer_mode", False)
        if val:
            # Check for optional dependencies required for full dev mode experience
            try:
                import psutil
            except ImportError:
                # Log a warning only once per session ideally, but here works too as it's a getter
                # But to avoid spamming logs, we might want to be careful.
                # However, this method is called frequently. Let's move the check to set_developer_mode or init.
                pass
        return val

    def set_developer_mode(self, enabled: bool):
        """Sets the developer mode and saves configuration."""
        if enabled:
            self._ensure_developer_dependencies()
        
        self._config["developer_mode"] = enabled
        self._save_config()
        logger.info(f"Developer mode set to: {enabled}")

    def _print_gpu_instruction(self):
        """Prints a detailed, pretty English instruction for GPU setup."""
        print("\n" + "=" * 70)
        print("  LUNAVOX - GPU Acceleration Not Found")
        print("=" * 70)
        print("\nYou have requested GPU mode, but the GPU runtime is not installed")
        print("or its CUDA dependencies are missing.")
        print("\nTo enable high-performance GPU inference, please run:")
        
        if sys.platform == "win32":
            print(f"\n  > scripts\\setup_gpu.bat")
        else:
            print(f"\n  $ bash scripts/setup_gpu.sh")
            
        print("\nThis script will:")
        print("  1. Uninstall the standard 'onnxruntime' (CPU)")
        print("  2. Install 'onnxruntime-gpu' (CUDA 12)")
        print("  3. Download portable CUDA/cuDNN DLLs (~600MB)")
        print("\nNote: You need an NVIDIA GPU with compatible drivers.")
        print("-" * 70 + "\n")

    def _ensure_developer_dependencies(self) -> bool:
        """
        Checks for optional developer dependencies and prompts for installation if missing.
        Returns True if all dependencies are available, False otherwise.
        """
        deps = []
        try:
            import psutil
        except ImportError:
            deps.append(("psutil", "RAM tracking"))
            
        try:
            import pynvml
        except ImportError:
            # nvidia-ml-py is the package name for pynvml
            deps.append(("nvidia-ml-py", "VRAM tracking"))
            
        if not deps:
            return True

        # Clear, visual English prompt with detailed explanation
        print("\n" + "=" * 70)
        print("  DEVELOPER MODE - Missing Dependencies")
        print("=" * 70)
        print("\nThe following packages are required for performance monitoring:\n")
        for pkg_name, purpose in deps:
            print(f"  • {pkg_name:<20} → {purpose}")
        print("\nThese enable accurate RAM/VRAM metrics during benchmarking.")
        print("Without them, memory usage will show as 0 in reports.")
        print("\n" + "-" * 70)
        
        try:
            # We use a raw input here to prompt the user in the terminal
            # Note: This might block in non-interactive environments
            choice = input(f"\nWould you like to install {len(deps)} missing developer dependencies now? (y/n): ").strip().lower()
            if choice == 'y':
                print("\nInstalling dependencies...")
                for pkg_name, _ in deps:
                    print(f"  → Installing {pkg_name}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name], 
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("\n✓ Successfully installed developer dependencies.")
                print("  Please restart the script to enable monitoring features.\n")
                print("=" * 70 + "\n")
                return True
            else:
                print("\n⚠ Skipping installation. Memory monitoring will be unavailable.")
                print("=" * 70 + "\n")
                return False
        except EOFError:
            # Handle non-interactive environments
            logger.warning("Non-interactive environment detected. Skipping dependency prompt.")
            return False
        except Exception as e:
            logger.error(f"Failed to install developer dependencies: {e}")
            return False

    def set_mode(self, mode: str):
        """Sets the desired mode and saves configuration."""
        if mode not in ["cpu", "gpu"]:
            raise ValueError("Mode must be 'cpu' or 'gpu'")
        
        # Check for immediate mismatch and provide guidance
        if mode == "gpu" and self.get_environment_status() == EnvironmentStatus.CPU_ONLY:
             self._print_gpu_instruction()
             logger.warning("GPU mode set, but environment is CPU-only. Please run setup_gpu script to fix.")
             
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
            
            # Additional check: attempt to create a dummy session to verify DLLs
            # We use a tiny constant or empty bytes if possible, but 
            # the most reliable way is often to check if we can initialize the provider.
            # Here we just check the version as well.
            target_ver = "1.22.0"
            if ort.__version__ != target_ver:
                logger.info(f"Current onnxruntime version {ort.__version__} mismatches target {target_ver}.")
                return False

            return True
        except Exception:
            return False

    def get_environment_status(self) -> EnvironmentStatus:
        """
        Detect the current runtime environment status.
        
        Returns:
            EnvironmentStatus.CPU_ONLY: Only CPU runtime available
            EnvironmentStatus.GPU_READY: GPU runtime ready
            EnvironmentStatus.GPU_DEPS_MISSING: GPU package installed but CUDA unavailable
        """
        if self._cached_env_status is not None:
            return self._cached_env_status
        
        try:
            import onnxruntime as ort
            available_providers = set(ort.get_available_providers())
            
            if "CUDAExecutionProvider" in available_providers:
                # Test if CUDA is actually functional
                self._cached_env_status = EnvironmentStatus.GPU_READY
                return self._cached_env_status
            
            # Check if onnxruntime-gpu is installed but CUDA not working
            # This happens when GPU package is installed but CUDA DLLs are missing
            try:
                # Check package name via distribution info
                from importlib.metadata import distribution
                dist = distribution('onnxruntime-gpu')
                if dist:
                    # GPU package installed but CUDA provider not available
                    self._cached_env_status = EnvironmentStatus.GPU_DEPS_MISSING
                    return self._cached_env_status
            except Exception:
                pass
            
            # Default: CPU only
            self._cached_env_status = EnvironmentStatus.CPU_ONLY
            return self._cached_env_status
            
        except ImportError:
            self._cached_env_status = EnvironmentStatus.CPU_ONLY
            return self._cached_env_status

    def invalidate_cache(self) -> None:
        """Clear cached environment status (call after environment changes)."""
        self._cached_env_status = None

    def temporary_mode(self, mode: str):
        """
        Context manager to temporarily override the mode without saving to disk.
        Example:
            with env_manager.temporary_mode("cpu"):
                # run task in cpu mode
        """
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
        """
        Validates the current environment against the requested mode.
        
        Returns:
            True if environment matches the requested mode.
            
        Raises:
            EnvironmentMismatchError: If mode is 'gpu' but environment is CPU_ONLY.
        """
        target_mode = self.get_mode()
        status = self.get_environment_status()

        if target_mode == "gpu":
            if status == EnvironmentStatus.CPU_ONLY:
                # Print detailed instruction before raising
                self._print_gpu_instruction()
                
                from ..Core.Model.ExecutionPolicy import EnvironmentMismatchError
                raise EnvironmentMismatchError(
                    "GPU mode requested but only CPU runtime is installed.\n"
                    "Please run scripts/setup_gpu to install the GPU acceleration package."
                )
            elif status == EnvironmentStatus.GPU_DEPS_MISSING:
                logger.warning(
                    "GPU package installed but CUDA dependencies are missing. "
                    "Falling back to CPU execution. Run setup_gpu script to fix."
                )
        
        return True


env_manager = EnvManager()
