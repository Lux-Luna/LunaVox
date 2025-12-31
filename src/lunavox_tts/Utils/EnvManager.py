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
        self._mode_override: Optional[str] = None
        
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

    def _ensure_developer_dependencies(self):
        """Checks for optional developer dependencies and prompts for installation if missing."""
        deps = []
        try:
            import psutil
        except ImportError:
            deps.append(("psutil", "psutil"))
            
        try:
            import pynvml
        except ImportError:
            # nvidia-ml-py is the package name for pynvml
            deps.append(("nvidia-ml-py", "pynvml"))
            
        if not deps:
            return

        logger.info("\n" + "!" * 60)
        logger.info("[DEVELOPER MODE] Required monitoring dependencies are missing.")
        logger.info(f"Missing: {', '.join([d[0] for d in deps])}")
        logger.info("These are required for RAM/VRAM tracking and detailed performance metrics.")
        
        try:
            # We use a raw input here to prompt the user in the terminal
            # Note: This might block in non-interactive environments
            choice = input(f"\nWould you like to install {len(deps)} missing developer dependencies now? (y/n): ").strip().lower()
            if choice == 'y':
                for pkg_name, _ in deps:
                    logger.info(f"Installing {pkg_name}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
                logger.info("Successfully installed developer dependencies.\n")
            else:
                logger.warning("\nSkipping installation. Some monitoring features will be unavailable.")
        except Exception as e:
            logger.error(f"Failed to handle developer dependency installation: {e}")
        logger.info("!" * 60 + "\n")

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

    def ensure_environment(self):
        """
        Validates the current environment against the requested mode.
        If a mismatch is found, it attempts to install the correct dependencies.
        Returns True if environment matches, False if a change was made (requires restart).
        """
        target_mode = self.get_mode()
        current_is_gpu = self.is_gpu_installed()

        if target_mode == "gpu" and not current_is_gpu:
            logger.warning("Target mode is GPU but onnxruntime-gpu or CUDA dependencies are not found.")
            
            # Terminal prompt for user confirmation (English)
            print("\n" + "="*60)
            print("GPU MODE DETECTED")
            print("="*60)
            print("The following dependencies are required for GPU acceleration but are missing:")
            print("  - onnxruntime-gpu==1.22.0")
            print("  - nvidia-*-cu12 (CUDA 12 Runtime Libraries)")
            print("\nThis process will uninstall the current CPU-only 'onnxruntime' if it exists.")
            
            try:
                choice = input("\nWould you like to install the required GPU dependencies now? (y/n): ").strip().lower()
                if choice == 'y':
                    self.install_gpu_runtime()
                    return False
                else:
                    print("\nSkipping GPU dependency installation. The application may run on CPU or fail if GPU is forced.")
                    return True # Let it proceed, though it might fail later
            except EOFError:
                # Handle non-interactive environments by defaulting to NO to avoid hanging
                logger.error("Non-interactive environment detected. Automatic GPU installation skipped.")
                return True
        
        if target_mode == "cpu" and current_is_gpu:
            logger.info("Target mode is CPU but onnxruntime-gpu is currently installed. Switching back to CPU runtime...")
            self.install_cpu_runtime()
            return False
            
        return True

    def install_gpu_runtime(self):
        """Uninstalls CPU runtime and installs GPU runtime with portable CUDA 12 dependencies."""
        logger.info("Switching to GPU runtime. This will uninstall onnxruntime and install onnxruntime-gpu with CUDA 12 libraries.")
        try:
            # Uninstall both just to be clean, though usually only one exists
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "onnxruntime", "-y"])
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "onnxruntime-gpu", "-y"])
            
            # Using 1.22.0 as requested
            logger.info("Installing onnxruntime-gpu==1.22.0 and full CUDA 12 runtime libraries...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "onnxruntime-gpu==1.22.0", 
                "nvidia-cudnn-cu12", 
                "nvidia-cublas-cu12", 
                "nvidia-cuda-runtime-cu12",
                "nvidia-cuda-nvrtc-cu12",
                "nvidia-cufft-cu12",
                "nvidia-curand-cu12",
                "nvidia-cusolver-cu12",
                "nvidia-cusparse-cu12",
                "numpy<2"
            ])
            
            logger.info("onnxruntime-gpu and portable CUDA libraries installed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install GPU runtime: {e}")
            raise RuntimeError(f"Dependency installation failed: {e}")

    def install_cpu_runtime(self):
        """Uninstalls GPU runtime and installs CPU runtime."""
        logger.info("Switching to CPU runtime. This will uninstall onnxruntime-gpu and install onnxruntime.")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "onnxruntime-gpu", "-y"])
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "onnxruntime", "-y"])
            
            # Explicitly lock to 1.22.1 for optimized CPU performance
            logger.info("Installing onnxruntime==1.22.1...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "onnxruntime==1.22.1",
                "numpy<2"
            ])
            logger.info("onnxruntime installed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install CPU runtime: {e}")
            raise RuntimeError(f"Dependency installation failed: {e}")

env_manager = EnvManager()
