"""
Diagnostics - Interactive prompts and setup guidance.

Extracted from EnvManager to improve modularity.
Contains user-facing messages for GPU setup and developer dependencies.
"""
import sys
import subprocess
import logging

logger = logging.getLogger(__name__)


def print_gpu_instruction() -> None:
    """Print detailed, pretty instruction for GPU setup."""
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


def ensure_developer_dependencies() -> bool:
    """
    Check for optional developer dependencies and prompt for installation if missing.
    
    Returns:
        True if all dependencies are available, False otherwise.
    """
    deps = []
    
    try:
        import psutil
    except ImportError:
        deps.append(("psutil", "RAM tracking"))
        
    try:
        import pynvml
    except ImportError:
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
        choice = input(f"\nWould you like to install {len(deps)} missing developer dependencies now? (y/n): ").strip().lower()
        if choice == 'y':
            print("\nInstalling dependencies...")
            for pkg_name, _ in deps:
                print(f"  → Installing {pkg_name}...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg_name],
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
            print("\n✓ Successfully installed developer dependencies.")
            print("  Please restart the script to enable monitoring features.\n")
            print("=" * 70 + "\n")
            return True
        else:
            print("\n⚠ Skipping installation. Memory monitoring will be unavailable.")
            print("=" * 70 + "\n")
            return False
    except EOFError:
        logger.warning("Non-interactive environment detected. Skipping dependency prompt.")
        return False
    except Exception as e:
        logger.error(f"Failed to install developer dependencies: {e}")
        return False
