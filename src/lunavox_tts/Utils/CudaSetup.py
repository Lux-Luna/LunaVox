"""
CudaSetup - Windows CUDA DLL path configuration.

Extracted from EnvManager to improve modularity.
"""
import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_portable_cuda_paths() -> list[str]:
    """
    Search for portable CUDA DLLs in the current Python environment's site-packages
    (installed via nvidia-*-cu12 pip packages) and add them to the DLL search path.
    
    This is critical for Windows users who don't have a system-wide CUDA Toolkit installed.
    
    Returns:
        List of paths that were added to the DLL search path.
    """
    if sys.platform != "win32":
        return []

    added_paths = []
    
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
            logger.info(f"Added portable CUDA DLL paths: {len(added_paths)} paths found.")
            for p in added_paths:
                logger.debug(f"  - {p}")
                
    except Exception as e:
        logger.warning(f"Failed to setup portable CUDA paths: {e}")
    
    return added_paths
