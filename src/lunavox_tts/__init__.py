import os
import sys

if sys.platform == 'win32':
    # Add CUDA DLLs to path if installed via pip
    import site
    packages_dirs = site.getsitepackages()
    if hasattr(site, 'getusersitepackages'):
        packages_dirs.append(site.getusersitepackages())

    for packages_dir in packages_dirs:
        # CUDA 12 dlls from nvidia-* packages
        for component in ['cublas', 'cudnn', 'cuda_runtime', 'cuda_cupti', 'curand', 'cusolver', 'cusparse', 'nvjitlink', 'cufft']:
            comp_path = os.path.join(packages_dir, "nvidia", component, "bin")
            if os.path.exists(comp_path):
                os.add_dll_directory(comp_path)
                # Also add to PATH for some older loading mechanisms
                if comp_path not in os.environ['PATH']:
                    os.environ['PATH'] = comp_path + os.pathsep + os.environ['PATH']

# Modern modular API
from .API import (
    load_character, 
    unload_character, 
    set_reference_audio, 
    tts_async, 
    tts, 
    stop, 
    convert_to_onnx,
    clear_reference_audio_cache, 
    launch_command_line_client,
    create_persona, 
    load_persona,
    initialize_tts
)

# For multi-reference support
from .API.synthesis import set_multi_reference_audio, create_multi_reference_audio

from .Interface.Server import start_server

__all__ = [
    "load_character",
    "unload_character",
    "set_reference_audio",
    "set_multi_reference_audio",
    "create_multi_reference_audio",
    "tts_async",
    "tts",
    "stop",
    "convert_to_onnx",
    "clear_reference_audio_cache",
    "launch_command_line_client",
    "start_server",
    "create_persona",
    "load_persona",
    "initialize_tts",
]
