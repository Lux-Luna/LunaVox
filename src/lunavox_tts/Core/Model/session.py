"""
Model Session Configuration - ONNX Runtime session utilities.

Provides session options, provider resolution, and FP16 weight loading.
"""
import os
import logging
import onnxruntime
import gc
from typing import Optional, List
from onnxruntime import InferenceSession

logger = logging.getLogger(__name__)

def get_default_sess_options() -> onnxruntime.SessionOptions:
    opts = onnxruntime.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.add_session_config_entry("session.use_env_allocators", "1")
    return opts

_DEFAULT_PROVIDER_ORDER: list[str] = [
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "ROCMExecutionProvider",
    "CPUExecutionProvider",
]

def resolve_providers() -> list[str]:
    """
    Resolve ONNX Runtime execution providers based on configured mode and environment.
    
    Returns:
        List of available providers in priority order.
        
    Raises:
        EnvironmentMismatchError: If GPU mode is requested but environment is CPU-only.
    """
    from ...Utils.EnvManager import env_manager, EnvironmentStatus
    from .ExecutionPolicy import EnvironmentMismatchError
    
    target_mode = env_manager.get_mode()
    
    # CPU mode: always use CPU provider regardless of environment
    if target_mode == "cpu":
        logger.debug("LunaVox is running in CPU mode as configured.")
        return ["CPUExecutionProvider"]
    
    # GPU mode: check environment status
    status = env_manager.get_environment_status()
    
    if status == EnvironmentStatus.CPU_ONLY:
        raise EnvironmentMismatchError(
            "GPU mode requested but only CPU runtime is installed.\n"
            "To enable GPU acceleration, run: scripts/setup_gpu.bat (Windows) or scripts/setup_gpu.sh (Linux/Mac)\n"
            "This will install onnxruntime-gpu and required CUDA libraries (~600MB)."
        )
    
    if status == EnvironmentStatus.GPU_DEPS_MISSING:
        logger.warning(
            "GPU package installed but CUDA dependencies are missing. "
            "Falling back to CPU execution. Run setup_gpu script to fix."
        )
        return ["CPUExecutionProvider"]
    
    # GPU_READY: proceed with GPU provider resolution
    try:
        available = set(onnxruntime.get_available_providers())
    except Exception as e:
        logger.warning(f"Failed to get available providers: {e}")
        available = {"CPUExecutionProvider"}

    env_value = os.getenv("LUNAVOX_ORT_PROVIDERS")
    if env_value:
        requested = [item.strip() for item in env_value.split(",") if item.strip()]
        resolved = [provider for provider in requested if provider in available]
        if resolved:
            logger.debug("Using ONNXRuntime providers from LUNAVOX_ORT_PROVIDERS: %s", ",".join(resolved))
            return resolved
    
    resolved = [provider for provider in _DEFAULT_PROVIDER_ORDER if provider in available]
    if resolved:
        logger.debug("Auto-detected ONNXRuntime providers: %s", ",".join(resolved))
        return resolved
    
    return ["CPUExecutionProvider"]


def load_session_with_fp16_conversion(
    onnx_path: str,
    fp16_bin_path: str,
    providers: List[str],
    sess_options: Optional[onnxruntime.SessionOptions] = None
) -> InferenceSession:
    """
    Reads ONNX and FP16 weights, converts to FP32 in memory, 
    injects into ONNX model, and creates InferenceSession without temp files.
    """
    import onnx
    import numpy as np

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX Model not found: {onnx_path}")
    if not os.path.exists(fp16_bin_path):
        raise FileNotFoundError(f"FP16 Weight file not found: {fp16_bin_path}")

    model_proto = onnx.load(onnx_path, load_external_data=False)
    fp16_data = np.fromfile(fp16_bin_path, dtype=np.float16)
    fp32_data = fp16_data.astype(np.float32)
    del fp16_data
    fp32_bytes = fp32_data.tobytes()
    del fp32_data

    for tensor in model_proto.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            offset = 0
            length = 0
            for entry in tensor.external_data:
                if entry.key == 'offset':
                    offset = int(entry.value)
                elif entry.key == 'length':
                    length = int(entry.value)

            if offset + length > len(fp32_bytes):
                logger.warning(
                    f"Tensor {tensor.name} requested a range that exceeds the bin file size."
                )
                continue

            tensor.raw_data = fp32_bytes[offset: offset + length]
            tensor.data_type = onnx.TensorProto.FLOAT
            del tensor.external_data[:]
            tensor.data_location = onnx.TensorProto.DEFAULT

    del fp32_bytes
    gc.collect()

    try:
        model_serialized = model_proto.SerializeToString()
        del model_proto
        gc.collect()
        
        session = InferenceSession(
            model_serialized,
            providers=providers,
            sess_options=sess_options or get_default_sess_options()
        )
        del model_serialized
        return session
    except Exception as e:
        logger.error(f"Failed to load in-memory model {os.path.basename(onnx_path)}: {e}")
        raise e
