"""
LunaVox Model Package.

Three-tier model management architecture:
- spec: Model architecture definitions
- registry: Model metadata tracking
- session: ONNX session utilities
- loader: Model loading operations
"""
from .spec import ModelSpec, ModelVersion, ModelFileSpec, get_model_spec, detect_model_version, V2_SPEC, V2_PRO_SPEC, V2_PRO_PLUS_SPEC
from .registry import ModelRegistry, ModelEntry, model_registry
from .session import get_default_sess_options, resolve_providers, load_session_with_fp16_conversion
from .loader import ModelLoader, model_loader

__all__ = [
    # Spec
    "ModelSpec",
    "ModelVersion",
    "ModelFileSpec",
    "get_model_spec",
    "detect_model_version",
    "V2_SPEC",
    "V2_PRO_SPEC",
    "V2_PRO_PLUS_SPEC",
    # Registry
    "ModelRegistry",
    "ModelEntry",
    "model_registry",
    # Session
    "get_default_sess_options",
    "resolve_providers",
    "load_session_with_fp16_conversion",
    # Loader
    "ModelLoader",
    "model_loader",
]
