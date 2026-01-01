"""
ModelSpec - Model architecture specifications and file mappings.

This module centralizes all model-related configuration that was previously
scattered across ModelManager, engine.py, and other modules.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class ModelVersion(Enum):
    """Supported model versions."""
    V2 = "v2"
    V2_PRO = "v2Pro"
    V2_PRO_PLUS = "v2ProPlus"


@dataclass
class ModelFileSpec:
    """Specification for a single model file."""
    onnx_file: str
    weight_file: Optional[str] = None  # FP16 weight file for patching
    required: bool = True


@dataclass 
class ModelSpec:
    """
    Complete specification for a model architecture.
    
    Defines what files are needed, what components are required/optional,
    and what features the model supports.
    """
    version: ModelVersion
    
    # Core model files (always required)
    t2s_encoder: ModelFileSpec = field(default_factory=lambda: ModelFileSpec("t2s_encoder_fp32.onnx"))
    t2s_first_stage_decoder: ModelFileSpec = field(default_factory=lambda: ModelFileSpec(
        "t2s_first_stage_decoder_fp32.onnx", "t2s_shared_fp16.bin"
    ))
    t2s_stage_decoder: ModelFileSpec = field(default_factory=lambda: ModelFileSpec(
        "t2s_stage_decoder_fp32.onnx", "t2s_shared_fp16.bin"
    ))
    vits: ModelFileSpec = field(default_factory=lambda: ModelFileSpec(
        "vits_fp32.onnx", "vits_fp16.bin"
    ))
    
    # Optional components
    prompt_encoder: Optional[ModelFileSpec] = None
    
    # Feature flags
    requires_global_emb: bool = False
    requires_sv_emb: bool = False
    supports_persona: bool = True
    
    def get_required_files(self) -> List[str]:
        """Return list of required ONNX files."""
        files = [
            self.t2s_encoder.onnx_file,
            self.t2s_first_stage_decoder.onnx_file,
            self.t2s_stage_decoder.onnx_file,
            self.vits.onnx_file,
        ]
        if self.prompt_encoder and self.prompt_encoder.required:
            files.append(self.prompt_encoder.onnx_file)
        return files
    
    def get_weight_files(self) -> Dict[str, str]:
        """Return mapping of ONNX file -> weight file for FP16 patching."""
        mapping = {}
        for spec in [self.t2s_first_stage_decoder, self.t2s_stage_decoder, self.vits]:
            if spec.weight_file:
                mapping[spec.onnx_file] = spec.weight_file
        if self.prompt_encoder and self.prompt_encoder.weight_file:
            mapping[self.prompt_encoder.onnx_file] = self.prompt_encoder.weight_file
        return mapping


# Pre-defined model specifications
V2_SPEC = ModelSpec(
    version=ModelVersion.V2,
    requires_global_emb=False,
    requires_sv_emb=False,
)

V2_PRO_SPEC = ModelSpec(
    version=ModelVersion.V2_PRO,
    requires_global_emb=False,
    requires_sv_emb=True,
)

V2_PRO_PLUS_SPEC = ModelSpec(
    version=ModelVersion.V2_PRO_PLUS,
    prompt_encoder=ModelFileSpec(
        "prompt_encoder_fp32.onnx", "prompt_encoder_fp16.bin", required=False
    ),
    requires_global_emb=True,
    requires_sv_emb=True,
)


def get_model_spec(version: str) -> ModelSpec:
    """Get the model specification for a given version string."""
    version_lower = version.lower().replace("_", "").replace("-", "")
    
    if version_lower in ("v2proplus", "v2pp"):
        return V2_PRO_PLUS_SPEC
    elif version_lower == "v2pro":
        return V2_PRO_SPEC
    else:
        return V2_SPEC


def detect_model_version(model_dir: str) -> str:
    """
    Detect model version from directory contents or metadata.
    
    Returns version string: 'v2', 'v2Pro', or 'v2ProPlus'
    """
    import os
    import json
    
    # Check for model_info.json first
    info_path = os.path.join(model_dir, "model_info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
                return info.get("version", "v2")
        except Exception:
            pass
    
    # Fallback: check for prompt_encoder (v2ProPlus indicator)
    if os.path.exists(os.path.join(model_dir, "prompt_encoder_fp32.onnx")):
        return "v2ProPlus"
    
    # Check directory path hints
    dir_lower = model_dir.lower()
    if "v2_pro_plus" in dir_lower or "v2pp" in dir_lower:
        return "v2ProPlus"
    elif "v2_pro" in dir_lower:
        return "v2Pro"
    
    return "v2"
