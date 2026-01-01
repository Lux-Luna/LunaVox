"""
Model Specification - Architecture definitions and file mappings.

Centralizes all model-related configuration for v2, v2Pro, and v2ProPlus.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
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
    weight_file: Optional[str] = None
    required: bool = True


@dataclass
class VocoderInputSpec:
    """
    Specification for vocoder input assembly.
    
    Defines which inputs are required/optional for each model version's vocoder,
    enabling spec-driven input assembly instead of hardcoded version checks.
    """
    required_inputs: List[str] = field(default_factory=lambda: ["text_seq", "pred_semantic"])
    optional_inputs: List[str] = field(default_factory=list)
    
    # Feature source mappings: input_name -> attribute path on FeaturePacket/ReferenceAudio
    feature_mappings: Dict[str, str] = field(default_factory=dict)
    
    def get_all_inputs(self) -> List[str]:
        """Get all possible inputs (required + optional)."""
        return self.required_inputs + self.optional_inputs


# Pre-defined vocoder input specifications
V2_VOCODER_INPUTS = VocoderInputSpec(
    required_inputs=["text_seq", "pred_semantic", "ref_audio"],
    optional_inputs=[],
    feature_mappings={
        "ref_audio": "audio_32k",
    }
)

V2_PRO_VOCODER_INPUTS = VocoderInputSpec(
    required_inputs=["text_seq", "pred_semantic", "ref_audio"],
    optional_inputs=["sv_emb"],
    feature_mappings={
        "ref_audio": "audio_32k",
        "sv_emb": "sv_emb",
    }
)

V2_PRO_PLUS_VOCODER_INPUTS = VocoderInputSpec(
    required_inputs=["text_seq", "pred_semantic", "ge", "ge_advanced"],
    optional_inputs=["ref_audio", "sv_emb"],
    feature_mappings={
        "ge": "global_emb",
        "ge_advanced": "global_emb_advanced",
        "ref_audio": "audio_32k",
        "sv_emb": "sv_emb",
    }
)


@dataclass 
class ModelSpec:
    """Complete specification for a model architecture."""
    version: ModelVersion
    
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
    
    prompt_encoder: Optional[ModelFileSpec] = None
    
    requires_global_emb: bool = False
    requires_sv_emb: bool = False
    supports_persona: bool = True
    
    # Vocoder input specification
    vocoder_inputs: VocoderInputSpec = field(default_factory=VocoderInputSpec)
    
    def get_required_files(self) -> List[str]:
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
        mapping = {}
        for spec in [self.t2s_first_stage_decoder, self.t2s_stage_decoder, self.vits]:
            if spec.weight_file:
                mapping[spec.onnx_file] = spec.weight_file
        if self.prompt_encoder and self.prompt_encoder.weight_file:
            mapping[self.prompt_encoder.onnx_file] = self.prompt_encoder.weight_file
        return mapping
    
    def assemble_vocoder_inputs(
        self, 
        text_seq: Any, 
        pred_semantic: Any, 
        features: Any,
        vocoder_session: Any = None,
    ) -> Dict[str, Any]:
        """
        Assemble vocoder inputs based on spec configuration.
        
        Args:
            text_seq: Tokenized text sequence
            pred_semantic: Predicted semantic tokens from T2S
            features: ReferenceAudio or FeaturePacket with extracted features
            vocoder_session: Optional ONNX session to check expected inputs
            
        Returns:
            Dictionary of vocoder inputs ready for inference
        """
        import numpy as np
        
        inputs = {
            "text_seq": text_seq,
            "pred_semantic": pred_semantic,
        }
        
        # Add required and optional inputs from features
        for input_name in self.vocoder_inputs.get_all_inputs():
            if input_name in inputs:
                continue  # Already added
                
            attr_name = self.vocoder_inputs.feature_mappings.get(input_name, input_name)
            value = getattr(features, attr_name, None)
            
            if value is not None:
                # Handle audio expansion if needed
                if input_name == "ref_audio" and isinstance(value, np.ndarray) and value.ndim == 1:
                    value = np.expand_dims(value, axis=0)
                inputs[input_name] = value
        
        # Robustness: If vocoder session provided, satisfy any expected inputs that features can provide
        # This handles cross-version compatibility (e.g., Universal Persona on v2 vs v2pp)
        if vocoder_session is not None:
            expected = {i.name for i in vocoder_session.get_inputs()}
            
            for name in expected:
                if name not in inputs:
                    # Try to get from features by attribute name or via feature_mappings
                    attr_name = self.vocoder_inputs.feature_mappings.get(name, name)
                    value = getattr(features, attr_name, None)
                    if value is not None:
                        if name == "ref_audio" and isinstance(value, np.ndarray) and value.ndim == 1:
                            value = np.expand_dims(value, axis=0)
                        inputs[name] = value
            
            # Filter to ONLY include inputs the vocoder expects (critical for cross-version compatibility)
            inputs = {k: v for k, v in inputs.items() if k in expected}
        
        return inputs


# Pre-defined model specifications
V2_SPEC = ModelSpec(
    version=ModelVersion.V2, 
    requires_global_emb=False, 
    requires_sv_emb=False,
    vocoder_inputs=V2_VOCODER_INPUTS,
)
V2_PRO_SPEC = ModelSpec(
    version=ModelVersion.V2_PRO, 
    requires_global_emb=False, 
    requires_sv_emb=True,
    vocoder_inputs=V2_PRO_VOCODER_INPUTS,
)
V2_PRO_PLUS_SPEC = ModelSpec(
    version=ModelVersion.V2_PRO_PLUS,
    prompt_encoder=ModelFileSpec("prompt_encoder_fp32.onnx", "prompt_encoder_fp16.bin", required=False),
    requires_global_emb=True,
    requires_sv_emb=True,
    vocoder_inputs=V2_PRO_PLUS_VOCODER_INPUTS,
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
    """Detect model version from directory contents or metadata."""
    import os
    import json
    
    info_path = os.path.join(model_dir, "model_info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
                return info.get("version", "v2")
        except Exception:
            pass
    
    if os.path.exists(os.path.join(model_dir, "prompt_encoder_fp32.onnx")):
        return "v2ProPlus"
    
    dir_lower = model_dir.lower()
    if "v2_pro_plus" in dir_lower or "v2pp" in dir_lower:
        return "v2ProPlus"
    elif "v2_pro" in dir_lower:
        return "v2Pro"
    
    return "v2"

