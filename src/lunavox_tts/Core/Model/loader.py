"""
Model Loader - Pure ONNX session loading logic.

Second layer of the three-tier architecture.
"""
import os
import logging
from typing import Dict, Optional

import onnxruntime

from .session import get_default_sess_options, resolve_providers, load_session_with_fp16_conversion
from .spec import ModelSpec, ModelFileSpec

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles pure ONNX session loading operations."""
    
    def __init__(self, providers: Optional[list] = None):
        self.providers = providers or resolve_providers()
        self.sess_options = get_default_sess_options()
    
    def load_component(
        self,
        model_dir: str,
        file_spec: ModelFileSpec,
    ) -> Optional[onnxruntime.InferenceSession]:
        onnx_path = os.path.join(model_dir, file_spec.onnx_file)
        bin_path = os.path.join(model_dir, file_spec.weight_file) if file_spec.weight_file else None
        
        if not os.path.exists(onnx_path):
            if file_spec.required:
                raise FileNotFoundError(f"Required model file not found: {onnx_path}")
            logger.debug(f"Optional model file not found: {onnx_path}")
            return None
        
        try:
            if bin_path and os.path.exists(bin_path):
                logger.debug(f"Loading with FP16 patching: {file_spec.onnx_file}")
                return load_session_with_fp16_conversion(
                    onnx_path, bin_path, self.providers, self.sess_options
                )
            else:
                logger.debug(f"Loading standard ONNX: {file_spec.onnx_file}")
                return onnxruntime.InferenceSession(
                    onnx_path,
                    providers=self.providers,
                    sess_options=self.sess_options,
                )
        except Exception as e:
            if file_spec.required:
                raise
            logger.warning(f"Failed to load optional component: {e}")
            return None
    
    def load_all(
        self,
        model_dir: str,
        spec: ModelSpec,
        skip_components: Optional[set] = None,
    ) -> Dict[str, onnxruntime.InferenceSession]:
        skip = skip_components or set()
        components = {}
        
        component_specs = [
            ("T2S_ENCODER", spec.t2s_encoder),
            ("T2S_FIRST_STAGE_DECODER", spec.t2s_first_stage_decoder),
            ("T2S_STAGE_DECODER", spec.t2s_stage_decoder),
            ("VITS", spec.vits),
        ]
        
        if spec.prompt_encoder:
            component_specs.append(("PROMPT_ENCODER", spec.prompt_encoder))
        
        for name, file_spec in component_specs:
            if name in skip:
                logger.debug(f"Skipping component: {name}")
                continue
            
            logger.info(f"Loading component: {name}...")
            session = self.load_component(model_dir, file_spec)
            if session is not None:
                components[name] = session
        
        return components
    
    def refresh_providers(self) -> None:
        self.providers = resolve_providers()


# Global loader instance
model_loader = ModelLoader()
