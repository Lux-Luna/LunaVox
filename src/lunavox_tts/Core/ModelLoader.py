"""
Model Loader - Pure ONNX session loading logic.

This is the second layer of the refactored ModelManager architecture:
- ModelRegistry: Tracks model metadata (paths, versions, status)
- ModelLoader: Handles ONNX session loading (this module)
- ModelManager: High-level facade for model operations
"""
import os
import logging
from typing import Dict, Optional, Tuple, Any

import onnxruntime

from .ModelSession import (
    get_default_sess_options,
    resolve_providers,
    load_session_with_fp16_conversion,
)
from .ModelSpec import ModelSpec, ModelFileSpec

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Handles pure ONNX session loading operations.
    
    This class is responsible only for loading models from disk,
    without any state management or caching concerns.
    """
    
    def __init__(self, providers: Optional[list] = None):
        self.providers = providers or resolve_providers()
        self.sess_options = get_default_sess_options()
    
    def load_component(
        self,
        model_dir: str,
        file_spec: ModelFileSpec,
    ) -> Optional[onnxruntime.InferenceSession]:
        """
        Load a single model component.
        
        Args:
            model_dir: Directory containing model files.
            file_spec: Specification for the file to load.
            
        Returns:
            InferenceSession or None if optional and not found.
        """
        onnx_path = os.path.join(model_dir, file_spec.onnx_file)
        bin_path = os.path.join(model_dir, file_spec.weight_file) if file_spec.weight_file else None
        
        # Check if file exists
        if not os.path.exists(onnx_path):
            if file_spec.required:
                raise FileNotFoundError(f"Required model file not found: {onnx_path}")
            logger.debug(f"Optional model file not found: {onnx_path}")
            return None
        
        try:
            # Use FP16 patching if weight file exists
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
        """
        Load all components defined in a model spec.
        
        Args:
            model_dir: Directory containing model files.
            spec: Model specification.
            skip_components: Set of component names to skip.
            
        Returns:
            Dictionary mapping component names to sessions.
        """
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
        """Refresh execution providers (e.g., after CPU/GPU switch)."""
        self.providers = resolve_providers()


# Global loader instance
model_loader = ModelLoader()
