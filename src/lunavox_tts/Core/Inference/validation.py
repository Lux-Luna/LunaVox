# Vocoder Input Validation
"""
Validation utilities for VITS/Vocoder inputs.
Extracted from Core/Inference.py for modularization.
"""

import logging
import numpy as np
import onnxruntime as ort
from typing import Dict, Any

logger = logging.getLogger(__name__)


def validate_vocoder_inputs(vocoder: ort.InferenceSession, inputs: Dict[str, Any]) -> None:
    """
    Validate vocoder input shapes and types before inference.
    Provides actionable error messages if validation fails.
    """
    # Get expected inputs from ONNX model
    expected_inputs = {inp.name: inp for inp in vocoder.get_inputs()}
    
    # Check all required inputs are provided
    for name in expected_inputs:
        if name not in inputs:
            if name == 'sv_emb':
                logger.error(
                    f"Missing 'sv_emb' input for vocoder. "
                    f"This model requires v2Pro/v2ProPlus with speaker vector. "
                    f"Please ensure the model was converted with correct version detection."
                )
            elif name in ['ge', 'ge_advanced']:
                logger.error(
                    f"Missing '{name}' input for vocoder. "
                    f"This model requires v2ProPlus with Prompt Encoder features. "
                    f"Please ensure the character was loaded as v2ProPlus."
                )
            else:
                logger.error(f"Missing required input: {name}")
            raise ValueError(f"Missing required input: {name}")
    
    # Validate shapes and types
    for name, value in inputs.items():
        if name not in expected_inputs:
            continue  # Skip extra inputs
        
        expected = expected_inputs[name]
        
        # Handle both numpy and OrtValue
        if isinstance(value, ort.OrtValue):
            actual_shape = value.shape()
            # Type validation for OrtValue is skipped for now
        else:
            actual_shape = value.shape
            actual_dtype = value.dtype
            
            # Validate dtype
            if expected.type == 'tensor(int64)' and actual_dtype != np.int64:
                logger.error(
                    f"Input '{name}' has wrong dtype: {actual_dtype}, expected int64"
                )
                raise TypeError(f"Input '{name}' dtype mismatch: {actual_dtype} != int64")
        
        # Validate specific shapes
        if name == 'sv_emb':
            if actual_shape != (1, 20480):
                logger.error(
                    f"Speaker embedding has wrong shape: {actual_shape}, expected (1, 20480). "
                    f"Please check ERes2NetV2 model output."
                )
                raise ValueError(f"Speaker embedding shape mismatch: {actual_shape} != (1, 20480)")
        elif name == 'ge':
            # v2ProPlus ge shape can be (1, 512) or (1, 1024, 1) depending on export
            if len(actual_shape) not in [2, 3] or actual_shape[0] != 1:
                logger.error(f"Global embedding (ge) has wrong shape: {actual_shape}")
                raise ValueError(f"ge shape invalid: {actual_shape}")
        elif name == 'ge_advanced':
            # v2ProPlus ge_advanced shape is usually (1, 512, 1)
            if len(actual_shape) not in [2, 3] or actual_shape[0] != 1:
                logger.error(f"Advanced global embedding (ge_advanced) has wrong shape: {actual_shape}")
                raise ValueError(f"ge_advanced shape invalid: {actual_shape}")
        elif name == 'text_seq':
            if len(actual_shape) != 2 or actual_shape[0] != 1:
                logger.error(
                    f"Text sequence has wrong shape: {actual_shape}, expected (1, N)"
                )
                raise ValueError(f"Text sequence shape invalid: {actual_shape}")
        elif name == 'pred_semantic':
            # Semantic tokens can be (1, M) or (1, 1, M)
            if len(actual_shape) not in [2, 3] or actual_shape[0] != 1:
                logger.error(
                    f"Semantic tokens have wrong shape: {actual_shape}, expected (1, M) or (1, 1, M)"
                )
                raise ValueError(f"Semantic tokens shape invalid: {actual_shape}")
        elif name == 'ref_audio':
            # Reference audio can be (1, samples) for raw audio or (1, H, W) for features
            if len(actual_shape) not in [2, 3] or actual_shape[0] != 1:
                logger.error(
                    f"Reference audio has wrong shape: {actual_shape}, expected (1, N) or (1, H, W)"
                )
                raise ValueError(f"Reference audio shape invalid: {actual_shape}")
    
    logger.debug(f"✓ Vocoder input validation passed")
