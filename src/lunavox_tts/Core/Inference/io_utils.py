# IO Utilities for ONNX Runtime
"""
ONNX Runtime utility functions for input casting and IO Binding helpers.
Extracted from Core/Inference.py for modularization.
"""

import numpy as np
import onnxruntime as ort
from typing import Dict, Any


def cast_inputs(session: ort.InferenceSession, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Cast inputs to match model precision requirements."""
    casted_inputs = {}
    for input_meta in session.get_inputs():
        name = input_meta.name
        if name not in inputs:
            continue
        
        val = inputs[name]
        if isinstance(val, ort.OrtValue):
            casted_inputs[name] = val
            continue

        target_dtype = input_meta.type
        if target_dtype == 'tensor(float)':
            casted_inputs[name] = val.astype(np.float32)
        elif target_dtype == 'tensor(float16)':
            casted_inputs[name] = val.astype(np.float16)
        elif target_dtype == 'tensor(int64)':
            casted_inputs[name] = val.astype(np.int64)
        elif target_dtype == 'tensor(int32)':
            casted_inputs[name] = val.astype(np.int32)
        else:
            casted_inputs[name] = val
    return casted_inputs
