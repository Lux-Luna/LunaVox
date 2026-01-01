# VITS/Vocoder Handler
"""
Vocoder (VITS) inference handler with Prompt Encoder support.
Extracted from Core/Inference.py for modularization.
"""

import logging
import numpy as np
import onnxruntime as ort
from typing import Dict, Any, Optional

from .io_utils import cast_inputs
from ...Utils.PerformanceMonitor import monitor

logger = logging.getLogger(__name__)


def run_prompt_encoder(session: ort.InferenceSession, prompt_audio: Any) -> None:
    """Extract global embeddings using Prompt Encoder with IO Binding support."""
    logger.debug("Running Prompt Encoder to extract global embeddings...")
    if prompt_audio.global_emb is not None:
        logger.debug("Using cached global embeddings.")
        return

    if prompt_audio.sv_emb is None:
        logger.warning("sv_emb is None, cannot update global_emb")
        return

    audio_input = prompt_audio.audio_32k
    if audio_input.ndim == 1:
        audio_input = np.expand_dims(audio_input, axis=0)

    inputs = {
        'ref_audio': audio_input.astype(np.float32),
        'sv_emb': prompt_audio.sv_emb.astype(np.float32),
    }
    
    inputs = cast_inputs(session, inputs)
    device = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
    
    try:
        io_binding = session.io_binding()
        for name, value in inputs.items():
            ort_value = ort.OrtValue.ortvalue_from_numpy(value, device, 0)
            io_binding.bind_ortvalue_input(name, ort_value)
        
        for output in session.get_outputs():
            io_binding.bind_output(output.name, device)
        
        session.run_with_iobinding(io_binding)
        outputs = io_binding.get_outputs()  # These are OrtValues on GPU if device is cuda
        
        # We store them as OrtValues to avoid CPU roundtrip if vocoder also on GPU
        prompt_audio.global_emb = outputs[0]
        prompt_audio.global_emb_advanced = outputs[1]
        
        # Log output shapes for debugging
        ge_shape = prompt_audio.global_emb.shape()
        ge_adv_shape = prompt_audio.global_emb_advanced.shape()
        logger.debug(f"✓ Global embeddings extracted (IO Binding): ge={ge_shape}, ge_adv={ge_adv_shape}, device={device}")
        
    except Exception as e:
        logger.warning(f"Failed to run prompt_encoder with IO binding ({e}). Falling back to regular execution.")
        prompt_audio.update_global_emb(session)
        if prompt_audio.global_emb is not None:
            logger.debug(f"✓ Global embeddings extracted (Standard): ge={prompt_audio.global_emb.shape}, ge_adv={prompt_audio.global_emb_advanced.shape}")


def run_vocoder(session: ort.InferenceSession, inputs: Dict[str, Any]) -> np.ndarray:
    """Run VITS vocoder with IO Binding for performance."""
    # Automatically cast inputs to match model precision
    inputs = cast_inputs(session, inputs)
    
    # Use IO Binding for performance, especially on GPU
    try:
        io_binding = session.io_binding()
        for name, value in inputs.items():
            if isinstance(value, ort.OrtValue):
                io_binding.bind_ortvalue_input(name, value)
            else:
                # Automatically handle device placement
                # If model is on CUDA, move numpy to CUDA
                device = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
                ort_value = ort.OrtValue.ortvalue_from_numpy(value, device, 0)
                io_binding.bind_ortvalue_input(name, ort_value)
        
        for output in session.get_outputs():
            io_binding.bind_output(output.name, "cpu")  # Pull result back to CPU for audio output
        
        with monitor.measure("Vocoder Kernel", category="LINK_DETAIL"):
            session.run_with_iobinding(io_binding)
            
        outputs = io_binding.copy_outputs_to_cpu()
        if outputs:
            return outputs[0]
    except Exception as exc:
        logger.warning(
            "Failed to run vocoder with IO binding (%s). Falling back to regular execution.",
            exc,
        )
    return session.run(None, inputs)[0]
