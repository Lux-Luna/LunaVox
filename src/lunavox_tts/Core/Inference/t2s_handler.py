# T2S (Text-to-Semantic) Handler
"""
T2S inference handler with IO Binding and KV Cache management.
Extracted from Core/Inference.py for modularization.
"""

import logging
import numpy as np
import onnxruntime as ort
from typing import List, Optional
import threading

from .io_utils import cast_inputs

logger = logging.getLogger(__name__)


def t2s_iobinding(
        ref_seq: np.ndarray,
        ref_bert: np.ndarray,
        text_seq: np.ndarray,
        text_bert: np.ndarray,
        ssl_content: np.ndarray,
        encoder: ort.InferenceSession,
        first_stage_decoder: ort.InferenceSession,
        stage_decoder: ort.InferenceSession,
        device: str = "cpu",
        stop_event: Optional[threading.Event] = None,
) -> Optional[np.ndarray]:
    """Runs T2S model with IO Binding and KV Cache staying on device (CPU/GPU)"""
    
    # 1. Encoder (Single run)
    encoder_inputs = {
        "ref_seq": ref_seq,
        "text_seq": text_seq,
        "ref_bert": ref_bert,
        "text_bert": text_bert,
        "ssl_content": ssl_content,
    }
    encoder_inputs = cast_inputs(encoder, encoder_inputs)
    
    enc_io = encoder.io_binding()
    for name, val in encoder_inputs.items():
        if isinstance(val, np.ndarray):
            d_val = ort.OrtValue.ortvalue_from_numpy(val, device, 0)
            enc_io.bind_ortvalue_input(name, d_val)
        else:
            enc_io.bind_ortvalue_input(name, val)
    
    for out in encoder.get_outputs():
        enc_io.bind_output(out.name, device)
        
    encoder.run_with_iobinding(enc_io)
    enc_outputs = enc_io.get_outputs()  # These are OrtValues on device
    enc_out_names = [o.name for o in encoder.get_outputs()]
    enc_out_map = {name: val for name, val in zip(enc_out_names, enc_outputs)}
    
    # 2. First Stage Decoder (Single run)
    fs_io = first_stage_decoder.io_binding()
    # Bind outputs from encoder directly to first stage inputs
    for name, d_val in enc_out_map.items():
        if name in [i.name for i in first_stage_decoder.get_inputs()]:
            fs_io.bind_ortvalue_input(name, d_val)
        elif name == "x" or name == "prompts":  # Handle potential name mismatch
            fs_io.bind_ortvalue_input(name, d_val)

    for out in first_stage_decoder.get_outputs():
        fs_io.bind_output(out.name, device)
        
    first_stage_decoder.run_with_iobinding(fs_io)
    fs_outputs = fs_io.get_outputs()  # OrtValues on device
    fs_out_info = first_stage_decoder.get_outputs()
    fs_out_names: List[str] = [o.name for o in fs_out_info]
    
    def _fs_get(name: str, default_idx: int):
        if name in fs_out_names:
            return fs_outputs[fs_out_names.index(name)]
        if default_idx < len(fs_outputs):
            return fs_outputs[default_idx]
        return None

    # Collect per-layer caches from first stage if available (Variant B)
    def _collect_fs_layers(prefix: str):
        layers = []
        for idx, nm in enumerate(fs_out_names):
            if nm.startswith(prefix):
                try:
                    li = int(nm.split("_layer_")[-1])
                except Exception:
                    li = idx
                layers.append((li, fs_outputs[idx]))
        layers.sort(key=lambda x: x[0])
        return [arr for _, arr in layers]

    d_y = _fs_get("y", 0)
    d_y_emb = _fs_get("y_emb", 3)
    d_x_example = _fs_get("x_example", 4)
    
    # Aggregated caches (Variant A)
    d_k_agg = _fs_get("k", 1)
    d_v_agg = _fs_get("v", 2)
    
    # Per-layer caches (Variant B)
    fs_k_layers = _collect_fs_layers("present_k_layer_")
    fs_v_layers = _collect_fs_layers("present_v_layer_")

    # 3. Stage Decoder (Autoregressive Loop)
    stage_in_info = stage_decoder.get_inputs()
    stage_in_names: List[str] = [i.name for i in stage_in_info]
    stage_out_info = stage_decoder.get_outputs()
    stage_out_names: List[str] = [o.name for o in stage_out_info]

    # Determine KV cache structure
    n_past_k = sum(1 for n in stage_in_names if n.startswith("past_k_layer_"))
    n_past_v = sum(1 for n in stage_in_names if n.startswith("past_v_layer_"))
    n_layers = max(n_past_k, n_past_v)

    # Handle split KV cache if needed
    past_kv_ort = {}

    if n_layers > 0:
        # Case 1: Already have per-layer caches from first stage (Variant B)
        if fs_k_layers and fs_v_layers:
            for i in range(min(len(fs_k_layers), n_layers)):
                past_kv_ort[f"past_k_layer_{i}"] = fs_k_layers[i]
                past_kv_ort[f"past_v_layer_{i}"] = fs_v_layers[i]

        # Case 2: Have aggregated cache, need to split (Variant A)
        elif d_k_agg is not None and d_v_agg is not None:
            # Splitting OrtValue is not directly supported, convert to numpy for split
            k_agg = d_k_agg.numpy()
            v_agg = d_v_agg.numpy()
            try:
                split_axis = 0
                # Robustly determine split axis
                if k_agg.shape[0] % n_layers == 0:
                    split_axis = 0
                elif len(k_agg.shape) > 1 and k_agg.shape[1] % n_layers == 0:
                    split_axis = 1
                else:
                    raise ValueError(f"Cannot determine split axis for k_agg shape {k_agg.shape} and {n_layers} layers")

                k_splits = np.split(k_agg, n_layers, axis=split_axis)
                v_splits = np.split(v_agg, n_layers, axis=split_axis)
                
                for i in range(n_layers):
                    # Use ascontiguousarray to ensure memory safety for ORT
                    past_kv_ort[f"past_k_layer_{i}"] = ort.OrtValue.ortvalue_from_numpy(
                        np.ascontiguousarray(k_splits[i]), device, 0)
                    past_kv_ort[f"past_v_layer_{i}"] = ort.OrtValue.ortvalue_from_numpy(
                        np.ascontiguousarray(v_splits[i]), device, 0)
                        
            except Exception as e:
                logger.error(f"Failed to split initial KV cache: {e}. k_agg shape: {k_agg.shape}, n_layers: {n_layers}")
                raise e  # Do not fallback to empty tensors, force error to avoid silence

    # Loop state
    d_iy = d_y
    d_iy_emb = d_y_emb

    # Collected output tokens
    # IMPORTANT: The first stage decoder already produced the first token (T1) in d_y.
    out_tokens = [int(d_y.numpy().flat[0])]

    # Create IO Binding once outside the loop
    io_binding = stage_decoder.io_binding()
    
    # Pre-bind static inputs
    if "ix_example" in stage_in_names and d_x_example:
        io_binding.bind_ortvalue_input("ix_example", d_x_example)
    if "ik" in stage_in_names and d_k_agg:
        io_binding.bind_ortvalue_input("ik", d_k_agg)
    if "iv" in stage_in_names and d_v_agg:
        io_binding.bind_ortvalue_input("iv", d_v_agg)
        
    # Bind all initial Past KV Layers
    for name, val in past_kv_ort.items():
        if name in stage_in_names:
            io_binding.bind_ortvalue_input(name, val)

    if d_k_agg:
        logger.debug(f"IK Agg Shape: {d_k_agg.numpy().shape}")

    for idx in range(500):
        if stop_event is not None and stop_event.is_set():
            return None

        # Only bind inputs that change in the loop
        io_binding.bind_ortvalue_input("iy", d_iy)
        io_binding.bind_ortvalue_input("iy_emb", d_iy_emb)
        
        # Re-bind Outputs because shapes change (KV cache grows)
        for out_name in stage_out_names:
            io_binding.bind_output(out_name, device)
        
        # Run
        try:
            stage_decoder.run_with_iobinding(io_binding)
        except Exception as e:
            logger.error(f"Error during T2S loop step {idx}: {e}")
            raise e
        
        # Retrieve Outputs (as OrtValues)
        raw_outputs = io_binding.get_outputs()
        out_map = {name: val for name, val in zip(stage_out_names, raw_outputs)}

        # Update State
        
        # 1. Samples (Stop Token) - Copy to CPU (minimal overhead)
        d_samples = out_map.get("samples")
        if d_samples:
            samples_cpu = d_samples.numpy()
            val = int(samples_cpu.flat[0])
            out_tokens.append(val)
            
            # Stop Check
            if val >= 1024:
                logger.debug(f"T2S EOS generated at step {idx}, Token={val}")
                break
            
            # Update next input 'iy' (int64)
            d_iy = d_samples
        else:
            # Fallback to y check if samples not present
            d_y_out = out_map.get("y")
            if d_y_out:
                y_cpu = d_y_out.numpy()
                val = int(y_cpu.flat[-1])
                out_tokens.append(val)
                if val >= 1024:
                    logger.debug(f"T2S EOS (from y) generated at step {idx}, Token={val}")
                    break
                d_iy = d_y_out
            else:
                logger.error("No valid output (samples or y) found in T2S step.")
                break

        # 2. Embeddings (y_emb) - Keep on device
        if "y_emb" in out_map:
            d_iy_emb = out_map["y_emb"]
        
        # 3. Update KV Cache - Keep on device and re-bind for next iteration
        for name, val in out_map.items():
            if name.startswith("present_k_layer_"):
                li = int(name.split("_layer_")[-1])
                in_name = f"past_k_layer_{li}"
                if in_name in stage_in_names:
                    io_binding.bind_ortvalue_input(in_name, val)
            elif name.startswith("present_v_layer_"):
                li = int(name.split("_layer_")[-1])
                in_name = f"past_v_layer_{li}"
                if in_name in stage_in_names:
                    io_binding.bind_ortvalue_input(in_name, val)

    # Reconstruct result
    if not out_tokens:
        return np.zeros((1, 0), dtype=np.int64)
    result = np.array([out_tokens], dtype=np.int64)
    return result


def t2s_cpu_deprecated(
        ref_seq: np.ndarray,
        ref_bert: np.ndarray,
        text_seq: np.ndarray,
        text_bert: np.ndarray,
        ssl_content: np.ndarray,
        encoder: ort.InferenceSession,
        first_stage_decoder: ort.InferenceSession,
        stage_decoder: ort.InferenceSession,
        stop_event: Optional[threading.Event] = None,
) -> Optional[np.ndarray]:
    """Deprecated CPU-only T2S implementation."""
    # Encoder
    x, prompts = encoder.run(
        None,
        {
            "ref_seq": ref_seq,
            "text_seq": text_seq,
            "ref_bert": ref_bert,
            "text_bert": text_bert,
            "ssl_content": ssl_content,
        },
    )

    # First Stage Decoder
    y, y_emb, *present_key_values = first_stage_decoder.run(
        None, {"x": x, "prompts": prompts}
    )
    
    # Stage Decoder setup
    stage_input_names = [inp.name for inp in stage_decoder.get_inputs()]
    
    idx: int = 0
    for idx in range(0, 500):
        if stop_event is not None and stop_event.is_set():
            return None
        
        input_feed = {
            name: data
            for name, data in zip(stage_input_names, [y, y_emb, *present_key_values])
        }
        
        outputs = stage_decoder.run(None, input_feed)
        y, y_emb, stop_condition, *present_key_values = outputs

        if stop_condition:
            break

    y[0, -1] = 0
    logger.info(f"T2S generated {idx} tokens")
    return np.expand_dims(y[:, -idx:], axis=0)
