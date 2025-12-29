import os
import logging
import onnxruntime as ort
import numpy as np
from typing import List, Optional
import threading
import time

from ..Audio.ReferenceAudio import ReferenceAudio
from ..Japanese.JapaneseG2P import japanese_to_phones
from ..English.EnglishG2P import english_to_phones
from ..Chinese.ChineseG2P import chinese_clean_g2p_and_norm
from ..Chinese.ZhBert import compute_bert_phone_features
from ..Utils.Constants import BERT_FEATURE_DIM

USE_IO_BINDING = os.getenv("LUNAVOX_USE_IO_BINDING", "0") == "1"
logger = logging.getLogger(__name__)


class LunaVoxEngine:
    def __init__(self):
        self.stop_event: threading.Event = threading.Event()

    def tts(
            self,
            text: str,
            prompt_audio: ReferenceAudio,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            stage_decoder: ort.InferenceSession,
            vocoder: ort.InferenceSession,
            language: str = "ja",
    ) -> Optional[np.ndarray]:
        t0 = time.time()
        if language == "en":
            ids = english_to_phones(text)
            text_seq: np.ndarray = np.array([ids], dtype=np.int64)
            text_bert = np.zeros((text_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)
        elif language == "zh":
            ids, word2ph, norm_text = chinese_clean_g2p_and_norm(text)
            text_seq: np.ndarray = np.array([ids], dtype=np.int64)
            # Full zh-BERT parity: compute 1024-d features and align to phones
            # Keep BERT on GPU but return numpy for ORT compatibility
            bert_phone = compute_bert_phone_features(norm_text, word2ph, return_tensor=False)
            if bert_phone.shape[0] != text_seq.shape[1]:
                text_bert = np.zeros((text_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)
            else:
                text_bert = bert_phone
        else:
            text_seq: np.ndarray = np.array([japanese_to_phones(text)], dtype=np.int64)
            text_bert = np.zeros((text_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)
        t_frontend = time.time()
        logger.info(f"Frontend ({language}) took: {(t_frontend - t0) * 1000:.2f}ms")

        ref_seq = prompt_audio.phonemes_seq
        if ref_seq is None:
            return None
        ref_bert = prompt_audio.text_bert
        if ref_bert is None or ref_bert.shape[0] != ref_seq.shape[1]:
            ref_bert = np.zeros((ref_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)

        semantic_tokens: np.ndarray = self.t2s_gpu(
            ref_seq=ref_seq,
            ref_bert=ref_bert,
            text_seq=text_seq,
            text_bert=text_bert,
            ssl_content=prompt_audio.ssl_content,
            encoder=encoder,
            first_stage_decoder=first_stage_decoder,
            stage_decoder=stage_decoder,
        )
        t_t2s = time.time()
        logger.info(f"T2S Inference took: {(t_t2s - t_frontend) * 1000:.2f}ms")

        if self.stop_event.is_set():
            return None

        eos_indices = np.where(semantic_tokens >= 1024)  # 剔除不合法的元素，例如 EOS Token。
        if len(eos_indices[0]) > 0:
            first_eos_index = eos_indices[-1][0]
            semantic_tokens = semantic_tokens[..., :first_eos_index]

        # Ensure semantic_tokens has correct shape (1, 1, N) for VITS
        if semantic_tokens.ndim == 2:
            semantic_tokens = np.expand_dims(semantic_tokens, axis=1)  # (1, M) -> (1, 1, M)
        
        # Prepare ref_audio based on model version
        # v2: uses raw audio (2D)
        # v2Pro/v2ProPlus: uses STFT spectrogram (3D)
        model_version = prompt_audio.model_version if hasattr(prompt_audio, 'model_version') else 'v2'
        
        if model_version in ['v2Pro', 'v2ProPlus']:
            # Extract STFT spectrogram for v2Pro/v2ProPlus (matches GPT-SoVITS get_spepc)
            try:
                from ..Audio.SpectrogramExtractor import extract_stft_spectrogram
                ref_audio_features = extract_stft_spectrogram(
                    prompt_audio.audio_32k,
                    n_fft=2048,  # filter_length → 1025 bins (2048//2+1)
                    hop_length=640,
                    win_length=2048,
                    center=False
                )
            except Exception as e:
                # Fallback: try mel-spectrogram
                import logging
                logging.getLogger(__name__).warning(
                    f"STFT spectrogram extraction failed ({e}), trying mel-spectrogram fallback"
                )
                try:
                    from ..Audio.MelExtractor import extract_mel_spectrogram
                    ref_audio_features = extract_mel_spectrogram(prompt_audio.audio_32k, n_mels=704)
                except:
                    # Last resort: use SSL features
                    logging.getLogger(__name__).warning(
                        "All feature extraction failed, using SSL features (may cause issues)"
                    )
                    ref_audio_features = prompt_audio.ssl_content
                    if ref_audio_features.ndim == 3:
                        ref_audio_features = np.transpose(ref_audio_features, (0, 2, 1))
        else:
            # v2: use raw audio (2D: batch, samples)
            ref_audio_features = np.expand_dims(prompt_audio.audio_32k, axis=0)
        
        # Build vocoder inputs
        vocoder_inputs = {
            "text_seq": text_seq,
            "pred_semantic": semantic_tokens,
            "ref_audio": ref_audio_features
        }
        
        # Add speaker vector for v2Pro/v2ProPlus
        if prompt_audio.sv_emb is not None:
            vocoder_inputs["sv_emb"] = prompt_audio.sv_emb
        
        # Validate inputs before calling vocoder
        self._validate_vocoder_inputs(vocoder, vocoder_inputs)
        
        # Run VITS
        vits_output = self._run_vocoder(vocoder, vocoder_inputs)
        
        t_vits = time.time()
        logger.info(f"VITS Inference took: {(t_vits - t_t2s) * 1000:.2f}ms")
        logger.info(f"Total TTS Latency: {(t_vits - t0) * 1000:.2f}ms")

        logger.debug(
            "VITS output: shape=%s, range=[%.6f, %.6f], RMS=%.6f",
            vits_output.shape,
            float(vits_output.min()),
            float(vits_output.max()),
            float(np.sqrt(np.mean(vits_output**2))),
        )
        
        return vits_output

    def _run_vocoder(self, session: ort.InferenceSession, inputs: dict) -> np.ndarray:
        # Automatically cast inputs to match model precision
        inputs = self._cast_inputs(session, inputs)
        
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
                io_binding.bind_output(output.name, "cpu") # Pull result back to CPU for audio output
            
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

    def _cast_inputs(self, session: ort.InferenceSession, inputs: dict) -> dict:
        """Automatically cast inputs to match the model's expected precision (fp32/fp16)."""
        input_meta = {inp.name: inp.type for inp in session.get_inputs()}
        new_inputs = {}
        for name, value in inputs.items():
            if name not in input_meta:
                new_inputs[name] = value
                continue
            
            # If it's a torch tensor, we might need to cast it on GPU
            if hasattr(value, 'dtype') and not isinstance(value, np.ndarray):
                # Assume it's a torch tensor
                import torch
                expected_type = input_meta[name]
                if expected_type == 'tensor(float16)' and value.dtype == torch.float32:
                    new_inputs[name] = value.to(torch.float16)
                elif expected_type == 'tensor(float)' and value.dtype == torch.float16:
                    new_inputs[name] = value.to(torch.float32)
                else:
                    new_inputs[name] = value
                continue

            expected_type = input_meta[name]
            # Handle float conversions
            if expected_type == 'tensor(float16)' and value.dtype == np.float32:
                new_inputs[name] = value.astype(np.float16)
            elif expected_type == 'tensor(float)' and value.dtype == np.float16:
                new_inputs[name] = value.astype(np.float32)
            else:
                new_inputs[name] = value
        return new_inputs

    def _validate_vocoder_inputs(self, vocoder: ort.InferenceSession, 
                                 inputs: dict) -> None:
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
                else:
                    logger.error(f"Missing required input: {name}")
                raise ValueError(f"Missing required input: {name}")
        
        # Validate shapes and types
        for name, value in inputs.items():
            if name not in expected_inputs:
                continue  # Skip extra inputs
            
            expected = expected_inputs[name]
            actual_shape = value.shape
            actual_dtype = value.dtype
            
            # Validate dtype
            if expected.type == 'tensor(int64)' and actual_dtype != np.int64:
                logger.error(
                    f"Input '{name}' has wrong dtype: {actual_dtype}, expected int64"
                )
                raise TypeError(f"Input '{name}' dtype mismatch: {actual_dtype} != int64")
            elif expected.type == 'tensor(float)' and actual_dtype != np.float32:
                pass # Allow float mismatch, will be cast later
            elif expected.type == 'tensor(float16)' and actual_dtype != np.float16:
                pass # Allow mismatch, will be cast later
            
            # Validate specific shapes
            if name == 'sv_emb':
                if actual_shape != (1, 20480):
                    logger.error(
                        f"Speaker embedding has wrong shape: {actual_shape}, expected (1, 20480). "
                        f"Please check ERes2NetV2 model output."
                    )
                    raise ValueError(f"Speaker embedding shape mismatch: {actual_shape} != (1, 20480)")
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

    def t2s_gpu(
            self,
            ref_seq: np.ndarray,
            ref_bert: np.ndarray,
            text_seq: np.ndarray,
            text_bert: np.ndarray,
            ssl_content: np.ndarray,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            stage_decoder: ort.InferenceSession,
    ) -> Optional[np.ndarray]:
        """Runs T2S model with IO Binding and KV Cache staying on GPU"""
        
        # 1. Encoder (Single run)
        encoder_inputs = {
            "ref_seq": ref_seq,
            "text_seq": text_seq,
            "ref_bert": ref_bert,
            "text_bert": text_bert,
            "ssl_content": ssl_content,
        }
        encoder_inputs = self._cast_inputs(encoder, encoder_inputs)
        
        enc_io = encoder.io_binding()
        for name, val in encoder_inputs.items():
            if isinstance(val, np.ndarray):
                d_val = ort.OrtValue.ortvalue_from_numpy(val, "cuda", 0)
                enc_io.bind_ortvalue_input(name, d_val)
            else:
                enc_io.bind_ortvalue_input(name, val)
        
        for out in encoder.get_outputs():
            enc_io.bind_output(out.name, "cuda")
            
        encoder.run_with_iobinding(enc_io)
        enc_outputs = enc_io.get_outputs() # These are OrtValues on GPU
        enc_out_names = [o.name for o in encoder.get_outputs()]
        enc_out_map = {name: val for name, val in zip(enc_out_names, enc_outputs)}
        
        # 2. First Stage Decoder (Single run)
        fs_io = first_stage_decoder.io_binding()
        # Bind outputs from encoder directly to first stage inputs
        for name, d_val in enc_out_map.items():
            if name in [i.name for i in first_stage_decoder.get_inputs()]:
                fs_io.bind_ortvalue_input(name, d_val)
            elif name == "x" or name == "prompts": # Handle potential name mismatch
                fs_io.bind_ortvalue_input(name, d_val)

        for out in first_stage_decoder.get_outputs():
            fs_io.bind_output(out.name, "cuda")
            
        first_stage_decoder.run_with_iobinding(fs_io)
        fs_outputs = fs_io.get_outputs() # OrtValues on GPU
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

        # Prepare GPU IO Binding for the loop
        # We need to allocate buffers for inputs/outputs on GPU to avoid transfers
        
        # Determine KV cache structure
        n_past_k = sum(1 for n in stage_in_names if n.startswith("past_k_layer_"))
        n_past_v = sum(1 for n in stage_in_names if n.startswith("past_v_layer_"))
        n_layers = max(n_past_k, n_past_v)

        # Handle split KV cache if needed
        past_kv_ort = {}
        
        # Helper to create empty tensors matching the required shape
        def _get_empty_past_kv(name):
            for inp in stage_in_info:
                if inp.name == name:
                    shape = list(inp.shape)
                    processed_shape = []
                    for dim in shape:
                        if isinstance(dim, str) or dim is None:
                            if 'seq' in str(dim).lower() or 'past' in str(dim).lower():
                                processed_shape.append(0)
                            else:
                                processed_shape.append(1) # Default to 1 for batch/heads
                        else:
                            processed_shape.append(dim)
                    
                    # Fallback for hidden size if last dim is dynamic
                    if processed_shape and processed_shape[-1] == 0:
                        processed_shape[-1] = 512
                    
                    return np.zeros(processed_shape, dtype=np.float16)
            return None

        if n_layers > 0:
            # Case 1: Already have per-layer caches from first stage (Variant B)
            if fs_k_layers and fs_v_layers:
                for i in range(min(len(fs_k_layers), n_layers)):
                    past_kv_ort[f"past_k_layer_{i}"] = fs_k_layers[i]
                    past_kv_ort[f"past_v_layer_{i}"] = fs_v_layers[i]
            
            # Case 2: Have aggregated cache, need to split (Variant A)
            elif d_k_agg is not None and d_v_agg is not None:
                # Splitting OrtValue is not directly supported, convert to numpy for split
                # This is a one-time thing before loop
                k_agg = d_k_agg.numpy()
                v_agg = d_v_agg.numpy()
                try:
                    split_axis = 0
                    if k_agg.shape[0] % n_layers != 0:
                        if len(k_agg.shape) > 1 and k_agg.shape[1] % n_layers == 0:
                            split_axis = 1
                    
                    k_splits = np.split(k_agg, n_layers, axis=split_axis)
                    v_splits = np.split(v_agg, n_layers, axis=split_axis)
                    for i in range(n_layers):
                        past_kv_ort[f"past_k_layer_{i}"] = ort.OrtValue.ortvalue_from_numpy(
                            np.ascontiguousarray(k_splits[i]), "cuda", 0)
                        past_kv_ort[f"past_v_layer_{i}"] = ort.OrtValue.ortvalue_from_numpy(
                            np.ascontiguousarray(v_splits[i]), "cuda", 0)
                except Exception as e:
                    logger.warning(f"Failed to split initial KV cache: {e}. Falling back to empty tensors.")
            
            # Ensure all required past layers are bound
            for i in range(n_layers):
                if f"past_k_layer_{i}" not in past_kv_ort:
                    empty_k = _get_empty_past_kv(f"past_k_layer_{i}")
                    if empty_k is not None:
                        past_kv_ort[f"past_k_layer_{i}"] = ort.OrtValue.ortvalue_from_numpy(empty_k, "cuda", 0)
                if f"past_v_layer_{i}" not in past_kv_ort:
                    empty_v = _get_empty_past_kv(f"past_v_layer_{i}")
                    if empty_v is not None:
                        past_kv_ort[f"past_v_layer_{i}"] = ort.OrtValue.ortvalue_from_numpy(empty_v, "cuda", 0)

        # Loop state
        # d_y and d_y_emb are already OrtValues on GPU
        d_iy = d_y
        d_iy_emb = d_y_emb

        # Collected output tokens
        out_tokens = []

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

        for idx in range(500):
            if self.stop_event.is_set():
                return None

            # Only bind inputs that change in the loop
            io_binding.bind_ortvalue_input("iy", d_iy)
            io_binding.bind_ortvalue_input("iy_emb", d_iy_emb)
            
            # Re-bind Outputs because shapes change (KV cache grows)
            for out_name in stage_out_names:
                io_binding.bind_output(out_name, "cuda")
            
            # Run
            try:
                stage_decoder.run_with_iobinding(io_binding)
            except Exception as e:
                logger.error(f"Error during T2S GPU loop step {idx}: {e}")
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
                        break
                    d_iy = d_y_out
                else:
                    break

            # 2. Embeddings (y_emb) - Keep on GPU
            if "y_emb" in out_map:
                d_iy_emb = out_map["y_emb"]
            
            # 3. Update KV Cache - Keep on GPU and re-bind for next iteration
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

    def t2s_cpu(
            self,
            ref_seq: np.ndarray,
            ref_bert: np.ndarray,
            text_seq: np.ndarray,
            text_bert: np.ndarray,
            ssl_content: np.ndarray,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            stage_decoder: ort.InferenceSession,
    ) -> Optional[np.ndarray]:
        """在CPU上运行T2S模型"""
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
        fs_outputs = first_stage_decoder.run(None, {"x": x, "prompts": prompts})
        fs_out_info = first_stage_decoder.get_outputs()
        fs_out_names: List[str] = [o.name for o in fs_out_info]

        # Expected (variant A): aggregated outputs [y, k, v, y_emb, x_example]
        def _fs_get(name: str, default_idx: int):
            if name in fs_out_names:
                return fs_outputs[fs_out_names.index(name)]
            if default_idx < len(fs_outputs):
                return fs_outputs[default_idx]
            return None

        # Variant B: per-layer caches 'present_k_layer_i'/'present_v_layer_i'
        def _collect_layers(prefix: str):
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

        y = _fs_get("y", 0)
        k_agg = _fs_get("k", 1)
        v_agg = _fs_get("v", 2)
        y_emb = _fs_get("y_emb", 3)
        x_example = _fs_get("x_example", 4)
        k_layers = _collect_layers("present_k_layer_")
        v_layers = _collect_layers("present_v_layer_")
        if not k_layers:
            k_layers = None
        if not v_layers:
            v_layers = None

        # Stage Decoder
        stage_in_info = stage_decoder.get_inputs()
        stage_in_names: List[str] = [i.name for i in stage_in_info]
        stage_out_info = stage_decoder.get_outputs()
        stage_out_names: List[str] = [o.name for o in stage_out_info]

        # Determine number of per-layer cache inputs expected
        n_past_k = sum(1 for n in stage_in_names if n.startswith("past_k_layer_"))
        n_past_v = sum(1 for n in stage_in_names if n.startswith("past_v_layer_"))
        n_layers = max(n_past_k, n_past_v)

        # If stage expects per-layer caches but only aggregated provided, try to split along axis 0
        if n_layers > 0 and k_layers is None and k_agg is not None:
            try:
                k_layers = list(np.split(k_agg, n_layers, axis=0))
            except Exception:
                k_layers = None
        if n_layers > 0 and v_layers is None and v_agg is not None:
            try:
                v_layers = list(np.split(v_agg, n_layers, axis=0))
            except Exception:
                v_layers = None

        def _build_stage_feed(_y, _y_emb, _k_layers, _v_layers, _k_agg, _v_agg, _x_example):
            feed = {}
            for name in stage_in_names:
                if name == "iy":
                    feed[name] = _y
                elif name == "iy_emb":
                    feed[name] = _y_emb
                elif name == "ix_example" and _x_example is not None:
                    feed[name] = _x_example
                elif name == "ik" and _k_agg is not None:
                    feed[name] = _k_agg
                elif name == "iv" and _v_agg is not None:
                    feed[name] = _v_agg
                elif name.startswith("past_k_layer_") and _k_layers is not None:
                    try:
                        li = int(name.split("_layer_")[-1])
                        if 0 <= li < len(_k_layers):
                            feed[name] = _k_layers[li]
                    except Exception:
                        pass
                elif name.startswith("past_v_layer_") and _v_layers is not None:
                    try:
                        li = int(name.split("_layer_")[-1])
                        if 0 <= li < len(_v_layers):
                            feed[name] = _v_layers[li]
                    except Exception:
                        pass
            return feed

        def _unpack_stage_outputs(outputs_list, prev_y_emb):
            out_map = {name: outputs_list[idx] for idx, name in enumerate(stage_out_names)}
            _y = out_map.get("y", outputs_list[0] if outputs_list else None)
            _y_emb = out_map.get("y_emb", prev_y_emb)
            _k_agg = out_map.get("k", None)
            _v_agg = out_map.get("v", None)
            # per-layer presents
            pres_k_layers = []
            pres_v_layers = []
            for nm, arr in out_map.items():
                if nm.startswith("present_k_layer_"):
                    try:
                        li = int(nm.split("_layer_")[-1])
                    except Exception:
                        li = 0
                    pres_k_layers.append((li, arr))
                elif nm.startswith("present_v_layer_"):
                    try:
                        li = int(nm.split("_layer_")[-1])
                    except Exception:
                        li = 0
                    pres_v_layers.append((li, arr))
            pres_k_layers = [a for _, a in sorted(pres_k_layers, key=lambda x: x[0])] if pres_k_layers else None
            pres_v_layers = [a for _, a in sorted(pres_v_layers, key=lambda x: x[0])] if pres_v_layers else None
            _logits = out_map.get("logits", None)
            _samples = out_map.get("samples", None)
            return _y, _y_emb, pres_k_layers, pres_v_layers, _k_agg, _v_agg, _logits, _samples

        idx: int = 0
        for idx in range(0, 500):
            if self.stop_event.is_set():
                return None

            input_feed = _build_stage_feed(y, y_emb, k_layers, v_layers, k_agg, v_agg, x_example)
            outputs_list = stage_decoder.run(None, input_feed)
            y, y_emb, new_k_layers, new_v_layers, new_k_agg, new_v_agg, logits, samples = _unpack_stage_outputs(outputs_list, y_emb)

            # Update caches for next step
            if new_k_layers is not None and new_v_layers is not None:
                k_layers, v_layers = new_k_layers, new_v_layers
                k_agg, v_agg = None, None
            else:
                k_agg, v_agg = new_k_agg if new_k_agg is not None else k_agg, new_v_agg if new_v_agg is not None else v_agg

            # EOS/停机判定：优先使用 samples，其次用 logits argmax，最后用 y 值范围
            stop = False
            if samples is not None:
                try:
                    val = int(samples.flat[0])
                    if val >= 1024:
                        stop = True
                except Exception:
                    pass
            elif logits is not None:
                try:
                    last = logits[..., -1, :]
                    val = int(np.argmax(last))
                    if val >= 1024:
                        stop = True
                except Exception:
                    pass
            else:
                try:
                    if int(y.flat[-1]) >= 1024:
                        stop = True
                except Exception:
                    pass

            if stop:
                break

        y[0, -1] = 0
        return np.expand_dims(y[:, -idx:], axis=0)


tts_client: LunaVoxEngine = LunaVoxEngine()
