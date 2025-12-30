import os
import logging
import re
import onnxruntime as ort
import numpy as np
from typing import List, Optional, Literal
import threading
import time

from ..Audio.ReferenceAudio import ReferenceAudio
from ..Core.TextFrontend import get_text_frontend
from ..Chinese.ZhBert import compute_bert_phone_features
from ..Utils.Constants import BERT_FEATURE_DIM
from ..Utils.PerformanceMonitor import monitor

USE_IO_BINDING = os.getenv("LUNAVOX_USE_IO_BINDING", "0") == "1"
logger = logging.getLogger(__name__)


class LunaVoxEngine:
    def __init__(self):
        self.stop_event: threading.Event = threading.Event()

    def split_language(self, text: str) -> List[dict]:
        """从文本中提取中文和英文部分，返回一个包含语言和内容的列表。"""
        pattern_eng = re.compile(r"[a-zA-Z]+")
        split = re.split(pattern_eng, text)
        matches = pattern_eng.findall(text)

        result = []
        for i, part in enumerate(split):
            if part.strip():
                result.append({'language': 'zh', 'content': part})
            if i < len(matches):
                result.append({'language': 'en', 'content': matches[i]})

        return result

    def tts(
            self,
            text: str,
            prompt_audio: ReferenceAudio,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            stage_decoder: ort.InferenceSession,
            vocoder: ort.InferenceSession,
            prompt_encoder: Optional[ort.InferenceSession] = None,
            language: str = "ja",
    ) -> Optional[np.ndarray]:
        
        with monitor.measure("Total TTS Latency"):
            # 文本前端补符策略：防止漏第一句
            if not text.startswith("。") and not text.startswith("."):
                text = "。" + text

            with monitor.measure(f"Frontend ({language})"):
                frontend = get_text_frontend()
                if language == "en":
                    ids = frontend.process_en(text)
                    from ..Japanese.SymbolsV2 import symbols_v2
                    phones = [symbols_v2[i] for i in ids]
                    monitor.log_data("LunaVox phones", phones)
                    text_seq: np.ndarray = np.array([ids], dtype=np.int64)
                    monitor.log_data("LunaVox text_seq", text_seq)
                    text_bert = np.zeros((text_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)
                elif language == "zh":
                    ids, word2ph, norm_text = frontend.process_zh(text)
                    text_seq: np.ndarray = np.array([ids], dtype=np.int64)
                    # Full zh-BERT parity: compute 1024-d features and align to phones
                    # Keep BERT on GPU but return numpy for ORT compatibility
                    bert_phone = compute_bert_phone_features(norm_text, word2ph, return_tensor=False)
                    if bert_phone.shape[0] != text_seq.shape[1]:
                        text_bert = np.zeros((text_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)
                    else:
                        text_bert = bert_phone
                elif language == "hybrid":
                    # 混合语言支持 (中英混合)
                    chunks = self.split_language(text)
                    all_ids = []
                    all_berts = []
                    for chunk in chunks:
                        if chunk['language'] == 'en':
                            ids = frontend.process_en(chunk['content'])
                            all_ids.extend(ids)
                            all_berts.append(np.zeros((len(ids), BERT_FEATURE_DIM), dtype=np.float32))
                        else:
                            ids, word2ph, norm_text = frontend.process_zh(chunk['content'])
                            all_ids.extend(ids)
                            bert_phone = compute_bert_phone_features(norm_text, word2ph, return_tensor=False)
                            if bert_phone.shape[0] != len(ids):
                                all_berts.append(np.zeros((len(ids), BERT_FEATURE_DIM), dtype=np.float32))
                            else:
                                all_berts.append(bert_phone)
                    text_seq = np.array([all_ids], dtype=np.int64)
                    if all_berts:
                        text_bert = np.concatenate(all_berts, axis=0)
                    else:
                        text_bert = np.zeros((text_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)
                else:
                    text_seq: np.ndarray = np.array([frontend.process_ja(text)], dtype=np.int64)
                    text_bert = np.zeros((text_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)

            ref_seq = prompt_audio.phonemes_seq
            if ref_seq is None:
                return None
            ref_bert = prompt_audio.text_bert
            if ref_bert is None or ref_bert.shape[0] != ref_seq.shape[1]:
                ref_bert = np.zeros((ref_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)

            from ..Utils.EnvManager import env_manager
            device_mode = env_manager.get_mode()
            device_name = "cuda" if device_mode == "gpu" else "cpu"
            
            # Use optimized IO Binding for both CPU and GPU to avoid Python<->C++ data copy overhead in the loop
            with monitor.measure("T2S Inference"):
                semantic_tokens: np.ndarray = self._t2s_iobinding(
                    ref_seq=ref_seq,
                    ref_bert=ref_bert,
                    text_seq=text_seq,
                    text_bert=text_bert,
                    ssl_content=prompt_audio.ssl_content,
                    encoder=encoder,
                    first_stage_decoder=first_stage_decoder,
                    stage_decoder=stage_decoder,
                    device=device_name
                )

            if self.stop_event.is_set():
                return None

            if semantic_tokens is None or semantic_tokens.size == 0:
                return None

            eos_indices = np.where(semantic_tokens >= 1024)  # 剔除不合法的元素，例如 EOS Token。
            if len(eos_indices[0]) > 0:
                first_eos_index = eos_indices[-1][0]
                semantic_tokens = semantic_tokens[..., :first_eos_index]

            # Ensure we still have tokens after EOS removal
            if semantic_tokens.size == 0:
                return None

            # Ensure semantic_tokens has correct shape (1, 1, N) for VITS
            if semantic_tokens.ndim == 2:
                semantic_tokens = np.expand_dims(semantic_tokens, axis=1)  # (1, M) -> (1, 1, M)
            
            # Prepare ref_audio based on model version
            # v2: uses raw audio (2D)
            # v2Pro: uses STFT spectrogram (3D)
            # v2ProPlus: uses Prompt Encoder features (no ref_audio needed for VITS)
            model_version = prompt_audio.model_version if hasattr(prompt_audio, 'model_version') else 'v2'
            
            if model_version == 'v2Pro':
                # Extract STFT spectrogram for v2Pro (matches GPT-SoVITS get_spec)
                try:
                    with monitor.measure("Reference Audio Feature Extraction"):
                        from ..Audio.SpectrogramExtractor import extract_stft_spectrogram
                        # IMPORTANT: GPT-SoVITS v2 Pro typically uses filter_length=1406 (704 bins)
                        ref_audio_features = extract_stft_spectrogram(
                            prompt_audio.audio_32k,
                            n_fft=1406,  
                            hop_length=640,
                            win_length=1406,
                            center=False
                        )
                except Exception as e:
                    logger.warning(f"STFT spectrogram extraction failed ({e}), using raw audio as last resort")
                    ref_audio_features = np.expand_dims(prompt_audio.audio_32k, axis=0)
            elif model_version == 'v2ProPlus':
                # v2ProPlus uses ge/ge_advanced from Prompt Encoder, ref_audio is not required for vocoder
                ref_audio_features = None
            else:
                # v2: use raw audio (2D: batch, samples)
                ref_audio_features = np.expand_dims(prompt_audio.audio_32k, axis=0)
                
                # WORKAROUND: Truncate reference audio for VITS if too long.
                # Long raw audio inputs (> ~4-5s) can cause FP16 overflow in VITS models, resulting in NaN output.
                MAX_VITS_AUDIO_SAMPLES = 128000
                if ref_audio_features.shape[1] > MAX_VITS_AUDIO_SAMPLES:
                    logger.warning(f"Truncating VITS ref_audio from {ref_audio_features.shape[1]} to {MAX_VITS_AUDIO_SAMPLES} to avoid FP16 overflow.")
                    ref_audio_features = ref_audio_features[:, :MAX_VITS_AUDIO_SAMPLES]
            
            # Build vocoder inputs
            vocoder_inputs = {
                "text_seq": text_seq,
                "pred_semantic": semantic_tokens,
            }
            
            if ref_audio_features is not None:
                vocoder_inputs["ref_audio"] = ref_audio_features
            
            # v2ProPlus: Use Prompt Encoder to extract global embeddings
            if model_version == 'v2ProPlus' and prompt_encoder is not None:
                with monitor.measure("Prompt Encoder"):
                    self._run_prompt_encoder(prompt_encoder, prompt_audio)
                if prompt_audio.global_emb is not None:
                    vocoder_inputs["ge"] = prompt_audio.global_emb
                if prompt_audio.global_emb_advanced is not None:
                    vocoder_inputs["ge_advanced"] = prompt_audio.global_emb_advanced
            
            # Add speaker vector for v2Pro (v2ProPlus uses ge/ge_advanced instead)
            if model_version == 'v2Pro' and prompt_audio.sv_emb is not None:
                vocoder_inputs["sv_emb"] = prompt_audio.sv_emb
            
            # Validate inputs before calling vocoder
            # self._validate_vocoder_inputs(vocoder, vocoder_inputs)
            
            # Run VITS
            # Use IOBinding via _run_vocoder
            with monitor.measure("VITS Inference"):
                vits_output = self._run_vocoder(vocoder, vocoder_inputs)
            
            if vits_output is not None:
                # Final safety check: clip output to avoid pops/noise from overflow
                # and handle NaNs if any slipped through
                vits_output = np.nan_to_num(vits_output)
                vits_output = np.clip(vits_output, -1.0, 1.0)
            
            monitor.log_data("VITS output", vits_output)
            
            return vits_output

    def _run_prompt_encoder(self, session: ort.InferenceSession, prompt_audio: ReferenceAudio) -> None:
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
        
        inputs = self._cast_inputs(session, inputs)
        device = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
        
        try:
            io_binding = session.io_binding()
            for name, value in inputs.items():
                ort_value = ort.OrtValue.ortvalue_from_numpy(value, device, 0)
                io_binding.bind_ortvalue_input(name, ort_value)
            
            for output in session.get_outputs():
                io_binding.bind_output(output.name, device)
            
            session.run_with_iobinding(io_binding)
            outputs = io_binding.get_outputs() # These are OrtValues on GPU if device is cuda
            
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
                # Type validation for OrtValue is skiped for now
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

    def _t2s_iobinding(
            self,
            ref_seq: np.ndarray,
            ref_bert: np.ndarray,
            text_seq: np.ndarray,
            text_bert: np.ndarray,
            ssl_content: np.ndarray,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            stage_decoder: ort.InferenceSession,
            device: str = "cpu",
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
        encoder_inputs = self._cast_inputs(encoder, encoder_inputs)
        
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
        enc_outputs = enc_io.get_outputs() # These are OrtValues on device
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
            fs_io.bind_output(out.name, device)
            
        first_stage_decoder.run_with_iobinding(fs_io)
        fs_outputs = fs_io.get_outputs() # OrtValues on device
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
                    
                    # Use float32 to match model precision
                    return np.zeros(processed_shape, dtype=np.float32)
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
                    # Robustly determine split axis
                    if k_agg.shape[0] % n_layers == 0:
                        split_axis = 0
                    elif len(k_agg.shape) > 1 and k_agg.shape[1] % n_layers == 0:
                        split_axis = 1
                    else:
                        raise ValueError(f"Cannot determine split axis for k_agg shape {k_agg.shape} and {n_layers} layers")
                    
                    if split_axis == 0 and k_agg.shape[0] != n_layers:
                        # Extra check: if axis 0 dim is not multiple, we might have issue, but above modulo check handles it.
                        # However, sometimes shape is (Layers, Batch, ...) where Batch=1.
                        # Let's ensure strict division results in expected per-layer shape.
                        pass

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
                    raise e # Do not fallback to empty tensors, force error to avoid silence


        # Loop state
        # d_y and d_y_emb are already OrtValues on device
        d_iy = d_y
        d_iy_emb = d_y_emb

        # Collected output tokens
        # IMPORTANT: The first stage decoder already produced the first token (T1) in d_y.
        # We must include it to avoid "swallowing" the beginning of the sentence.
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

        if d_k_agg: logger.debug(f"IK Agg Shape: {d_k_agg.numpy().shape}")

        for idx in range(500):
            if self.stop_event.is_set():
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
            if self.stop_event.is_set():
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


tts_client: LunaVoxEngine = LunaVoxEngine()
