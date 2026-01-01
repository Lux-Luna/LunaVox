# LunaVox TTS Engine
"""
Main TTS engine class that orchestrates text-to-speech synthesis.
Extracted from Core/Inference.py for modularization.
"""

import os
import logging
import re
import onnxruntime as ort
import numpy as np
from typing import List, Optional
import threading

from ...Audio.ReferenceAudio import ReferenceAudio
from ...Core.TextFrontend import get_text_frontend
from ...Chinese.ZhBert import compute_bert_phone_features
from ...Utils.Constants import BERT_FEATURE_DIM
from ...Utils.PerformanceMonitor import monitor

from .t2s_handler import t2s_iobinding
from .vits_handler import run_vocoder, run_prompt_encoder

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
        
        with monitor.measure("Total TTS Latency", category="USER_PERCEIVED"):
            # 文本前端补符策略：防止漏第一句
            if not text.startswith("。") and not text.startswith("."):
                text = "。" + text
            
            # 文本尾部补符策略：防止最后一句被截断
            if not text.strip().endswith((".", "。", "?", "？", "!", "！", "…", "—", "-")):
                text = text + "。"

            with monitor.measure(f"Frontend ({language})"):
                frontend = get_text_frontend()
                if language == "en":
                    ids = frontend.process_en(text)
                    from ...Japanese.SymbolsV2 import symbols_v2
                    phones = [symbols_v2[i] for i in ids]
                    monitor.log_data("LunaVox phones", phones)
                    text_seq: np.ndarray = np.array([ids], dtype=np.int64)
                    monitor.log_data("LunaVox text_seq", text_seq)
                    text_bert = np.zeros((text_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)
                elif language == "zh":
                    ids, word2ph, norm_text = frontend.process_zh(text)
                    text_seq: np.ndarray = np.array([ids], dtype=np.int64)
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

            from ...Utils.EnvManager import env_manager
            device_mode = env_manager.get_mode()
            device_name = "cuda" if device_mode == "gpu" else "cpu"
            
            # Use optimized IO Binding for both CPU and GPU
            with monitor.measure("T2S Inference"):
                semantic_tokens: np.ndarray = t2s_iobinding(
                    ref_seq=ref_seq,
                    ref_bert=ref_bert,
                    text_seq=text_seq,
                    text_bert=text_bert,
                    ssl_content=prompt_audio.ssl_content,
                    encoder=encoder,
                    first_stage_decoder=first_stage_decoder,
                    stage_decoder=stage_decoder,
                    device=device_name,
                    stop_event=self.stop_event,
                )

            if self.stop_event.is_set():
                return None

            if semantic_tokens is None or semantic_tokens.size == 0:
                return None

            eos_indices = np.where(semantic_tokens >= 1024)
            if len(eos_indices[0]) > 0:
                first_eos_index = eos_indices[-1][0]
                semantic_tokens = semantic_tokens[..., :first_eos_index]

            if semantic_tokens.size == 0:
                return None

            if semantic_tokens.ndim == 2:
                semantic_tokens = np.expand_dims(semantic_tokens, axis=1)
            
            # Prepare ref_audio based on model version
            if prompt_encoder is not None:
                model_version = 'v2ProPlus'
            else:
                model_version = getattr(prompt_audio, 'model_version', 'v2')
            
            if model_version == 'v2Pro':
                try:
                    with monitor.measure("Reference Audio Feature Extraction"):
                        from ...Audio.SpectrogramExtractor import extract_stft_spectrogram
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
                ref_audio_features = None
            else:
                ref_audio_features = np.expand_dims(prompt_audio.audio_32k, axis=0)
                
                # WORKAROUND: Truncate reference audio for VITS if too long on GPU
                if device_mode == "gpu" and not prompt_audio.is_persona_based:
                    MAX_VITS_AUDIO_SAMPLES = 128000
                    if ref_audio_features.shape[1] > MAX_VITS_AUDIO_SAMPLES:
                        logger.warning(f"Truncating VITS ref_audio (GPU) from {ref_audio_features.shape[1]} to {MAX_VITS_AUDIO_SAMPLES} to avoid FP16 overflow.")
                        ref_audio_features = ref_audio_features[:, :MAX_VITS_AUDIO_SAMPLES]
            
            # Build vocoder inputs
            vocoder_inputs = {
                "text_seq": text_seq,
                "pred_semantic": semantic_tokens,
            }
            
            if ref_audio_features is not None:
                vocoder_inputs["ref_audio"] = ref_audio_features
            
            # v2ProPlus: Inject global embeddings
            if model_version == 'v2ProPlus':
                if prompt_audio.global_emb is not None:
                    vocoder_inputs["ge"] = prompt_audio.global_emb
                    vocoder_inputs["ge_advanced"] = prompt_audio.global_emb_advanced
                elif prompt_encoder is not None:
                    with monitor.measure("Prompt Encoder"):
                        run_prompt_encoder(prompt_encoder, prompt_audio)
                    vocoder_inputs["ge"] = prompt_audio.global_emb
                    vocoder_inputs["ge_advanced"] = prompt_audio.global_emb_advanced
                else:
                    raise RuntimeError(
                        "v2ProPlus model requires either cached global_emb (Persona mode) "
                        "or a loaded prompt_encoder session (Reference Audio mode)"
                    )
            
            # Add speaker vector for v2Pro
            if model_version == 'v2Pro' and prompt_audio.sv_emb is not None:
                vocoder_inputs["sv_emb"] = prompt_audio.sv_emb
            
            # Run VITS
            with monitor.measure("VITS Inference"):
                vits_output = run_vocoder(vocoder, vocoder_inputs)
            
            if vits_output is not None:
                vits_output = np.nan_to_num(vits_output)
                vits_output = np.clip(vits_output, -1.0, 1.0)
                
                # Add trailing silence
                silence_samples = int(0.40 * 32000)
                original_shape = vits_output.shape
                flat_audio = vits_output.flatten()
                silence = np.zeros(silence_samples, dtype=vits_output.dtype)
                padded_audio = np.concatenate([flat_audio, silence])
                if len(original_shape) > 1:
                    vits_output = padded_audio.reshape(original_shape[0], -1)
                else:
                    vits_output = padded_audio
            
            monitor.log_data("VITS output", vits_output)
            
            return vits_output


# Module-level singleton for backward compatibility
tts_client: LunaVoxEngine = LunaVoxEngine()
