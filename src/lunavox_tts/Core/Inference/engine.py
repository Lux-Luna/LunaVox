"""
LunaVox TTS Engine.

Centralized inference engine that orchestrates text processing, 
feature extraction, and model execution via a modular pipeline.
"""
import os
import logging
import threading
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import onnxruntime as ort

from ...Core.Frontend.processor import text_processor
from ...Core.Processors.feature_extractor import feature_extractor
from ...Utils.PerformanceMonitor import monitor
from ...Utils.EnvManager import env_manager
from ...Core.Session import SynthesisSession

from .t2s_handler import t2s_iobinding
from .vits_handler import run_vocoder, run_prompt_encoder

logger = logging.getLogger(__name__)

# Constants
BERT_FEATURE_DIM = 1024


class LunaVoxEngine:
    """
    Main TTS inference engine using a staged pipeline.
    """
    
    def __init__(self, model_provider=None):
        self.stop_event: threading.Event = threading.Event()
        self._model_provider = model_provider

    def tts(
        self,
        text: str,
        prompt_audio: Any,
        encoder: ort.InferenceSession,
        first_stage_decoder: ort.InferenceSession,
        stage_decoder: ort.InferenceSession,
        vocoder: ort.InferenceSession,
        prompt_encoder: Optional[ort.InferenceSession] = None,
        language: str = "ja",
    ) -> Optional[np.ndarray]:
        """Backward compatible entry point."""
        # Wrap legacy call into session-based pipeline logic internally
        # Note: This is a bridge until all callers use SynthesisSession
        session = SynthesisSession(
            speaker="legacy",
            language=language,
            prompt_audio=prompt_audio,
            model_version='v2ProPlus' if prompt_encoder else getattr(prompt_audio, 'model_version', 'v2')
        )
        # Create a mock GSVModel-like object for the sessions
        from dataclasses import dataclass
        @dataclass
        class MockModel:
            T2S_ENCODER: ort.InferenceSession
            T2S_FIRST_STAGE_DECODER: ort.InferenceSession
            T2S_STAGE_DECODER: ort.InferenceSession
            VITS: ort.InferenceSession
            PROMPT_ENCODER: Optional[ort.InferenceSession]
        
        session.model = MockModel(encoder, first_stage_decoder, stage_decoder, vocoder, prompt_encoder)
        return self.generate(text, session)

    def generate(self, text: str, session: SynthesisSession) -> Optional[np.ndarray]:
        """
        Main entry point for session-based TTS generation.
        Executes the full pipeline: Preprocess -> Features -> T2S -> Vocoder -> Postprocess.
        """
        with monitor.measure("Total TTS Latency", category="USER_PERCEIVED"):
            # 1. Pipeline: Preprocess (Linguistic)
            text_seq, text_bert = self.preprocess_stage(text, session)
            
            # 2. Pipeline: Feature Extraction (Acoustic Reference)
            feature_extractor.process(
                session.prompt_audio, 
                session.model_version, 
                session.model.PROMPT_ENCODER if session.model else None
            )
            
            if self.stop_event.is_set(): return None

            # 3. Pipeline: T2S (Text-to-Semantic)
            semantic_tokens = self.t2s_stage(text_seq, text_bert, session)
            if semantic_tokens is None: return None
            
            if self.stop_event.is_set(): return None

            # 4. Pipeline: Vocoder (VITS Synthesis)
            audio = self.vocoder_stage(semantic_tokens, text_seq, session)
            if audio is None: return None
            
            # 5. Pipeline: Postprocess (Audio Cleanup)
            return self.postprocess_stage(audio)

    def preprocess_stage(self, text: str, session: SynthesisSession) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stage 1: Text Frontend.
        Adds punctuation, tokenizes, and extracts BERT features.
        """
        with monitor.measure(f"Frontend ({session.language})"):
            # Normalize punctuation for stability
            text = text_processor.normalize_punctuation(text, session.language)
            
            from ...Core.Frontend import get_language_frontend
            frontend = get_language_frontend(session.language)
            ids, bert_features = frontend.process(text)
            
            text_seq = np.array([ids], dtype=np.int64)
            return text_seq, bert_features

    def t2s_stage(self, text_seq: np.ndarray, text_bert: np.ndarray, session: SynthesisSession) -> Optional[np.ndarray]:
        """
        Stage 3: T2S Inference.
        Generates semantic tokens from linguistic features.
        """
        prompt_audio = session.prompt_audio
        ref_seq = prompt_audio.phonemes_seq
        if ref_seq is None:
            return None
            
        ref_bert = prompt_audio.text_bert
        if ref_bert is None or ref_bert.shape[0] != ref_seq.shape[1]:
            ref_bert = np.zeros((ref_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)

        device_mode = env_manager.get_mode()
        device_name = "cuda" if device_mode == "gpu" else "cpu"
        
        with monitor.measure("T2S Inference"):
            semantic_tokens = t2s_iobinding(
                ref_seq=ref_seq,
                ref_bert=ref_bert,
                text_seq=text_seq,
                text_bert=text_bert,
                ssl_content=prompt_audio.ssl_content,
                encoder=session.model.T2S_ENCODER,
                first_stage_decoder=session.model.T2S_FIRST_STAGE_DECODER,
                stage_decoder=session.model.T2S_STAGE_DECODER,
                device=device_name,
                stop_event=self.stop_event,
            )

        if semantic_tokens is None or semantic_tokens.size == 0:
            return None

        # EOS Handling
        eos_indices = np.where(semantic_tokens >= 1024)
        if len(eos_indices[0]) > 0:
            first_eos_index = eos_indices[-1][0]
            semantic_tokens = semantic_tokens[..., :first_eos_index]

        if semantic_tokens.size == 0:
            return None

        if semantic_tokens.ndim == 2:
            semantic_tokens = np.expand_dims(semantic_tokens, axis=1)
            
        return semantic_tokens

    def vocoder_stage(self, semantic_tokens: np.ndarray, text_seq: np.ndarray, session: SynthesisSession) -> Optional[np.ndarray]:
        """
        Stage 4: VITS Vocoder.
        Synthesizes waveform from semantic tokens and acoustic features.
        """
        prompt_audio = session.prompt_audio
        model_version = session.model_version
        device_mode = env_manager.get_mode()
        
        # Prepare vocoder inputs
        vocoder_inputs = {
            "text_seq": text_seq,
            "pred_semantic": semantic_tokens,
        }
        
        # Handle different model feature requirements
        if model_version == 'v2Pro':
            with monitor.measure("Reference Audio Feature Extraction"):
                from ...Resources.Audio.SpectrogramExtractor import extract_stft_spectrogram
                ref_audio_features = extract_stft_spectrogram(
                    prompt_audio.audio_32k,
                    n_fft=1406,
                    hop_length=640,
                    win_length=1406,
                    center=False
                )
            vocoder_inputs["ref_audio"] = ref_audio_features
            if prompt_audio.sv_emb is not None:
                vocoder_inputs["sv_emb"] = prompt_audio.sv_emb
                
        elif model_version == 'v2ProPlus':
            if prompt_audio.global_emb is not None:
                vocoder_inputs["ge"] = prompt_audio.global_emb
                vocoder_inputs["ge_advanced"] = prompt_audio.global_emb_advanced
            else:
                # Lazy prompt encoder run if not pre-extracted (Referrence mode fallback)
                if session.model and session.model.PROMPT_ENCODER:
                    with monitor.measure("Prompt Encoder"):
                        run_prompt_encoder(session.model.PROMPT_ENCODER, prompt_audio)
                    vocoder_inputs["ge"] = prompt_audio.global_emb
                    vocoder_inputs["ge_advanced"] = prompt_audio.global_emb_advanced
                else:
                    raise RuntimeError("v2ProPlus requires global_emb or prompt_encoder")
            
            # v2ProPlus also optionally takes ref_audio for some variants, but usually skips it
            # We'll let the robustness check below handle it if requested.
        else:
            # Standard v2
            ref_audio_features = np.expand_dims(prompt_audio.audio_32k, axis=0)
            # WORKAROUND: Truncate reference audio for VITS if too long on GPU
            if device_mode == "gpu" and not prompt_audio.is_persona_based:
                MAX_VITS_AUDIO_SAMPLES = 128000
                if ref_audio_features.shape[1] > MAX_VITS_AUDIO_SAMPLES:
                    ref_audio_features = ref_audio_features[:, :MAX_VITS_AUDIO_SAMPLES]
            vocoder_inputs["ref_audio"] = ref_audio_features

        # Robustness check: Satisfy missing required inputs if persona has them
        # This handles cross-version compatibility (e.g. Universal Persona on v2 model that is actually v2pp)
        expected_inputs = [i.name for i in session.model.VITS.get_inputs()]
        
        if "ge" in expected_inputs and "ge" not in vocoder_inputs:
            if prompt_audio.global_emb is not None:
                vocoder_inputs["ge"] = prompt_audio.global_emb
                vocoder_inputs["ge_advanced"] = prompt_audio.global_emb_advanced
                logger.debug("Automatically provided missing 'ge' inputs from Universal Persona.")
            elif session.model.PROMPT_ENCODER:
                with monitor.measure("Prompt Encoder (Fallback)"):
                    run_prompt_encoder(session.model.PROMPT_ENCODER, prompt_audio)
                vocoder_inputs["ge"] = prompt_audio.global_emb
                vocoder_inputs["ge_advanced"] = prompt_audio.global_emb_advanced
        
        if "sv_emb" in expected_inputs and "sv_emb" not in vocoder_inputs:
            if prompt_audio.sv_emb is not None:
                vocoder_inputs["sv_emb"] = prompt_audio.sv_emb
                logger.debug("Automatically provided missing 'sv_emb' input from persona.")
        
        if "ref_audio" in expected_inputs and "ref_audio" not in vocoder_inputs:
             vocoder_inputs["ref_audio"] = np.expand_dims(prompt_audio.audio_32k, axis=0)

        # Run Vocoder
        with monitor.measure("VITS Inference"):
            audio = run_vocoder(session.model.VITS, vocoder_inputs)
            
        return audio

    def postprocess_stage(self, audio: np.ndarray) -> np.ndarray:
        """
        Stage 5: Post-processing.
        Cleans up NaNs and adds comfort silence.
        """
        if audio is None:
            return None
            
        audio = np.nan_to_num(audio)
        audio = np.clip(audio, -1.0, 1.0)
        
        # Add trailing silence (400ms @ 32k)
        silence_samples = int(0.40 * 32000)
        original_shape = audio.shape
        flat_audio = audio.flatten()
        silence = np.zeros(silence_samples, dtype=audio.dtype)
        padded_audio = np.concatenate([flat_audio, silence])
        
        if len(original_shape) > 1:
            return padded_audio.reshape(original_shape[0], -1)
        return padded_audio


# Module-level singleton
tts_client: LunaVoxEngine = LunaVoxEngine()
