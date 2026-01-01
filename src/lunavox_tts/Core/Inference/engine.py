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

    def generate(self, text: str, session: SynthesisSession, stop_event: threading.Event = None) -> Optional[np.ndarray]:
        """
        Main entry point for session-based TTS generation.
        Executes the full pipeline: Preprocess -> Features -> T2S -> Vocoder -> Postprocess.
        
        Args:
            text: Text to synthesize
            session: SynthesisSession with model and audio config
            stop_event: Optional threading.Event for cancellation
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
            
            if stop_event and stop_event.is_set(): return None

            # 3. Pipeline: T2S (Text-to-Semantic)
            semantic_tokens = self.t2s_stage(text_seq, text_bert, session, stop_event)
            if semantic_tokens is None: return None
            
            if stop_event and stop_event.is_set(): return None

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

    def t2s_stage(self, text_seq: np.ndarray, text_bert: np.ndarray, session: SynthesisSession, stop_event: threading.Event = None) -> Optional[np.ndarray]:
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
                stop_event=stop_event,
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
        
        Uses spec-driven input assembly to eliminate hardcoded version checks.
        """
        from ...Core.Model import get_model_spec
        
        prompt_audio = session.prompt_audio
        model_version = session.model_version
        device_mode = env_manager.get_mode()
        
        # Get spec for this model version
        spec = get_model_spec(model_version)
        
        # Handle lazy prompt encoder for v2ProPlus if global_emb not cached
        if spec.requires_global_emb and prompt_audio.global_emb is None:
            if session.model and session.model.PROMPT_ENCODER:
                with monitor.measure("Prompt Encoder"):
                    run_prompt_encoder(session.model.PROMPT_ENCODER, prompt_audio)
            else:
                raise RuntimeError(f"{model_version} requires global_emb or prompt_encoder")
        
        # WORKAROUND: Truncate reference audio for VITS if too long on GPU (v2 only)
        if model_version not in ('v2ProPlus', 'v2Pro', 'v2pp'):
            if device_mode == "gpu" and not prompt_audio.is_persona_based:
                if prompt_audio.audio_32k is not None and len(prompt_audio.audio_32k) > 128000:
                    # Create a copy to avoid mutating the cached reference
                    import copy
                    prompt_audio = copy.copy(prompt_audio)
                    prompt_audio.audio_32k = prompt_audio.audio_32k[:128000]
        
        # Use spec-driven input assembly
        vocoder_inputs = spec.assemble_vocoder_inputs(
            text_seq=text_seq,
            pred_semantic=semantic_tokens,
            features=prompt_audio,
            vocoder_session=session.model.VITS,
        )

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
