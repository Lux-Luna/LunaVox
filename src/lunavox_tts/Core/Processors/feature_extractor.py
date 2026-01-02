"""
FeatureExtractor Service.

Responsible for extracting acoustic and linguistic features from ReferenceAudio objects.
Separates data storage (ReferenceAudio) from processing logic.
"""
import logging
import numpy as np
import soxr
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...Resources.Audio.ReferenceAudio import ReferenceAudio
    import onnxruntime

from ...Utils.PerformanceMonitor import monitor
from ...ModelManager import model_manager
from ...Core.Frontend import get_language_frontend
from .FeaturePacket import FeaturePacket

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Service class to extract all necessary features for TTS inference.
    """
    
    @staticmethod
    def process(
        ref_audio: "ReferenceAudio", 
        model_version: str = 'v2', 
        prompt_encoder: Optional["onnxruntime.InferenceSession"] = None
    ) -> FeaturePacket:
        """
        Extract all features for a ReferenceAudio instance based on the target model version.
        
        This method both mutates ref_audio (for backward compatibility) AND returns
        a FeaturePacket for new session-based workflows.
        
        Args:
            ref_audio: The ReferenceAudio instance to process.
            model_version: Target model version ('v2', 'v2Pro', 'v2ProPlus').
            prompt_encoder: Optional prompt encoder session for v2ProPlus.
            
        Returns:
            FeaturePacket containing all extracted features.
        """
        with monitor.measure("Feature Extraction"):
            # 1. Ensure audio is loaded and resampled
            if ref_audio.audio_32k is None:
                 ref_audio._load_audio()
            
            if ref_audio.audio_16k is None:
                ref_audio.audio_16k = soxr.resample(ref_audio.audio_32k, 32000, 16000, quality="hq")
                if np.isnan(ref_audio.audio_16k).any():
                    ref_audio.audio_16k = np.nan_to_num(ref_audio.audio_16k)

            # 2. Text Features (Linguistic)
            if ref_audio.phonemes_seq is None or ref_audio.text_bert is None:
                FeatureExtractor.extract_text_features(ref_audio)

            # 3. SSL Content (Acoustic - HuBERT)
            if ref_audio.ssl_content is None:
                FeatureExtractor.extract_ssl_content(ref_audio)

            # 4. Speaker Vector (Timbre)
            if model_version in ['v2Pro', 'v2ProPlus'] and ref_audio.sv_emb is None:
                FeatureExtractor.extract_sv_embedding(ref_audio)

            # 5. Global Embeddings (v2ProPlus specific)
            if model_version == 'v2ProPlus' and ref_audio.global_emb is None and prompt_encoder is not None:
                FeatureExtractor.extract_global_emb(ref_audio, prompt_encoder)
        
        # Build and return FeaturePacket
        return FeaturePacket.from_reference_audio(ref_audio)

    @staticmethod
    def extract_all(ref_audio: "ReferenceAudio", model_version: str = 'v2') -> FeaturePacket:
        """
        Extract all features and return as FeaturePacket without prompt encoder.
        
        Convenience method for persona creation where prompt_encoder is handled separately.
        """
        return FeatureExtractor.process(ref_audio, model_version, prompt_encoder=None)

    @staticmethod
    def extract_text_features(ref_audio: "ReferenceAudio"):
        """Extract phonemes and BERT features from reference text."""
        from ...Resources.Audio.ReferenceAudio import _decide_language
        lang = _decide_language(ref_audio.text, ref_audio.language)
        frontend = get_language_frontend(lang)
        ids, bert_features = frontend.process(ref_audio.text)
        
        ref_audio.phonemes_seq = np.array([ids], dtype=np.int64)
        ref_audio.text_bert = bert_features
        ref_audio.resolved_language = lang

    @staticmethod
    def extract_ssl_content(ref_audio: "ReferenceAudio"):
        """Extract SSL content using HuBERT."""
        from ...Utils.AssetManager import asset_manager
        asset_manager.ensure_extractor()
        
        if not model_manager.cn_hubert:
            model_manager.load_cn_hubert()
            
        audio_16k_batch = np.expand_dims(ref_audio.audio_16k, axis=0)
        ref_audio.ssl_content = model_manager.cn_hubert.run(
            None, {"input_values": audio_16k_batch}
        )[0]

    @staticmethod
    def extract_sv_embedding(ref_audio: "ReferenceAudio"):
        """Extract speaker vector embedding."""
        from ...Resources.Audio.SpeakerVector import extract_sv_embedding as extract_sv
        
        ref_audio.sv_emb = extract_sv(ref_audio.audio_16k)
        if ref_audio.sv_emb is None:
             logger.warning("Failed to extract speaker embedding.")

    @staticmethod
    def extract_global_emb(ref_audio: "ReferenceAudio", prompt_encoder: "onnxruntime.InferenceSession"):
        """Extract global embeddings for v2ProPlus."""
        if ref_audio.sv_emb is None:
            FeatureExtractor.extract_sv_embedding(ref_audio)
            
        if ref_audio.sv_emb is None:
            return

        try:
            audio_input = ref_audio.audio_32k
            if audio_input.ndim == 1:
                audio_input = np.expand_dims(audio_input, axis=0)

            ref_audio.global_emb, ref_audio.global_emb_advanced = prompt_encoder.run(None, {
                'ref_audio': audio_input.astype(np.float32),
                'sv_emb': ref_audio.sv_emb.astype(np.float32),
            })
        except Exception as e:
            logger.error(f"Failed to extract global embeddings: {e}")

feature_extractor = FeatureExtractor()

