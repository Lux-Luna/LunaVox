import os
from typing import Optional

import numpy as np
import soxr

from ..Audio.Audio import load_audio
from ...Core.Frontend import get_language_frontend
from ...ModelManager import model_manager
from ...Utils.Shared import context
from ...Utils.Utils import LRUCacheDict

# Import SV extraction for v2Pro/v2ProPlus
try:
    from .SpeakerVector import extract_sv_embedding
    _SV_AVAILABLE = True
except ImportError:
    _SV_AVAILABLE = False


class ReferenceAudio:
    _prompt_cache: dict[tuple[str, str, str], "ReferenceAudio"] = LRUCacheDict(
        capacity=int(os.getenv("Max_Cached_Reference_Audio", "5"))
    )
    
    # Persona support: indicates if this instance was loaded from cached features
    _is_persona_based: bool = False

    @classmethod
    def from_persona(cls, persona_dir: str) -> "ReferenceAudio":
        """
        Create a ReferenceAudio from cached Persona features (no wav required).
        
        This enables reference-free TTS by loading pre-computed features
        instead of extracting them from audio at runtime.
        
        Args:
            persona_dir: Path to the persona directory containing metadata.json and features.npz.
            
        Returns:
            A ReferenceAudio instance with all features pre-loaded.
        """
        from ..Resources.Persona.PersonaManager import PersonaManager
        return PersonaManager.load(persona_dir)

    def export_persona(self, save_dir: str, character_name: str, source_audio_path: Optional[str] = None) -> str:
        """
        Export current features to a Persona directory for reference-free TTS.
        
        Args:
            save_dir: Directory path to save the persona files.
            character_name: Name of the character.
            source_audio_path: Optional path to source audio for MD5 validation.
            
        Returns:
            The path to the saved persona directory.
        """
        from ..Resources.Persona.PersonaManager import PersonaManager
        return PersonaManager.export(self, save_dir, character_name, source_audio_path)

    @property
    def is_persona_based(self) -> bool:
        """Returns True if this instance was loaded from a Persona (no wav processing)."""
        return getattr(self, '_is_persona_based', False)

    def __new__(cls, prompt_wav: str, prompt_text: str, language: str = "auto", model_version: str = 'v2'):
        # Cache key includes model_version to avoid conflicts
        key = (prompt_wav, (language or "auto"), model_version)
        if key in cls._prompt_cache:
            instance = cls._prompt_cache[key]
            if instance.text != prompt_text or instance.language != language:
                instance.set_text(prompt_text, language)
            return instance

        instance = super().__new__(cls)
        cls._prompt_cache[key] = instance
        return instance

    def __init__(self, prompt_wav: str, prompt_text: str, language: str = "auto", model_version: str = 'v2'):
        if hasattr(self, "_initialized"):
            return

        self.text: str = prompt_text
        self.language: str = language or "auto"
        self.model_version: str = model_version
        self.phonemes_seq: Optional[np.ndarray] = None
        self.text_bert: Optional[np.ndarray] = None
        self.sv_emb: Optional[np.ndarray] = None
        self.global_emb: Optional[np.ndarray] = None
        self.global_emb_advanced: Optional[np.ndarray] = None
        self.set_text(prompt_text, language)

        self.audio_32k: Optional[np.ndarray] = load_audio(
            audio_path=prompt_wav,
            target_sampling_rate=32000,
        )
        # Check for NaNs immediately after loading
        if self.audio_32k is not None and np.isnan(self.audio_32k).any():
            import logging
            logging.getLogger(__name__).warning(f"NaNs detected in loaded audio: {prompt_wav}. Replacing with zeros.")
            self.audio_32k = np.nan_to_num(self.audio_32k)

        audio_16k: np.ndarray = soxr.resample(self.audio_32k, 32000, 16000, quality="hq")
        # Check NaNs after resampling
        if np.isnan(audio_16k).any():
             audio_16k = np.nan_to_num(audio_16k)
        
        # Ensure extractor resources are available (HuBERT, SV)
        from ...Utils.ResourceManager import resource_manager
        resource_manager.ensure_extractor()
        
        # Extract SSL content (always needed)
        audio_16k_batch = np.expand_dims(audio_16k, axis=0)
        if not model_manager.cn_hubert:
            model_manager.load_cn_hubert()
        self.ssl_content: Optional[np.ndarray] = model_manager.cn_hubert.run(
            None, {"input_values": audio_16k_batch}
        )[0]

        
        # Extract speaker vector (Always extract for full version compatibility in Personas)
        if _SV_AVAILABLE:
            self.sv_emb = extract_sv_embedding(audio_16k)  # Pass 1D array
            if self.sv_emb is None:
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to extract speaker embedding. "
                    f"v2ProPlus models will not work correctly."
                )
            else:
                # Validate SV embedding shape
                import logging
                logger = logging.getLogger(__name__)
                if self.sv_emb.shape != (1, 20480):
                    logger.error(
                        f"Invalid speaker embedding shape: {self.sv_emb.shape}, expected (1, 20480). "
                    )
                    raise ValueError(f"Speaker embedding shape mismatch: {self.sv_emb.shape} != (1, 20480)")
                logger.debug(f"✓ Speaker embedding extracted: shape={self.sv_emb.shape}")
        else:
            import logging
            logging.getLogger(__name__).warning(
                f"Speaker vector extraction is not available. v2ProPlus features will be skipped."
            )

        self._initialized = True

        # Optimization: Clear audio_32k to save memory if not needed for v2ProPlus global embedding update
        # For v2, audio_32k is required for VITS inference (ref_audio).
        # For v2Pro, audio_32k is required for STFT extraction (ref_audio).
        # For v2ProPlus, audio_32k is required for Prompt Encoder (global_emb).
        # We cannot clear it immediately in __init__ as it's needed for inference.
        # Ideally we clear it after feature extraction, but ReferenceAudio doesn't control that lifecycle.
        
    def set_text(self, prompt_text: str, language: str = "auto") -> None:
        self.text = prompt_text
        lang = _decide_language(prompt_text, language)
        self.language = lang
        
        frontend = get_language_frontend(lang)
        ids, bert_features = frontend.process(prompt_text)
        
        self.phonemes_seq = np.array([ids], dtype=np.int64)
        self.text_bert = bert_features

        if lang in {"ja", "en", "zh"}:
            context.current_language = lang

    @classmethod
    def clear_cache(cls) -> None:
        cls._prompt_cache.clear()

    def update_global_emb(self, prompt_encoder: "onnxruntime.InferenceSession") -> None:
        """Extract global embeddings for v2ProPlus using Prompt Encoder."""
        if self.global_emb is not None:
            return
        
        if self.sv_emb is None:
            import logging
            logging.getLogger(__name__).warning("sv_emb is None, cannot update global_emb")
            return

        try:
            # Ensure audio_32k has batch dimension (1, N)
            audio_input = self.audio_32k
            if audio_input.ndim == 1:
                audio_input = np.expand_dims(audio_input, axis=0)

            # Prompt Encoder expects: ref_audio (B, N), sv_emb (B, 20480)
            # Output: ge (B, 512), ge_advanced (B, 512, 1) or similar
            self.global_emb, self.global_emb_advanced = prompt_encoder.run(None, {
                'ref_audio': audio_input.astype(np.float32),
                'sv_emb': self.sv_emb.astype(np.float32),
            })
            import logging
            logging.getLogger(__name__).debug(f"✓ Global embeddings updated: ge={self.global_emb.shape}, ge_adv={self.global_emb_advanced.shape}")
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to update global embeddings: {e}")


def _decide_language(text: str, language: Optional[str]) -> str:
    lang = (language or "auto").lower()
    if lang == "auto":
        if _looks_english(text):
            return "en"
        if _looks_chinese(text):
            return "zh"
        return "ja"
    if lang in {"ja", "en", "zh"}:
        return lang
    return "ja"





def _looks_english(text: str) -> bool:
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    non_ascii = sum(not ch.isascii() and not ch.isspace() for ch in text)
    return ascii_letters > 0 and ascii_letters >= non_ascii


def _looks_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


