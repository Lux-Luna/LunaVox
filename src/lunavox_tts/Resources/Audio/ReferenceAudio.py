import os
import logging
from typing import Optional, TYPE_CHECKING

import numpy as np

from ..Audio.Audio import load_audio
from ...Utils.Utils import LRUCacheDict

if TYPE_CHECKING:
    import onnxruntime

logger = logging.getLogger(__name__)


class ReferenceAudio:
    """
    Lightweight data container for reference audio and its extracted features.
    
    This class acts as a DTO (Data Transfer Object). Actual feature extraction
    is handled by Core.Processors.FeatureExtractor.
    """
    
    _prompt_cache: dict[tuple[str, str, str], "ReferenceAudio"] = LRUCacheDict(
        capacity=int(os.getenv("Max_Cached_Reference_Audio", "5"))
    )
    
    # Persona support: indicates if this instance was loaded from cached features
    _is_persona_based: bool = False

    @classmethod
    def from_persona(cls, persona_dir: str) -> "ReferenceAudio":
        """Load ReferenceAudio from a Persona directory."""
        from ..Resources.Persona.PersonaManager import PersonaManager
        return PersonaManager.load(persona_dir)

    def export_persona(self, save_dir: str, character_name: str, source_audio_path: Optional[str] = None) -> str:
        """Export current features to a Persona directory."""
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
                instance.text = prompt_text
                instance.language = language
                instance._invalidate_features()
            return instance

        instance = super().__new__(cls)
        cls._prompt_cache[key] = instance
        return instance

    def __init__(self, prompt_wav: str, prompt_text: str, language: str = "auto", model_version: str = 'v2'):
        if hasattr(self, "_initialized"):
            return

        self.wav_path: str = prompt_wav
        self.text: str = prompt_text
        self.language: str = language or "auto"
        self.model_version: str = model_version
        
        # Features (lazily populated by FeatureExtractor)
        self.phonemes_seq: Optional[np.ndarray] = None
        self.text_bert: Optional[np.ndarray] = None
        self.sv_emb: Optional[np.ndarray] = None
        self.global_emb: Optional[np.ndarray] = None
        self.global_emb_advanced: Optional[np.ndarray] = None
        self.ssl_content: Optional[np.ndarray] = None
        self.resolved_language: Optional[str] = None
        
        # Audio buffers
        self.audio_32k: Optional[np.ndarray] = None
        self.audio_16k: Optional[np.ndarray] = None

        self._initialized = True

    def _load_audio(self):
        """Internal method to load audio data."""
        if self.audio_32k is not None:
            return
            
        self.audio_32k = load_audio(
            audio_path=self.wav_path,
            target_sampling_rate=32000,
        )
        if self.audio_32k is not None and np.isnan(self.audio_32k).any():
            logger.warning(f"NaNs detected in loaded audio: {self.wav_path}. Replacing with zeros.")
            self.audio_32k = np.nan_to_num(self.audio_32k)

    def _invalidate_features(self):
        """Clear extracted features when text or language changes."""
        self.phonemes_seq = None
        self.text_bert = None
        self.ssl_content = None
        self.sv_emb = None
        self.global_emb = None
        self.global_emb_advanced = None
        self.resolved_language = None

    @classmethod
    def clear_cache(cls) -> None:
        cls._prompt_cache.clear()

    def update_global_emb(self, prompt_encoder: "onnxruntime.InferenceSession") -> None:
        """Deprecated: Use FeatureExtractor instead."""
        from ...Core.Processors.feature_extractor import feature_extractor
        feature_extractor.extract_global_emb(self, prompt_encoder)


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
