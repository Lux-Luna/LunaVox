"""
FeaturePacket - Immutable data container for extracted TTS features.

This dataclass encapsulates all features needed for inference, providing
a clean separation between feature extraction and inference stages.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=False)
class FeaturePacket:
    """
    Container for all extracted features required by the TTS pipeline.
    
    This replaces in-place mutation of ReferenceAudio objects, enabling:
    - Thread-safe feature passing between pipeline stages
    - Clean separation of data storage from processing logic
    - Future async pipeline support with queue-based data transfer
    
    Attributes:
        phonemes_seq: Tokenized phoneme sequence [1, seq_len]
        text_bert: BERT embeddings [seq_len, 1024]
        ssl_content: HuBERT SSL features [1, T, 768]
        sv_emb: Speaker vector embedding (optional, for v2Pro+)
        global_emb: Global embedding (optional, for v2ProPlus)
        global_emb_advanced: Advanced global embedding (optional, for v2ProPlus)
        audio_32k: Reference audio at 32kHz (optional, for vocoder)
        audio_16k: Reference audio at 16kHz (optional, for SSL extraction)
        resolved_language: Detected/resolved language code
        is_persona_based: True if features were loaded from cached Persona
    """
    phonemes_seq: np.ndarray
    text_bert: np.ndarray
    ssl_content: np.ndarray
    sv_emb: Optional[np.ndarray] = None
    global_emb: Optional[np.ndarray] = None
    global_emb_advanced: Optional[np.ndarray] = None
    audio_32k: Optional[np.ndarray] = None
    audio_16k: Optional[np.ndarray] = None
    resolved_language: str = "ja"
    is_persona_based: bool = False
    
    def validate(self) -> bool:
        """Check if packet has minimum required features for inference."""
        return (
            self.phonemes_seq is not None and
            self.text_bert is not None and
            self.ssl_content is not None
        )
    
    @classmethod
    def from_reference_audio(cls, ref_audio: "ReferenceAudio") -> "FeaturePacket":
        """
        Create a FeaturePacket from a ReferenceAudio instance.
        
        This is a migration helper to bridge old code that uses ReferenceAudio
        with new code that expects FeaturePacket.
        """
        return cls(
            phonemes_seq=ref_audio.phonemes_seq,
            text_bert=ref_audio.text_bert,
            ssl_content=ref_audio.ssl_content,
            sv_emb=ref_audio.sv_emb,
            global_emb=ref_audio.global_emb,
            global_emb_advanced=ref_audio.global_emb_advanced,
            audio_32k=ref_audio.audio_32k,
            audio_16k=ref_audio.audio_16k,
            resolved_language=getattr(ref_audio, 'resolved_language', 'ja'),
            is_persona_based=getattr(ref_audio, '_is_persona_based', False),
        )
