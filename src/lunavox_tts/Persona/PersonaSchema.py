"""
Persona Schema - Data structures for Feature Space Solidification.

This module defines the data structures used to serialize and deserialize
cached TTS features (Personas). A Persona allows reference-free TTS by
storing all pre-computed features from a reference audio.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class PersonaMetadata:
    """
    Stores metadata for the serialized persona.
    
    This information is saved as JSON and allows validation
    and debugging without loading the full feature tensors.
    """
    character_name: str
    language: str
    prompt_text: str
    model_version: str  # v2, v2Pro, v2ProPlus
    created_at: str     # ISO timestamp
    
    # Optional validation fields
    source_audio_md5: Optional[str] = None
    lunavox_version: Optional[str] = None
    
    # Feature shape information for validation
    ssl_content_shape: Optional[tuple] = None
    audio_32k_length: Optional[int] = None


@dataclass
class PersonaFeatures:
    """
    Stores the pre-computed feature tensors for reference-free TTS.
    
    These tensors are what would normally be extracted at runtime
    from reference audio. By caching them, we skip the extraction step.
    """
    # Core features (required for all model versions)
    ssl_content: np.ndarray     # HuBERT features: shape (1, T, D)
    text_bert: np.ndarray       # BERT phone features: shape (N, 1024)
    phonemes_seq: np.ndarray    # Phoneme ID sequence: shape (1, N)
    audio_32k: np.ndarray       # Preprocessed audio for VITS: shape (N,)
    
    # v2Pro/v2ProPlus specific features
    sv_emb: Optional[np.ndarray] = None           # Speaker vector: shape (1, 20480)
    global_emb: Optional[np.ndarray] = None       # Global embedding from Prompt Encoder
    global_emb_advanced: Optional[np.ndarray] = None  # Advanced global embedding
    
    def validate(self) -> bool:
        """
        Validate that all required features have expected shapes.
        
        Returns:
            True if all validations pass, False otherwise.
        """
        # Check core features exist
        if self.ssl_content is None or self.ssl_content.size == 0:
            return False
        if self.phonemes_seq is None or self.phonemes_seq.size == 0:
            return False
        if self.audio_32k is None or self.audio_32k.size == 0:
            return False
        
        # Check shapes are reasonable
        if self.ssl_content.ndim != 3:  # (1, T, D)
            return False
        if self.phonemes_seq.ndim != 2:  # (1, N)
            return False
        
        # Speaker vector shape check for v2Pro/v2ProPlus
        if self.sv_emb is not None and self.sv_emb.shape != (1, 20480):
            return False
        
        return True
