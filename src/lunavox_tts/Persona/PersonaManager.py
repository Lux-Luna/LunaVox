"""
Persona Manager - Core logic for Feature Space Solidification.

This module handles the export and import of cached TTS features (Personas).
It enables reference-free TTS by pre-computing and storing all features
that would normally be extracted from reference audio at runtime.
"""
import os
import json
import logging
import hashlib
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import numpy as np

from .PersonaSchema import PersonaMetadata, PersonaFeatures

if TYPE_CHECKING:
    from ..Audio.ReferenceAudio import ReferenceAudio

logger = logging.getLogger(__name__)

# File names for persona storage
METADATA_FILE = "metadata.json"
FEATURES_FILE = "features.npz"


class PersonaManager:
    """
    Manages the export and import of Persona files.
    
    A Persona is a cached set of features extracted from reference audio,
    enabling reference-free TTS by skipping the audio preprocessing step.
    """
    
    @staticmethod
    def export(
        ref_audio: "ReferenceAudio",
        save_dir: str,
        character_name: str,
        source_audio_path: Optional[str] = None,
    ) -> str:
        """
        Export a ReferenceAudio's features to a Persona directory.
        
        Args:
            ref_audio: The ReferenceAudio instance to export.
            save_dir: Directory path to save the persona files.
            character_name: Name of the character.
            source_audio_path: Optional path to source audio for MD5 validation.
            
        Returns:
            The path to the saved persona directory.
            
        Raises:
            ValueError: If required features are missing from ref_audio.
        """
        # Validate required features
        if ref_audio.ssl_content is None:
            raise ValueError("Cannot export persona: ssl_content is None")
        if ref_audio.phonemes_seq is None:
            raise ValueError("Cannot export persona: phonemes_seq is None")
        if ref_audio.audio_32k is None:
            raise ValueError("Cannot export persona: audio_32k is None")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Compute source audio MD5 if path provided
        source_md5 = None
        if source_audio_path and os.path.exists(source_audio_path):
            try:
                with open(source_audio_path, "rb") as f:
                    source_md5 = hashlib.md5(f.read()).hexdigest()
            except Exception as e:
                logger.warning(f"Failed to compute source audio MD5: {e}")
        
        # Determine supported versions based on available features
        supported_versions = ["v2"]  # v2 is always supported
        has_global_emb = ref_audio.global_emb is not None
        if has_global_emb:
            supported_versions.append("v2ProPlus")
        
        # Build metadata
        metadata = PersonaMetadata(
            character_name=character_name,
            language=ref_audio.language,
            prompt_text=ref_audio.text,
            supported_versions=supported_versions,
            created_at=datetime.utcnow().isoformat() + "Z",
            source_audio_md5=source_md5,
            lunavox_version=_get_lunavox_version(),
            ssl_content_shape=tuple(ref_audio.ssl_content.shape),
            audio_32k_length=len(ref_audio.audio_32k),
            has_global_emb=has_global_emb,
        )
        
        # Save metadata as JSON
        metadata_path = os.path.join(save_dir, METADATA_FILE)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({
                "character_name": metadata.character_name,
                "language": metadata.language,
                "prompt_text": metadata.prompt_text,
                "supported_versions": metadata.supported_versions,
                "created_at": metadata.created_at,
                "source_audio_md5": metadata.source_audio_md5,
                "lunavox_version": metadata.lunavox_version,
                "ssl_content_shape": metadata.ssl_content_shape,
                "audio_32k_length": metadata.audio_32k_length,
                "has_global_emb": metadata.has_global_emb,
            }, f, ensure_ascii=False, indent=2)
        
        # Build feature dictionary
        features_dict = {
            "ssl_content": ref_audio.ssl_content,
            "text_bert": ref_audio.text_bert if ref_audio.text_bert is not None else np.array([]),
            "phonemes_seq": ref_audio.phonemes_seq,
            "audio_32k": ref_audio.audio_32k,
        }
        
        # Add v2Pro/v2ProPlus specific features
        if ref_audio.sv_emb is not None:
            features_dict["sv_emb"] = ref_audio.sv_emb
        if ref_audio.global_emb is not None:
            # Handle OrtValue objects by converting to numpy
            if hasattr(ref_audio.global_emb, 'numpy'):
                features_dict["global_emb"] = ref_audio.global_emb.numpy()
            else:
                features_dict["global_emb"] = ref_audio.global_emb
        if ref_audio.global_emb_advanced is not None:
            if hasattr(ref_audio.global_emb_advanced, 'numpy'):
                features_dict["global_emb_advanced"] = ref_audio.global_emb_advanced.numpy()
            else:
                features_dict["global_emb_advanced"] = ref_audio.global_emb_advanced
        
        # Save features as compressed NPZ
        features_path = os.path.join(save_dir, FEATURES_FILE)
        np.savez_compressed(features_path, **features_dict)
        
        logger.info(f"✓ Persona exported to: {save_dir}")
        logger.debug(f"  - SSL content shape: {ref_audio.ssl_content.shape}")
        logger.debug(f"  - Audio length: {len(ref_audio.audio_32k)} samples")
        logger.debug(f"  - Supported versions: {metadata.supported_versions}")
        
        return save_dir
    
    @staticmethod
    def load(persona_dir: str) -> "ReferenceAudio":
        """
        Load a Persona directory into a ReferenceAudio instance.
        
        This creates a ReferenceAudio WITHOUT loading any wav file,
        using the cached features directly.
        
        Args:
            persona_dir: Path to the persona directory.
            
        Returns:
            A ReferenceAudio instance with all features pre-loaded.
            
        Raises:
            FileNotFoundError: If persona files are missing.
            ValueError: If persona data is invalid.
        """
        metadata_path = os.path.join(persona_dir, METADATA_FILE)
        features_path = os.path.join(persona_dir, FEATURES_FILE)
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Persona metadata not found: {metadata_path}")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Persona features not found: {features_path}")
        
        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        
        # Handle legacy metadata format (model_version -> supported_versions)
        if "supported_versions" in meta_dict:
            supported_versions = meta_dict["supported_versions"]
        else:
            # Legacy format: convert single model_version to list
            legacy_version = meta_dict.get("model_version", "v2")
            supported_versions = ["v2"]
            if legacy_version in ["v2ProPlus", "v2pp"]:
                supported_versions.append("v2ProPlus")
        
        metadata = PersonaMetadata(
            character_name=meta_dict["character_name"],
            language=meta_dict["language"],
            prompt_text=meta_dict["prompt_text"],
            supported_versions=supported_versions,
            created_at=meta_dict["created_at"],
            source_audio_md5=meta_dict.get("source_audio_md5"),
            lunavox_version=meta_dict.get("lunavox_version"),
            ssl_content_shape=tuple(meta_dict.get("ssl_content_shape", [])),
            audio_32k_length=meta_dict.get("audio_32k_length"),
            has_global_emb=meta_dict.get("has_global_emb", False),
        )
        
        # Load features
        with np.load(features_path, allow_pickle=False) as data:
            features = PersonaFeatures(
                ssl_content=data["ssl_content"],
                text_bert=data["text_bert"] if data["text_bert"].size > 0 else None,
                phonemes_seq=data["phonemes_seq"],
                audio_32k=data["audio_32k"],
                sv_emb=data.get("sv_emb"),
                global_emb=data.get("global_emb"),
                global_emb_advanced=data.get("global_emb_advanced"),
            )
        
        # Validate features
        if not features.validate():
            raise ValueError(f"Persona features validation failed for: {persona_dir}")
        
        # Create a ReferenceAudio instance bypassing normal __init__
        ref_audio = _create_reference_audio_from_features(metadata, features)
        
        logger.info(f"✓ Persona loaded from: {persona_dir}")
        logger.debug(f"  - Character: {metadata.character_name}")
        logger.debug(f"  - Supported versions: {metadata.supported_versions}")
        logger.debug(f"  - Has global_emb: {metadata.has_global_emb}")
        
        return ref_audio


def _create_reference_audio_from_features(
    metadata: PersonaMetadata,
    features: PersonaFeatures,
) -> "ReferenceAudio":
    """
    Create a ReferenceAudio instance from cached features.
    
    This bypasses the normal __init__ which loads and processes wav files.
    """
    # Import here to avoid circular dependency
    from ..Audio.ReferenceAudio import ReferenceAudio
    
    # Create instance without calling __init__ (bypass wav processing)
    instance = object.__new__(ReferenceAudio)
    
    # Set all attributes manually
    instance.text = metadata.prompt_text
    instance.language = metadata.language
    # Store supported versions for version compatibility checks
    instance.supported_versions = metadata.supported_versions
    # Set model_version for backward compatibility with inference code
    instance.model_version = "v2ProPlus" if "v2ProPlus" in metadata.supported_versions else "v2"
    
    # Set pre-computed features
    instance.ssl_content = features.ssl_content
    instance.text_bert = features.text_bert
    instance.phonemes_seq = features.phonemes_seq
    instance.audio_32k = features.audio_32k
    instance.sv_emb = features.sv_emb
    instance.global_emb = features.global_emb
    instance.global_emb_advanced = features.global_emb_advanced
    
    # Mark as initialized and persona-based
    instance._initialized = True
    instance._is_persona_based = True
    
    return instance


def _get_lunavox_version() -> str:
    """Get LunaVox version string for metadata."""
    try:
        from importlib.metadata import version
        return version("lunavox-tts")
    except Exception:
        return "unknown"


# Convenience functions for direct use
def export_persona(
    ref_audio: "ReferenceAudio",
    save_dir: str,
    character_name: str,
    source_audio_path: Optional[str] = None,
) -> str:
    """Export a ReferenceAudio to a Persona directory."""
    return PersonaManager.export(ref_audio, save_dir, character_name, source_audio_path)


def load_persona(persona_dir: str) -> "ReferenceAudio":
    """Load a Persona directory into a ReferenceAudio instance."""
    return PersonaManager.load(persona_dir)
