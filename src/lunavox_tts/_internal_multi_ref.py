"""
Multi-reference audio support for v2Pro/v2ProPlus models.

This module provides utilities for handling multiple reference audios and averaging
their speaker vector embeddings for more stable timbre control.
"""
import os
from os import PathLike
from typing import Union, List, Optional
import logging

from .Audio.ReferenceAudio import ReferenceAudio
from .ModelManager import model_manager

try:
    from .Audio.SpeakerVector import average_sv_embeddings
    _SV_AVERAGING_AVAILABLE = True
except ImportError:
    _SV_AVERAGING_AVAILABLE = False

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTS = {'.wav', '.flac', '.ogg', '.aiff', '.aif'}


def create_multi_reference_audio(
    character_name: str,
    audio_paths: List[Union[str, PathLike]],
    audio_texts: List[str],
    audio_languages: Optional[List[str]] = None,
) -> Optional[ReferenceAudio]:
    """
    Create a reference audio with averaged speaker vectors from multiple reference audios.
    
    This is useful for v2Pro/v2ProPlus models to achieve more stable timbre by averaging
    speaker embeddings from multiple reference samples.
    
    Args:
        character_name: Name of the character
        audio_paths: List of paths to reference audio files
        audio_texts: List of transcripts corresponding to each audio
        audio_languages: Optional list of languages for each audio
        
    Returns:
        ReferenceAudio with averaged speaker vector, or None if failed
    """
    if not audio_paths or not audio_texts:
        logger.error("audio_paths and audio_texts must not be empty")
        return None
    
    if len(audio_paths) != len(audio_texts):
        logger.error("audio_paths and audio_texts must have the same length")
        return None
    
    # Get model version
    model_version = model_manager.get_character_version(character_name)
    
    if model_version not in ['v2Pro', 'v2ProPlus']:
        logger.warning(
            f"Multi-reference audio is designed for v2Pro/v2ProPlus, "
            f"but character {character_name} uses {model_version}. "
            f"Using first reference only."
        )
        audio_paths = audio_paths[:1]
        audio_texts = audio_texts[:1]
        if audio_languages:
            audio_languages = audio_languages[:1]
    
    # Validate all audio paths
    for i, audio_path in enumerate(audio_paths):
        audio_path_str = os.fspath(audio_path)
        ext = os.path.splitext(audio_path_str)[1].lower()
        if ext not in SUPPORTED_AUDIO_EXTS:
            logger.error(
                f"Audio {i+1} format '{ext}' is not supported. "
                f"Supported: {SUPPORTED_AUDIO_EXTS}"
            )
            return None
        if not os.path.exists(audio_path_str):
            logger.error(f"Audio {i+1} not found: {audio_path_str}")
            return None
    
    # Prepare languages
    if audio_languages is None:
        audio_languages = ['auto'] * len(audio_paths)
    elif len(audio_languages) != len(audio_paths):
        logger.warning("audio_languages length mismatch, using 'auto' for all")
        audio_languages = ['auto'] * len(audio_paths)
    
    # Create individual reference audios
    ref_audios: List[ReferenceAudio] = []
    for i, (path, text, lang) in enumerate(zip(audio_paths, audio_texts, audio_languages)):
        try:
            ref_audio = ReferenceAudio(
                prompt_wav=os.fspath(path),
                prompt_text=text,
                language=lang,
                model_version=model_version,
            )
            ref_audios.append(ref_audio)
            logger.info(f"Loaded reference audio {i+1}/{len(audio_paths)}")
        except Exception as e:
            logger.error(f"Failed to load reference audio {i+1}: {e}")
            return None
    
    if not ref_audios:
        logger.error("No reference audios loaded successfully")
        return None
    
    # For v2, just return the first reference
    if model_version == 'v2':
        return ref_audios[0]
    
    # For v2Pro/v2ProPlus, average speaker vectors
    if not _SV_AVERAGING_AVAILABLE:
        logger.warning("Speaker vector averaging not available, using first reference only")
        return ref_audios[0]
    
    # Extract all speaker vectors
    sv_embs = [ref.sv_emb for ref in ref_audios if ref.sv_emb is not None]
    
    if not sv_embs:
        logger.warning("No speaker vectors extracted, using first reference")
        return ref_audios[0]
    
    if len(sv_embs) < len(ref_audios):
        logger.warning(
            f"Only {len(sv_embs)}/{len(ref_audios)} speaker vectors extracted successfully"
        )
    
    # Average speaker vectors
    averaged_sv = average_sv_embeddings(sv_embs)
    
    if averaged_sv is None:
        logger.error("Failed to average speaker vectors")
        return ref_audios[0]
    
    logger.info(f"✓ Averaged speaker vectors from {len(sv_embs)} references")
    
    # Use first reference audio as base, but replace sv_emb with average
    base_ref = ref_audios[0]
    base_ref.sv_emb = averaged_sv
    
    return base_ref


def set_multi_reference_audio(
    character_name: str,
    audio_paths: List[Union[str, PathLike]],
    audio_texts: List[str],
    audio_languages: Optional[List[str]] = None,
) -> bool:
    """
    Set multiple reference audios for a character (v2Pro/v2ProPlus).
    
    Speaker vectors from all references will be averaged for more stable timbre.
    
    Args:
        character_name: Name of the character
        audio_paths: List of paths to reference audio files
        audio_texts: List of transcripts corresponding to each audio
        audio_languages: Optional list of languages for each audio
        
    Returns:
        True if successful, False otherwise
    """
    ref_audio = create_multi_reference_audio(
        character_name, audio_paths, audio_texts, audio_languages
    )
    
    if ref_audio is None:
        return False
    
    # Store in module-level dict (imported from _internal.py)
    from . import _internal
    
    model_version = model_manager.get_character_version(character_name)
    _internal._reference_audios[character_name] = {
        'audio_path': audio_paths[0],  # Store first path for compatibility
        'audio_text': audio_texts[0],
        'audio_lang': audio_languages[0] if audio_languages else 'auto',
        'model_version': model_version,
        'multi_ref': True,
        'num_refs': len(audio_paths),
    }
    _internal.context.current_prompt_audio = ref_audio
    
    logger.info(
        f"✓ Set {len(audio_paths)} reference audios for {character_name} "
        f"(model: {model_version})"
    )
    
    return True

