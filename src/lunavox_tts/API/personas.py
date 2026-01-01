# Persona Management API
"""
API for creating and loading Personas for reference-free TTS.
Extracted from _internal.py for modularization.
"""

import os
import logging
from os import PathLike
from typing import Union

from ..Audio.ReferenceAudio import ReferenceAudio
from ..ModelManager import model_manager
from ..Utils.Shared import context
from ..Utils.ResourceManager import resource_manager
from ..Persona.PersonaManager import export_persona, load_persona as persona_loader
from .state import (
    SUPPORTED_AUDIO_EXTS,
    set_reference_audio_config,
)
from .characters import load_character

logger = logging.getLogger(__name__)


def create_persona(
        character_name: str,
        audio_path: Union[str, PathLike],
        audio_text: str,
        save_dir: Union[str, PathLike],
        audio_language: str = None,
) -> str:
    """
    Create and save a Persona from reference audio for reference-free TTS.
    
    After calling this function, you can use `load_persona()` to enable
    TTS without providing the reference audio again.
    
    Args:
        character_name (str): The name of the character.
        audio_path (str | PathLike): Path to the reference audio file.
        audio_text (str): The transcript of the reference audio.
        save_dir (str | PathLike): Directory to save the persona files.
        audio_language (str, optional): Language of the reference audio.
        
    Returns:
        str: Path to the saved persona directory.
        
    Example:
        >>> lunavox.create_persona("klee", "ref.wav", "Hello world", "./personas/klee")
        >>> # Later, without the original audio:
        >>> lunavox.load_persona("klee", "./personas/klee")
        >>> lunavox.tts("klee", "Welcome!")
    """
    audio_path_str = os.fspath(audio_path)
    save_dir_str = os.fspath(save_dir)
    
    # Validate audio format
    ext = os.path.splitext(audio_path_str)[1].lower()
    if ext not in SUPPORTED_AUDIO_EXTS:
        raise ValueError(
            f"Audio format '{ext}' is not supported. "
            f"Supported formats: {SUPPORTED_AUDIO_EXTS}"
        )
    
    # Get model version for the character
    model_version = model_manager.get_character_version(character_name)
    
    # Create ReferenceAudio and export persona in CPU mode to ensure stability/precision
    from ..Utils.EnvManager import env_manager
    with env_manager.temporary_mode("cpu"):
        # Create ReferenceAudio (this extracts all features)
        ref = ReferenceAudio(
            prompt_wav=audio_path_str,
            prompt_text=audio_text,
            language=audio_language or 'auto',
            model_version=model_version,
        )
        
        # Export to persona directory using PersonaManager
        persona_path = export_persona(ref, save_dir_str, character_name, audio_path_str)
    
    logger.info(f"✓ Persona created for '{character_name}' at: {persona_path}")
    
    return persona_path


def load_persona(
        character_name: str,
        persona_dir: Union[str, PathLike],
) -> None:
    """
    Load a previously saved Persona for reference-free TTS.
    
    After calling this function, `tts()` and `tts_async()` will use
    the cached features from the Persona, skipping audio preprocessing.
    
    Args:
        character_name (str): The name of the character.
        persona_dir (str | PathLike): Path to the persona directory.
        
    Raises:
        FileNotFoundError: If the persona directory doesn't exist.
        ValueError: If the persona data is invalid.
        
    Example:
        >>> lunavox.load_persona("klee", "./personas/klee")
        >>> lunavox.tts("klee", "Hello!")  # No reference audio needed
    """
    persona_dir_str = os.fspath(persona_dir)
    
    # --- AUTO-DOWNLOAD BUILT-IN PERSONAS ---
    if not os.path.isdir(persona_dir_str):
        if "luna_en" in persona_dir_str:
            resource_manager.ensure_base()
        elif "luna_zh" in persona_dir_str:
            resource_manager.ensure_chinese()
        elif "luna_ja" in persona_dir_str:
            resource_manager.ensure_japanese()

    if not os.path.isdir(persona_dir_str):
        raise FileNotFoundError(f"Persona directory not found: {persona_dir_str}")
    
    # Load persona using PersonaManager
    ref = persona_loader(persona_dir_str)
    
    # Prioritize loaded model's version over persona metadata
    if model_manager.has_character(character_name):
        model_version = model_manager.get_character_version(character_name)
    else:
        model_version = getattr(ref, 'model_version', 'v2')
    
    # Register in reference audios dict
    set_reference_audio_config(character_name, {
        'persona_dir': persona_dir_str,
        'model_version': model_version,
        'is_persona': True,
    })
    
    # Set as current prompt audio
    context.current_prompt_audio = ref
    
    # --- AUTO-LOAD BASE MODEL ---
    if not model_manager.has_character(character_name):
        model_version_lower = model_version.lower()
        if "v2_pro_plus" in model_version_lower or "v2pp" in model_version_lower:
            base_model_dir = resource_manager.char_data_dir / "model" / "v2_pro_plus" / "pretrained"
        else:
            base_model_dir = resource_manager.char_data_dir / "model" / "v2" / "pretrained"
            
        if base_model_dir.exists():
            logger.info(f"Auto-loading base {model_version} models for persona '{character_name}'...")
            load_character(character_name, base_model_dir)
        else:
            logger.warning(
                f"Base model directory for version '{model_version}' not found at: {base_model_dir}. "
                f"You may need to call 'load_character' manually."
            )
    
    # --- OPTIMIZATION: Warmup & Cleanup ---
    from ..Core.TextFrontend import get_text_frontend
    frontend = get_text_frontend()
    try:
        native_lang = model_version.split('_')[-1] if '_' in model_version else 'en'
        frontend.warmup(language=native_lang) 
        
        if native_lang != 'zh':
            frontend.warmup(language='zh')
    except (ImportError, Exception) as e:
        logger.debug(f"Optional language warmup skipped: {e}")

    model_manager.unload_cn_hubert()
    model_manager.unload_sv_model()
    
    # Optimization: Unload Prompt Encoder if Persona has cached global embeddings
    if ref.global_emb is not None:
        model_manager.unload_prompt_encoder(character_name)
    
    logger.info(f"✓ Persona loaded for '{character_name}' from: {persona_dir_str}")
