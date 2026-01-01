# Persona Management API
"""
API for creating and loading Personas for reference-free TTS.
Extracted from _internal.py for modularization.
"""

import os
import logging
from os import PathLike
from typing import Union

from ..Resources.Audio.ReferenceAudio import ReferenceAudio
from ..ModelManager import model_manager
from ..Utils.ResourceManager import resource_manager
from ..Resources.Persona.PersonaManager import export_persona, load_persona as persona_loader
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
    Create and save a Universal Persona from reference audio for reference-free TTS.
    
    The resulting Persona supports both v2 and v2ProPlus models. When used with
    v2ProPlus models, the prompt_encoder is not needed as global embeddings are
    pre-computed and stored in the Persona.
    
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
    
    # Create Universal Persona in CPU mode for maximum precision
    from ..Utils.EnvManager import env_manager
    with env_manager.temporary_mode("cpu"):
        # Ensure all extractor resources (HuBERT + SV + PromptEncoder)
        resource_manager.ensure_extractor()
        
        # Create ReferenceAudio with v2ProPlus to extract all features
        ref = ReferenceAudio(
            prompt_wav=audio_path_str,
            prompt_text=audio_text,
            language=audio_language or 'auto',
            model_version="v2ProPlus",  # Request full feature extraction
        )
        
        # Compute global embeddings for v2ProPlus compatibility
        resource_manager.ensure_character_data(v2pp=True)
        prompt_encoder_path = (
            resource_manager.char_data_dir / "model" / "v2_pro_plus" / "pretrained" / "prompt_encoder_fp32.onnx"
        )
        
        if prompt_encoder_path.exists():
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            prompt_encoder = ort.InferenceSession(
                str(prompt_encoder_path),
                providers=["CPUExecutionProvider"],
                sess_options=sess_options
            )
            logger.info("Computing global embeddings for v2ProPlus compatibility...")
            from ..Core.Processors.feature_extractor import feature_extractor
            feature_extractor.extract_global_emb(ref, prompt_encoder)
            if ref.global_emb is not None:
                logger.info(f"✓ Global embeddings computed: ge={ref.global_emb.shape}")
            else:
                logger.warning("Failed to compute global embeddings. Persona will only support v2 models.")
        else:
            logger.warning(f"prompt_encoder not found at {prompt_encoder_path}. Persona will only support v2 models.")
        
        # Export to persona directory using PersonaManager
        persona_path = export_persona(ref, save_dir_str, character_name, audio_path_str)
    
    logger.info(f"✓ Universal Persona created for '{character_name}' at: {persona_path}")
    
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
        'prompt_audio': ref  # Store the actual ReferenceAudio object
    })
    
    # --- AUTO-LOAD BASE MODEL ---
    # Check if persona has cached global embeddings (enables skipping prompt_encoder)
    has_cached_ge = ref.global_emb is not None
    
    if not model_manager.has_character(character_name):
        model_version_lower = model_version.lower()
        if "v2_pro_plus" in model_version_lower or "v2pp" in model_version_lower or "v2proplus" in model_version_lower:
            base_model_dir = resource_manager.char_data_dir / "model" / "v2_pro_plus" / "pretrained"
        else:
            base_model_dir = resource_manager.char_data_dir / "model" / "v2" / "pretrained"
            
        logger.info(f"Auto-loading base {model_version} models for persona '{character_name}'...")
        # Skip prompt_encoder if persona has cached global embeddings
        # load_character will handle downloading if the directory doesn't exist
        load_character(character_name, base_model_dir, skip_prompt_encoder=has_cached_ge)
    
    # --- OPTIMIZATION: Warmup & Cleanup ---
    from ..Core.Frontend import get_language_frontend
    try:
        native_lang = model_version.split('_')[-1] if '_' in model_version else 'en'
        # Warmup: access the frontend to pre-load resources
        _ = get_language_frontend(native_lang)
        
        if native_lang != 'zh':
            _ = get_language_frontend('zh')
    except (ImportError, ValueError) as e:
        logger.debug(f"Optional language warmup skipped: {e}")

    model_manager.unload_cn_hubert()
    model_manager.unload_sv_model()
    
    # Optimization: Unload Prompt Encoder if Persona has cached global embeddings
    if ref.global_emb is not None:
        model_manager.unload_prompt_encoder(character_name)
    
    logger.info(f"✓ Persona loaded for '{character_name}' from: {persona_dir_str}")
