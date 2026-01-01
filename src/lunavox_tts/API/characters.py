# Character Management API
"""
API for loading, unloading, and configuring characters.
Extracted from _internal.py for modularization.
"""

import os
import logging
from os import PathLike
from typing import Optional, Union

from ..Resources.Audio.ReferenceAudio import ReferenceAudio
from ..ModelManager import model_manager
from ..Utils.Shared import context
from ..Utils.ResourceManager import resource_manager
from .state import (
    SUPPORTED_AUDIO_EXTS,
    set_reference_audio_config,
)

logger = logging.getLogger(__name__)


def load_character(
        character_name: str,
        onnx_model_dir: Union[str, PathLike],
        skip_prompt_encoder: bool = False,
) -> None:
    """
    Loads a character model from an ONNX model directory.

    Args:
        character_name (str): The name to assign to the loaded character.
        onnx_model_dir (str | PathLike): The directory path containing the ONNX model files.
        skip_prompt_encoder (bool): If True, skip loading the prompt_encoder.
                                    Useful when using Personas with cached global_emb.
    """
    model_path: str = os.fspath(onnx_model_dir)
    
    # --- AUTO-DOWNLOAD BUILT-IN MODELS ---
    if not os.path.isdir(model_path):
        normalized_path = model_path.replace("\\", "/")
        if "CharacterData/model/v2/pretrained" in normalized_path:
            resource_manager.ensure_base()
        elif "CharacterData/model/v2_pro_plus/pretrained" in normalized_path:
            resource_manager.ensure_v2pp(skip_prompt_encoder=skip_prompt_encoder)

    if not os.path.isdir(model_path):
        logger.error(f"Character model directory not found: {model_path}")
        return

    model_manager.load_character(
        character_name=character_name,
        model_dir=model_path,
        skip_prompt_encoder=skip_prompt_encoder,
    )


def unload_character(
        character_name: str,
) -> None:
    """
    Unloads a previously loaded character model to free up resources.

    Args:
        character_name (str): The name of the character to unload.
    """
    model_manager.remove_character(
        character_name=character_name,
    )


def set_reference_audio(
        character_name: str,
        audio_path: Union[str, PathLike],
        audio_text: str,
        audio_language: Optional[str] = None,
) -> None:
    """
    Sets the reference audio for a character to be used for voice cloning.

    This must be called for a character before using 'tts' or 'tts_async'.

    Args:
        character_name (str): The name of the character.
        audio_path (str | PathLike): The file path to the reference audio (e.g., a WAV file).
        audio_text (str): The transcript of the reference audio.
        audio_language (str, optional): Language of the reference audio.
    """
    audio_path_str: str = os.fspath(audio_path)

    # 检查文件后缀是否支持
    ext = os.path.splitext(audio_path_str)[1].lower()
    if ext not in SUPPORTED_AUDIO_EXTS:
        logger.error(
            f"Audio format '{ext}' is not supported. Only the following formats are supported: {SUPPORTED_AUDIO_EXTS}"
        )
        return

    # Get model version for the character
    model_version = model_manager.get_character_version(character_name)

    ref = ReferenceAudio(
        prompt_wav=audio_path_str,
        prompt_text=audio_text,
        language=audio_language or 'auto',
        model_version=model_version,
    )

    set_reference_audio_config(character_name, {
        'audio_path': audio_path_str,
        'audio_text': audio_text,
        'audio_lang': audio_language,
        'model_version': model_version,
        'prompt_audio': ref,
    })
