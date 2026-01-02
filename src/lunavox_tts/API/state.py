# API State Management
"""
Runtime state management for LunaVox TTS.
Extracted from _internal.py for modularization.
"""

from typing import Dict, Any, Optional

# A module-level private dictionary to store reference audio configurations.
_reference_audios: Dict[str, Dict[str, Any]] = {}

# Supported audio extensions for reference audio
SUPPORTED_AUDIO_EXTS = {'.wav', '.flac', '.ogg', '.aiff', '.aif'}


def normalize_language(code: Optional[str]) -> str:
    """Normalize language code to supported values."""
    lang = (code or "ja").lower()
    return lang if lang in {"ja", "en", "zh"} else "ja"


def get_reference_audio(character_name: str) -> Optional[Dict[str, Any]]:
    """Get reference audio configuration for a character."""
    return _reference_audios.get(character_name)


def set_reference_audio_config(character_name: str, config: Dict[str, Any]) -> None:
    """Set reference audio configuration for a character."""
    _reference_audios[character_name] = config


def has_reference_audio(character_name: str) -> bool:
    """Check if a character has reference audio configured."""
    return character_name in _reference_audios


def remove_reference_audio(character_name: str) -> bool:
    """
    Remove reference audio configuration for a specific character.
    
    Args:
        character_name: Name of the character to remove.
        
    Returns:
        True if the character was found and removed, False otherwise.
    """
    if character_name in _reference_audios:
        del _reference_audios[character_name]
        return True
    return False


def clear_all_reference_audio() -> None:
    """Clear all reference audio configurations."""
    _reference_audios.clear()
