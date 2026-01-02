# LunaVox API Package
"""
Public API for LunaVox TTS.

This package provides a clean, modular interface for:
- Character management (load/unload characters, set reference audio)
- Persona management (create/load personas for reference-free TTS)
- TTS synthesis (sync and async)
- Utilities (conversion, cache management)
"""

from .characters import (
    load_character,
    unload_character,
    set_reference_audio,
)
from .personas import (
    create_persona,
    load_persona,
)
from .synthesis import (
    tts,
    tts_async,
    stop,
)
from .utilities import (
    convert_to_onnx,
    clear_reference_audio_cache,
    launch_command_line_client,
)
from .facade import initialize_tts

__all__ = [
    # Facade
    "initialize_tts",
    # Characters
    "load_character",
    "unload_character",
    "set_reference_audio",
    # Personas
    "create_persona",
    "load_persona",
    # Synthesis
    "tts",
    "tts_async",
    "stop",
    # Utilities
    "convert_to_onnx",
    "clear_reference_audio_cache",
    "launch_command_line_client",
]
