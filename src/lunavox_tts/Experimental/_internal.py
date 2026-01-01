# _internal.py - Backward Compatibility Layer
"""
This module re-exports all public API functions from the new modular API package.
It exists solely for backward compatibility with existing code that imports from _internal.

For new code, import directly from lunavox_tts.API instead.
"""

# 请严格遵循导入顺序。
# 1、环境变量。
import os

os.environ["HF_HUB_ENABLE_PROGRESS_BAR"] = "1"

# 2、Logging。
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]"
)

# 3、ONNX。
import onnxruntime

onnxruntime.set_default_logger_severity(3)

# Re-export all public API functions from the modular API package
from .API import (
    load_character,
    unload_character,
    set_reference_audio,
    create_persona,
    load_persona,
    tts,
    tts_async,
    stop,
    convert_to_onnx,
    clear_reference_audio_cache,
    launch_command_line_client,
    load_predefined_character,
)

__all__ = [
    "load_character",
    "unload_character",
    "set_reference_audio",
    "create_persona",
    "load_persona",
    "tts",
    "tts_async",
    "stop",
    "convert_to_onnx",
    "clear_reference_audio_cache",
    "launch_command_line_client",
    "load_predefined_character",
]
