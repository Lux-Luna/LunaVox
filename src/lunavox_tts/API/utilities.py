# Utility API Functions
"""
Miscellaneous utility functions for LunaVox.
Extracted from _internal.py for modularization.
"""

import os
import json
import logging
from os import PathLike
from typing import Union

from ..Resources.Audio.ReferenceAudio import ReferenceAudio
from ..ModelManager import model_manager

from .state import set_reference_audio_config

logger = logging.getLogger(__name__)


def convert_to_onnx(
        torch_ckpt_path: Union[str, PathLike],
        torch_pth_path: Union[str, PathLike],
        output_dir: Union[str, PathLike],
) -> None:
    """
    Converts PyTorch model checkpoints to the ONNX format.
    """
    import sys
    from pathlib import Path
    # Add converter to path if not already there
    converter_path = Path(__file__).parent.parent.parent.parent / "converter"
    if str(converter_path) not in sys.path:
        sys.path.insert(0, str(converter_path.parent))
    from converter import convert
    convert(ckpt_path=torch_ckpt_path, pth_path=torch_pth_path, output_dir=output_dir, format="fp16")


def clear_reference_audio_cache() -> None:
    """
    Clears the cache of reference audio data.
    """
    ReferenceAudio.clear_cache()


def launch_command_line_client() -> None:
    """
    Launch the command-line client.
    """
    from ..Client import Client
    cmd_client: Client = Client()
    cmd_client.run()



