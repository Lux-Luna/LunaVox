# TTS Synthesis API
"""
Core TTS synthesis API functions.
Extracted from _internal.py for modularization.
"""

import os
import asyncio
import logging
from os import PathLike
from typing import AsyncIterator, Optional, Union

from ..Resources.Audio.ReferenceAudio import ReferenceAudio
from ..Core.TTSPlayer import tts_player
from ..ModelManager import model_manager
from ..Utils.Shared import context
from ..Utils.ResourceManager import resource_manager
from .state import (
    normalize_language,
    get_reference_audio,
    has_reference_audio,
)

logger = logging.getLogger(__name__)


async def tts_async(
        character_name: str,
        text: str,
        play: bool = False,
        split_sentence: bool = False,
        save_path: Union[str, PathLike, None] = None,
        language: str = "ja",
) -> AsyncIterator[bytes]:
    """
    Asynchronously generates speech from text and yields audio chunks.

    This function returns an async iterator that provides the audio data in
    real-time as it's being generated.

    Args:
        character_name (str): The name of the character to use for synthesis.
        text (str): The text to be synthesized into speech.
        play (bool, optional): If True, plays the audio as it's generated. Defaults to False.
        split_sentence (bool, optional): If True, splits the text into sentences for synthesis. Defaults to False.
        save_path (str | PathLike | None, optional): If provided, saves the generated audio to this file path. Defaults to None.

    Yields:
        bytes: A chunk of the generated audio data.

    Raises:
        ValueError: If 'set_reference_audio' or 'load_persona' has not been called for the character.
    """
    if not has_reference_audio(character_name):
        raise ValueError("Please call 'set_reference_audio' or 'load_persona' first.")

    if save_path:
        save_path = os.fspath(save_path)
        parent_dir = os.path.dirname(save_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    # 1. 创建 asyncio 队列和获取当前事件循环
    stream_queue: asyncio.Queue[Union[bytes, None]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # 2. 定义回调函数，用于在线程和 asyncio 之间安全地传递数据
    def tts_chunk_callback(chunk: Optional[bytes]):
        """This callback is called from the TTS worker thread."""
        loop.call_soon_threadsafe(stream_queue.put_nowait, chunk)

    session_language = normalize_language(language)

    # Lazy load language-specific resources
    if session_language == "zh":
        resource_manager.ensure_chinese()
    elif session_language == "ja":
        resource_manager.ensure_japanese()
    elif session_language == "en":
        resource_manager.ensure_base()

    ref_info = get_reference_audio(character_name)
    # Always use loaded model's version to ensure correct inference path
    model_version = model_manager.get_character_version(character_name)
    
    # Check if persona mode (skip audio preprocessing)
    if ref_info.get('is_persona', False):
        # Use cached prompt audio from load_persona()
        prompt_audio = ref_info.get('prompt_audio')
    else:
        # Standard mode: create ReferenceAudio from wav
        prompt_audio = ReferenceAudio(
            prompt_wav=ref_info['audio_path'],
            prompt_text=ref_info['audio_text'],
            language=ref_info.get('audio_lang') or 'auto',
            model_version=model_version,
        )

    # 3. 使用新的回调接口启动 TTS 会话
    tts_player.start_session(
        play=play,
        split=split_sentence,
        save_path=save_path,
        chunk_callback=tts_chunk_callback,
        speaker=character_name,
        prompt_audio=prompt_audio,
        language=session_language,
    )

    # 馈送文本并通知会话结束
    tts_player.feed(text)
    tts_player.end_session()

    # 4. 从队列中异步读取数据并产生
    while True:
        chunk = await stream_queue.get()
        if chunk is None:
            break
        yield chunk


def tts(
        character_name: str,
        text: str,
        play: bool = False,
        split_sentence: bool = True,
        save_path: Union[str, PathLike, None] = None,
        language: str = "ja",
) -> None:
    """
    Synchronously generates speech from text.

    This is a blocking function that will not return until the entire TTS
    process is complete.

    Args:
        character_name (str): The name of the character to use for synthesis.
        text (str): The text to be synthesized into speech.
        play (bool, optional): If True, plays the audio.
        split_sentence (bool, optional): If True, splits the text into sentences for synthesis.
        save_path (str | PathLike | None, optional): If provided, saves the generated audio to this file path. Defaults to None.
    """
    if not has_reference_audio(character_name):
        logger.error("Please call 'set_reference_audio' or 'load_persona' first.")
        return

    if save_path:
        save_path = os.fspath(save_path)
        parent_dir = os.path.dirname(save_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    normalized_language = normalize_language(language)
    
    # Lazy load language-specific resources
    if normalized_language == "zh":
        resource_manager.ensure_chinese()
    elif normalized_language == "ja":
        resource_manager.ensure_japanese()
    elif normalized_language == "en":
        resource_manager.ensure_base()

    ref_info = get_reference_audio(character_name)
    # Always use loaded model's version to ensure correct inference path
    model_version = model_manager.get_character_version(character_name)
    
    # Check if persona mode (skip audio preprocessing)
    if ref_info.get('is_persona', False):
        # Use cached prompt audio from load_persona()
        prompt_audio = ref_info.get('prompt_audio')
    else:
        # Standard mode: create ReferenceAudio from wav
        prompt_audio = ReferenceAudio(
            prompt_wav=ref_info['audio_path'],
            prompt_text=ref_info['audio_text'],
            language=ref_info.get('audio_lang') or 'auto',
            model_version=model_version,
        )

    tts_player.start_session(
        play=play,
        split=split_sentence,
        save_path=save_path,
        speaker=character_name,
        prompt_audio=prompt_audio,
        language=normalized_language,
    )
    tts_player.feed(text)
    tts_player.end_session()
    tts_player.wait_for_tts_completion()


def stop() -> None:
    """
    Stops the currently playing text-to-speech audio.
    """
    tts_player.stop()
