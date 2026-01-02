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
from ..Utils.AssetManager import asset_manager
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
        asset_manager.ensure_chinese()
    elif session_language == "ja":
        asset_manager.ensure_japanese()
    elif session_language == "en":
        asset_manager.ensure_base()

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
        asset_manager.ensure_chinese()
    elif normalized_language == "ja":
        asset_manager.ensure_japanese()
    elif normalized_language == "en":
        asset_manager.ensure_base()

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


# =============================================================================
# Multi-Reference Audio Support (moved from Experimental)
# =============================================================================

from typing import List

try:
    from ..Resources.Audio.SpeakerVector import average_sv_embeddings
    _SV_AVERAGING_AVAILABLE = True
except ImportError:
    _SV_AVERAGING_AVAILABLE = False

SUPPORTED_AUDIO_EXTS = {'.wav', '.flac', '.ogg', '.aiff', '.aif', '.mp3'}


def create_multi_reference_audio(
    character_name: str,
    audio_paths: List[Union[str, PathLike]],
    audio_texts: List[str],
    audio_languages: Optional[List[str]] = None,
) -> Optional[ReferenceAudio]:
    """
    Create a reference audio with averaged speaker vectors from multiple reference audios.
    """
    if not audio_paths or not audio_texts:
        logger.error("audio_paths and audio_texts must not be empty")
        return None
    
    if len(audio_paths) != len(audio_texts):
        logger.error("audio_paths and audio_texts must have the same length")
        return None
    
    model_version = model_manager.get_character_version(character_name)
    
    if model_version not in ['v2Pro', 'v2ProPlus']:
        audio_paths = audio_paths[:1]
        audio_texts = audio_texts[:1]
        if audio_languages:
            audio_languages = audio_languages[:1]
    
    for i, audio_path in enumerate(audio_paths):
        audio_path_str = os.fspath(audio_path)
        ext = os.path.splitext(audio_path_str)[1].lower()
        if ext not in SUPPORTED_AUDIO_EXTS:
            logger.error(f"Audio {i+1} format '{ext}' is not supported.")
            return None
        if not os.path.exists(audio_path_str):
            logger.error(f"Audio {i+1} not found: {audio_path_str}")
            return None
    
    if audio_languages is None:
        audio_languages = ['auto'] * len(audio_paths)
    elif len(audio_languages) != len(audio_paths):
        audio_languages = ['auto'] * len(audio_paths)
    
    ref_audios: List[ReferenceAudio] = []
    for i, (path, text, lang) in enumerate(zip(audio_paths, audio_texts, audio_languages)):
        try:
            ref_audios.append(ReferenceAudio(
                prompt_wav=os.fspath(path), prompt_text=text,
                language=lang, model_version=model_version,
            ))
        except Exception as e:
            logger.error(f"Failed to load reference audio {i+1}: {e}")
            return None
    
    if not ref_audios:
        return None
    if model_version == 'v2' or not _SV_AVERAGING_AVAILABLE:
        return ref_audios[0]
    
    sv_embs = [ref.sv_emb for ref in ref_audios if ref.sv_emb is not None]
    if not sv_embs:
        return ref_audios[0]
    
    averaged_sv = average_sv_embeddings(sv_embs)
    if averaged_sv is None:
        return ref_audios[0]
    
    base_ref = ref_audios[0]
    base_ref.sv_emb = averaged_sv
    return base_ref


def set_multi_reference_audio(
    character_name: str,
    audio_paths: List[Union[str, PathLike]],
    audio_texts: List[str],
    audio_languages: Optional[List[str]] = None,
) -> bool:
    """Set multiple reference audios for a character (v2Pro/v2ProPlus)."""
    from .state import set_reference_audio_config
    
    ref_audio = create_multi_reference_audio(
        character_name, audio_paths, audio_texts, audio_languages
    )
    if ref_audio is None:
        return False
    
    model_version = model_manager.get_character_version(character_name)
    set_reference_audio_config(character_name, {
        'audio_path': audio_paths[0],
        'audio_text': audio_texts[0],
        'audio_lang': audio_languages[0] if audio_languages else 'auto',
        'model_version': model_version,
        'multi_ref': True,
        'num_refs': len(audio_paths),
        'prompt_audio': ref_audio,
    })
    return True

