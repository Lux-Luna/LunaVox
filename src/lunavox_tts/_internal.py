# 请严格遵循导入顺序。
# 1、环境变量。
import os
from os import PathLike

os.environ["HF_HUB_ENABLE_PROGRESS_BAR"] = "1"

# 2、Logging。
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]"
)
logger = logging.getLogger(__name__)

# 3、ONNX。
import onnxruntime

onnxruntime.set_default_logger_severity(3)

# 导入剩余库。

import json
import asyncio
from typing import AsyncIterator, Optional, Union

from .Audio.ReferenceAudio import ReferenceAudio
from .Core.TTSPlayer import tts_player
from .ModelManager import model_manager
from .Utils.Shared import context
from .Client import Client
from .PredefinedCharacter import download_predefined_character_model

# Import for multi-reference SV averaging
try:
    from .Audio.SpeakerVector import average_sv_embeddings
    _SV_AVERAGING_AVAILABLE = True
except ImportError:
    _SV_AVERAGING_AVAILABLE = False

# A module-level private dictionary to store reference audio configurations.
_reference_audios: dict[str, dict] = {}
SUPPORTED_AUDIO_EXTS = {'.wav', '.flac', '.ogg', '.aiff', '.aif'}


def _normalize_language(code: Optional[str]) -> str:
    lang = (code or "ja").lower()
    return lang if lang in {"ja", "en", "zh"} else "ja"


def load_character(
        character_name: str,
        onnx_model_dir: Union[str, PathLike],
) -> None:
    """
    Loads a character model from an ONNX model directory.

    Args:
        character_name (str): The name to assign to the loaded character.
        onnx_model_dir (str | PathLike): The directory path containing the ONNX model files.
    """
    model_path: str = os.fspath(onnx_model_dir)
    model_manager.load_character(
        character_name=character_name,
        model_dir=model_path,
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
    audio_path: str = os.fspath(audio_path)

    # 检查文件后缀是否支持
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in SUPPORTED_AUDIO_EXTS:
        logger.error(
            f"Audio format '{ext}' is not supported. Only the following formats are supported: {SUPPORTED_AUDIO_EXTS}"
        )
        return

    # Get model version for the character
    model_version = model_manager.get_character_version(character_name)

    _reference_audios[character_name] = {
        'audio_path': audio_path,
        'audio_text': audio_text,
        'audio_lang': audio_language,
        'model_version': model_version,
    }
    context.current_prompt_audio = ReferenceAudio(
        prompt_wav=audio_path,
        prompt_text=audio_text,
        language=audio_language or 'auto',
        model_version=model_version,
    )


def create_persona(
        character_name: str,
        audio_path: Union[str, PathLike],
        audio_text: str,
        save_dir: Union[str, PathLike],
        audio_language: Optional[str] = None,
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
    
    # Create ReferenceAudio (this extracts all features)
    ref = ReferenceAudio(
        prompt_wav=audio_path_str,
        prompt_text=audio_text,
        language=audio_language or 'auto',
        model_version=model_version,
    )
    
    # Export to persona directory
    persona_path = ref.export_persona(save_dir_str, character_name, audio_path_str)
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
    
    if not os.path.isdir(persona_dir_str):
        raise FileNotFoundError(f"Persona directory not found: {persona_dir_str}")
    
    # Load persona using ReferenceAudio.from_persona
    ref = ReferenceAudio.from_persona(persona_dir_str)
    
    # Get model version from loaded persona
    model_version = getattr(ref, 'model_version', 'v2')
    
    # Register in reference audios dict
    _reference_audios[character_name] = {
        'persona_dir': persona_dir_str,
        'model_version': model_version,
        'is_persona': True,
    }
    
    # Set as current prompt audio
    context.current_prompt_audio = ref
    
    logger.info(f"✓ Persona loaded for '{character_name}' from: {persona_dir_str}")


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
    if character_name not in _reference_audios:
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

    # 设置 TTS 上下文
    session_language = _normalize_language(language)
    context.current_speaker = character_name
    ref_info = _reference_audios[character_name]
    model_version = ref_info.get('model_version', model_manager.get_character_version(character_name))
    
    # Check if persona mode (skip audio preprocessing)
    if ref_info.get('is_persona', False):
        # Use cached prompt audio from load_persona()
        prompt_audio = context.current_prompt_audio
    else:
        # Standard mode: create ReferenceAudio from wav
        context.current_prompt_audio = ReferenceAudio(
            prompt_wav=ref_info['audio_path'],
            prompt_text=ref_info['audio_text'],
            language=ref_info.get('audio_lang') or 'auto',
            model_version=model_version,
        )
        prompt_audio = context.current_prompt_audio

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
    if character_name not in _reference_audios:
        logger.error("Please call 'set_reference_audio' or 'load_persona' first.")
        return

    if save_path:
        save_path = os.fspath(save_path)
        parent_dir = os.path.dirname(save_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    context.current_speaker = character_name
    normalized_language = _normalize_language(language)
    context.current_language = normalized_language
    ref_info = _reference_audios[character_name]
    model_version = ref_info.get('model_version', model_manager.get_character_version(character_name))
    
    # Check if persona mode (skip audio preprocessing)
    if ref_info.get('is_persona', False):
        # Use cached prompt audio from load_persona()
        prompt_audio = context.current_prompt_audio
    else:
        # Standard mode: create ReferenceAudio from wav
        context.current_prompt_audio = ReferenceAudio(
            prompt_wav=ref_info['audio_path'],
            prompt_text=ref_info['audio_text'],
            language=ref_info.get('audio_lang') or 'auto',
            model_version=model_version,
        )
        prompt_audio = context.current_prompt_audio

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


def convert_to_onnx(
        torch_ckpt_path: Union[str, PathLike],
        torch_pth_path: Union[str, PathLike],
        output_dir: Union[str, PathLike],
) -> None:
    """
    Converts PyTorch model checkpoints to the ONNX format.

    This function requires PyTorch to be installed.

    Args:
        torch_ckpt_path (str | PathLike): The path to the T2S model (.ckpt) file.
        torch_pth_path (str | PathLike): The path to the VITS model (.pth) file.
        output_dir (str | PathLike): The directory where the ONNX models will be saved.
    """
    from .Converter.version_detector import ensure_torch
    ensure_torch()
    import torch

    from .Converter.v2.Converter import convert

    torch_ckpt_path = os.fspath(torch_ckpt_path)
    torch_pth_path = os.fspath(torch_pth_path)
    output_dir = os.fspath(output_dir)

    convert(
        torch_pth_path=torch_pth_path,
        torch_ckpt_path=torch_ckpt_path,
        output_dir=output_dir,
    )


def clear_reference_audio_cache() -> None:
    """
    Clears the cache of reference audio data.
    """
    ReferenceAudio.clear_cache()


def launch_command_line_client() -> None:
    """
    Launch the command-line client.
    """
    cmd_client: Client = Client()
    cmd_client.run()


def load_predefined_character(character_name: str) -> None:
    """
    Download and load a predefined character model for TTS inference.
    """
    character_name_list: list[str] = ['misono_mika']
    if character_name not in character_name_list:
        logger.error(f"No predefined character model found for {character_name}")

    save_path: str = download_predefined_character_model(character_name)
    model_manager.load_character(
        character_name=character_name,
        model_dir=save_path,
    )

    audio_path = os.path.join(save_path, "prompt.wav")
    with open(os.path.join(save_path, "prompt_wav.json"), "r", encoding="utf-8") as f:
        audio_text = json.load(f)["Normal"]["text"]
    _reference_audios[character_name] = {
        'audio_path': audio_path,
        'audio_text': audio_text,
    }
    context.current_prompt_audio = ReferenceAudio(
        prompt_wav=audio_path,
        prompt_text=audio_text,
    )
