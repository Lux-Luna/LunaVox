from os import PathLike
from typing import AsyncIterator


def load_character(character_name: str, onnx_model_dir: str | PathLike) -> None:
    """
    Loads a character model from an ONNX model directory.

    Args:
        character_name (str): The name to assign to the loaded character.
        onnx_model_dir (str | PathLike): The directory path containing the ONNX model files.
    """
    pass


def unload_character(character_name: str) -> None:
    """
    Unloads a previously loaded character model to free up resources.

    Args:
        character_name (str): The name of the character to unload.
    """
    pass


def set_reference_audio(character_name: str, audio_path: str | PathLike, audio_text: str) -> None:
    """
    Sets the reference audio for a character to be used for voice cloning.

    This must be called for a character before using 'tts' or 'tts_async'.

    Args:
        character_name (str): The name of the character.
        audio_path (str | PathLike): The file path to the reference audio (e.g., a WAV file).
        audio_text (str): The transcript of the reference audio.
    """
    pass


async def tts_async(character_name: str, text: str, play: bool = False, split_sentence: bool = False,
                    save_path: str | PathLike | None = None) -> AsyncIterator[bytes]:
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
        ValueError: If 'set_reference_audio' has not been called for the character.
    """
    pass


def tts(character_name: str, text: str, play: bool = False, split_sentence: bool = True,
        save_path: str | PathLike | None = None) -> None:
    """
    Synchronously generates speech from text.

    Args:
        character_name (str): The name of the character to use for synthesis.
        text (str): The text to be synthesized into speech.
        play (bool, optional): If True, plays the audio.
        split_sentence (bool, optional): If True, splits the text into sentences for synthesis.
        save_path (str | PathLike | None, optional): If provided, saves the generated audio to this file path. Defaults to None.
    """
    pass


def stop() -> None:
    """
    Stops the currently playing text-to-speech audio.
    """
    pass


def convert_to_onnx(torch_ckpt_path: str | PathLike, torch_pth_path: str | PathLike,
                    output_dir: str | PathLike) -> None:
    """
    Converts PyTorch model checkpoints to the ONNX format.

    Args:
        torch_ckpt_path (str | PathLike): The path to the T2S model (.ckpt) file.
        torch_pth_path (str | PathLike): The path to the VITS model (.pth) file.
        output_dir (str | PathLike): The directory where the ONNX models will be saved.
    """
    pass


def clear_reference_audio_cache() -> None:
    """
    Clears the cache of reference audio data.
    """
    pass


def launch_command_line_client() -> None:
    """
    Launch the command-line client.
    """
    pass


def load_predefined_character(character_name: str) -> None:
    """
    Download and load a predefined character model for TTS inference.
    """
    pass
