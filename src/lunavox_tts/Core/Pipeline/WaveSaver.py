"""
WaveSaver - Utility for saving audio chunks to WAV files.

Extracted from TTSPlayer to improve modularity.
"""
import os
import wave
import logging
import numpy as np
from typing import List

logger = logging.getLogger(__name__)


def preprocess_for_playback(audio_float: np.ndarray) -> bytes:
    """
    Convert float32 audio to int16 bytes for playback/saving.
    
    Args:
        audio_float: Audio waveform as float32 array, values in [-1.0, 1.0]
        
    Returns:
        Audio data as int16 bytes
    """
    if np.isnan(audio_float).any() or np.isinf(audio_float).any():
        audio_float = np.nan_to_num(audio_float, nan=0.0, posinf=0.0, neginf=0.0)
        
    audio_float = np.clip(audio_float, -1.0, 1.0)
    audio_int16 = (audio_float.squeeze() * 32767).astype(np.int16)
    return audio_int16.tobytes()


def save_wav(
    audio_chunks: List[np.ndarray],
    save_path: str,
    sample_rate: int = 32000,
    channels: int = 1,
    bytes_per_sample: int = 2
) -> bool:
    """
    Save audio chunks to a WAV file.
    
    Args:
        audio_chunks: List of audio arrays to concatenate and save
        save_path: Output file path
        sample_rate: Audio sample rate (default 32000)
        channels: Number of channels (default 1)
        bytes_per_sample: Bytes per sample (default 2 for int16)
        
    Returns:
        True if saved successfully, False otherwise
    """
    if not audio_chunks:
        logger.warning("No audio chunks to save")
        return False
    
    try:
        # Flatten and concatenate all chunks
        flattened_chunks = [
            chunk.flatten() if chunk.ndim > 1 else chunk 
            for chunk in audio_chunks
        ]
        full_audio = np.concatenate(flattened_chunks, axis=0)
        
        # Write WAV file
        with wave.open(save_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(bytes_per_sample)
            wf.setframerate(sample_rate)
            wf.writeframes(preprocess_for_playback(full_audio))
        
        logger.info(f"Audio saved to {os.path.abspath(save_path)}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")
        return False
