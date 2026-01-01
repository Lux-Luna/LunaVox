"""
Audio Processing - Post-processing logic for TTS output.

Handles audio normalization, silence padding, and NaN cleanup.
"""
import numpy as np

DEFAULT_TRAILING_SILENCE = 0.40
SAMPLE_RATE = 32000


def postprocess_audio(
    audio: np.ndarray,
    trailing_silence: float = DEFAULT_TRAILING_SILENCE,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Apply postprocessing to TTS output audio."""
    if audio is None:
        return None
    
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    
    if trailing_silence > 0:
        silence_samples = int(trailing_silence * sample_rate)
        original_shape = audio.shape
        flat_audio = audio.flatten()
        silence = np.zeros(silence_samples, dtype=audio.dtype)
        padded_audio = np.concatenate([flat_audio, silence])
        
        if len(original_shape) > 1:
            audio = padded_audio.reshape(original_shape[0], -1)
        else:
            audio = padded_audio
    
    return audio


def convert_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 audio to PCM16 bytes for playback."""
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio.squeeze() * 32767).astype(np.int16)
    return audio_int16.tobytes()


def trim_eos_tokens(semantic_tokens: np.ndarray, eos_threshold: int = 1024) -> np.ndarray:
    """Trim semantic tokens after EOS token."""
    eos_indices = np.where(semantic_tokens >= eos_threshold)
    if len(eos_indices[0]) > 0:
        first_eos_index = eos_indices[-1][0]
        return semantic_tokens[..., :first_eos_index]
    return semantic_tokens
