"""
Mel-spectrogram extraction for reference audio.
"""
import numpy as np
import librosa


def extract_mel_spectrogram(audio_32k: np.ndarray, 
                           n_mels: int = 704,
                           n_fft: int = 2048,
                           hop_length: int = 640,
                           win_length: int = 2048) -> np.ndarray:
    """
    Extract mel-spectrogram from audio.
    
    Args:
        audio_32k: Audio waveform at 32kHz, shape (samples,)
        n_mels: Number of mel frequency bins (default: 704 for v2ProPlus)
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        win_length: Window length
        
    Returns:
        Mel-spectrogram, shape (1, n_mels, n_frames)
    """
    try:
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_32k,
            sr=32000,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            fmin=0,
            fmax=16000
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Add batch dimension: (n_mels, n_frames) -> (1, n_mels, n_frames)
        mel_spec_db = np.expand_dims(mel_spec_db, axis=0).astype(np.float32)
        
        return mel_spec_db
        
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to extract mel-spectrogram: {e}")
        raise

