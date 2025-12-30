"""
STFT Spectrogram extraction for reference audio (VITS input).

This matches the GPT-SoVITS spectrogram_torch function.
"""
import numpy as np


def extract_stft_spectrogram(audio_32k: np.ndarray,
                            n_fft: int = 1406,  # filter_length
                            hop_length: int = 640,
                            win_length: int = 1406,
                            center: bool = False) -> np.ndarray:
    """
    Extract STFT spectrogram from audio using parameters matching GPT-SoVITS.
    Pure NumPy implementation to replace librosa.
    
    Args:
        audio_32k: Audio waveform at 32kHz, shape (samples,)
        n_fft: FFT size (filter_length), default 1406 → 704 bins
        hop_length: Hop size, default 640
        win_length: Window length, default 1406
        center: Whether to center the window
        
    Returns:
        Spectrogram, shape (1, n_fft//2+1, n_frames)
    """
    try:
        # Normalize audio to [-1, 1]
        max_val = np.abs(audio_32k).max()
        if max_val > 1.2:
            audio_32k = audio_32k / max_val
        
        # Pad audio (reflect mode)
        if not center:
            pad_len = int((n_fft - hop_length) / 2)
            audio_padded = np.pad(audio_32k, (pad_len, pad_len), mode='reflect')
        else:
            audio_padded = audio_32k
        
        # Windowing (Hann window)
        window = np.hanning(win_length)
        
        # Framing using sliding_window_view (Efficient NumPy)
        # Note: sliding_window_view is available in NumPy 1.20+
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            frames = sliding_window_view(audio_padded, win_length)[::hop_length]
        except ImportError:
            # Fallback for older NumPy
            num_frames = 1 + (len(audio_padded) - win_length) // hop_length
            frames = np.zeros((num_frames, win_length))
            for i in range(num_frames):
                start = i * hop_length
                frames[i] = audio_padded[start:start+win_length]
        
        # Apply window
        frames = frames * window
        
        # RFFT
        stft_matrix = np.fft.rfft(frames, n=n_fft)
        
        # Get magnitude (sqrt(real^2 + imag^2))
        magnitude = np.abs(stft_matrix).T  # (n_fft//2+1, n_frames)
        magnitude = magnitude + 1e-8
        
        # Add batch dimension: (n_fft//2+1, n_frames) -> (1, n_fft//2+1, n_frames)
        spec = np.expand_dims(magnitude, axis=0).astype(np.float32)
        
        return spec
        
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to extract STFT spectrogram (NumPy): {e}")
        raise

