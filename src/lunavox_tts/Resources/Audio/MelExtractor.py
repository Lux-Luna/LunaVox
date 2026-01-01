"""
Mel-spectrogram extraction for reference audio.
"""
import numpy as np


def extract_mel_spectrogram(audio_32k: np.ndarray, 
                           n_mels: int = 704,
                           n_fft: int = 2048,
                           hop_length: int = 640,
                           win_length: int = 2048) -> np.ndarray:
    """
    Extract mel-spectrogram from audio. Pure NumPy implementation.
    
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
        # 1. STFT (Pure NumPy)
        # Hann window
        window = np.hanning(win_length)
        # Pad reflect
        pad_len = n_fft // 2
        audio_padded = np.pad(audio_32k, pad_len, mode='reflect')
        
        # Framing
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            frames = sliding_window_view(audio_padded, win_length)[::hop_length]
        except ImportError:
            num_frames = 1 + (len(audio_padded) - win_length) // hop_length
            frames = np.zeros((num_frames, win_length))
            for i in range(num_frames):
                start = i * hop_length
                frames[i] = audio_padded[start:start+win_length]
        
        # FFT
        stft_matrix = np.fft.rfft(frames * window, n=n_fft)
        power_spec = np.abs(stft_matrix).T ** 2
        
        # 2. Mel Filterbank
        # Hz to Mel: m = 2595 * log10(1 + f / 700)
        # Mel to Hz: f = 700 * (10^(m / 2595) - 1)
        sr = 32000
        fmin = 0
        fmax = 16000
        
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700.0)
        
        def mel_to_hz(mel):
            return 700.0 * (10.0**(mel / 2595.0) - 1.0)
        
        min_mel = hz_to_mel(fmin)
        max_mel = hz_to_mel(fmax)
        mel_pts = np.linspace(min_mel, max_mel, n_mels + 2)
        hz_pts = mel_to_hz(mel_pts)
        
        # Bin indices
        bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
        
        # Create filterbank
        fb = np.zeros((n_mels, n_fft // 2 + 1))
        for m in range(1, n_mels + 1):
            for k in range(bin_pts[m-1], bin_pts[m]):
                fb[m-1, k] = (k - bin_pts[m-1]) / (bin_pts[m] - bin_pts[m-1])
            for k in range(bin_pts[m], bin_pts[m+1]):
                fb[m-1, k] = (bin_pts[m+1] - k) / (bin_pts[m+1] - bin_pts[m])
        
        # 3. Apply filterbank
        mel_spec = np.dot(fb, power_spec)
        
        # Convert to log scale (matching librosa.power_to_db default)
        # Power to dB: 10 * log10(S / ref)
        ref = np.max(mel_spec)
        mel_spec_db = 10 * np.log10(np.maximum(1e-10, mel_spec) / np.maximum(1e-10, ref))
        mel_spec_db = np.maximum(mel_spec_db, mel_spec_db.max() - 80.0) # Clip to 80dB range
        
        # Add batch dimension: (n_mels, n_frames) -> (1, n_mels, n_frames)
        mel_spec_db = np.expand_dims(mel_spec_db, axis=0).astype(np.float32)
        
        return mel_spec_db
        
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to extract mel-spectrogram (NumPy): {e}")
        raise

