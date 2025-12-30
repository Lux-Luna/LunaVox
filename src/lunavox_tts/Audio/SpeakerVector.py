"""
Speaker Vector (SV) extraction for v2Pro/v2ProPlus models.

This module extracts speaker embeddings from reference audio using the ERes2NetV2 model.
The embeddings are used to control speaker timbre in v2Pro/v2ProPlus inference.
"""
import os
import logging
from typing import Optional
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# Global speaker embedding model session
_sv_model: Optional[ort.InferenceSession] = None
_sv_model_path: Optional[str] = None


def _get_fbank_features(waveform_16k: np.ndarray, num_mel_bins: int = 80) -> np.ndarray:
    """
    Extract Kaldi-style fbank features from 16kHz waveform using pure NumPy.
    
    Args:
        waveform_16k: Audio waveform at 16kHz, shape (n_samples,)
        num_mel_bins: Number of mel filterbank bins (default: 80)
        
    Returns:
        Fbank features, shape (n_frames, num_mel_bins)
    """
    # Ensure waveform is 1D
    if waveform_16k.ndim > 1:
        waveform_16k = waveform_16k.flatten()

    sample_rate = 16000
    frame_length_ms = 25.0
    frame_shift_ms = 10.0
    
    frame_length = int(sample_rate * frame_length_ms / 1000)
    frame_shift = int(sample_rate * frame_shift_ms / 1000)
    
    # Pre-emphasis (simplistic)
    # waveform_16k = np.append(waveform_16k[0], waveform_16k[1:] - 0.97 * waveform_16k[:-1])

    # Framing
    num_samples = len(waveform_16k)
    num_frames = 1 + int((num_samples - frame_length) / frame_shift)
    
    if num_frames <= 0:
        # Handle very short audio by padding
        pad_len = frame_length - num_samples
        waveform_16k = np.pad(waveform_16k, (0, pad_len), mode='constant')
        num_frames = 1
    
    # Povey window approximation (Hanning is usually close enough for typical use, or Hamming)
    # Kaldi's Povey window is pow(0.5 - 0.5*cos(2*pi*n/(N-1)), 0.85)
    window = np.hanning(frame_length)
    # window = np.power(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame_length) / (frame_length - 1)), 0.85)

    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        start = i * frame_shift
        frames[i] = waveform_16k[start:start+frame_length] * window
        
    # FFT
    n_fft = 512
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    pow_frames = ((mag_frames ** 2) / n_fft)
    
    # Mel Filterbank
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_mel_bins + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((n_fft + 1) * hz_points / sample_rate)

    fbank = np.zeros((num_mel_bins, int(n_fft / 2 + 1)))
    for m in range(1, num_mel_bins + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical stability
    filter_banks = np.log(filter_banks)  # dB

    # Mean normalization (optional, depending on model training)
    # filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    
    return filter_banks.astype(np.float32)


def load_sv_model(model_path: Optional[str] = None) -> bool:
    """
    Load the speaker embedding ONNX model.
    
    Args:
        model_path: Path to eres2netv2.onnx. If None, uses default location.
        
    Returns:
        True if loaded successfully, False otherwise.
    """
    global _sv_model, _sv_model_path
    
    if _sv_model is not None and _sv_model_path == model_path:
        return True
    
    if model_path is None:
        # Default location: LunaVox/TTSData/sv/eres2netv2.onnx
        from pathlib import Path
        from ..Utils.EnvManager import env_manager
        from ..Utils.ResourceManager import resource_manager
        resource_manager.ensure_tts_data(v2pp=True)
        model_path = str(env_manager.repo_root / "TTSData" / "sv" / "eres2netv2.onnx")
    
    if not os.path.exists(model_path):
        logger.error(
            f"Speaker embedding model not found at {model_path}. "
            f"Please export ERes2NetV2 to ONNX first. "
            f"See documentation for manual export instructions."
        )
        return False
    
    try:
        from ..Utils.EnvManager import env_manager
        device_mode = env_manager.get_mode()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device_mode == "gpu" else ["CPUExecutionProvider"]
        
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        _sv_model = ort.InferenceSession(
            model_path,
            providers=providers,
            sess_options=sess_options
        )
        _sv_model_path = model_path
        logger.debug(f"✓ Loaded speaker embedding model from {model_path} (mode={device_mode})")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load speaker embedding model: {e}")
        return False


def extract_sv_embedding(waveform_16k: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract speaker vector embedding from 16kHz waveform.
    
    Args:
        waveform_16k: Audio waveform at 16kHz, shape (n_samples,) or (1, n_samples)
        
    Returns:
        Speaker embedding of shape (1, 20480), or None if extraction fails.
    """
    global _sv_model
    
    # Ensure model is loaded
    if _sv_model is None:
        if not load_sv_model():
            logger.warning("Speaker embedding model not available, returning None")
            return None
    
    try:
        # Ensure waveform is 1D for fbank extraction
        if waveform_16k.ndim == 2:
            if waveform_16k.shape[0] == 1:
                waveform_16k = waveform_16k[0]
            else:
                waveform_16k = waveform_16k.mean(axis=0)
        
        # Determine model input type
        input_info = _sv_model.get_inputs()[0]
        input_name = input_info.name
        input_shape = input_info.shape
        
        # Prepare input
        if input_name == "waveform" or (len(input_shape) == 2):
            logger.debug(f"Sending raw waveform to SV model (input_name={input_name}, shape={input_shape})")
            waveform_input = np.expand_dims(waveform_16k, axis=0).astype(np.float32)
            model_input = waveform_input
        else:
            # Fallback to Fbank extraction
            logger.debug(f"Extracting Fbank features for SV model (input_name={input_name}, shape={input_shape})")
            fbank_feat = _get_fbank_features(waveform_16k, num_mel_bins=80)
            fbank_feat = np.expand_dims(fbank_feat, axis=0).astype(np.float32)
            model_input = fbank_feat

        # Inference with optional IO Binding for performance
        providers = _sv_model.get_providers()
        if "CUDAExecutionProvider" in providers:
            try:
                io_binding = _sv_model.io_binding()
                device_id = 0 # Default
                
                # Bind input
                input_ort = ort.OrtValue.ortvalue_from_numpy(model_input, 'cuda', device_id)
                io_binding.bind_ortvalue_input(input_name, input_ort)
                
                # Bind output
                output_name = _sv_model.get_outputs()[0].name
                io_binding.bind_output(output_name, 'cpu') # Pull back for further processing
                
                _sv_model.run_with_iobinding(io_binding)
                sv_emb = io_binding.copy_outputs_to_cpu()[0]
            except Exception as e:
                logger.warning(f"SV IO Binding failed: {e}, falling back to regular run")
                sv_emb = _sv_model.run(None, {input_name: model_input})[0]
        else:
            sv_emb = _sv_model.run(None, {input_name: model_input})[0]
        
        # Ensure output shape is (1, 20480)
        if sv_emb.shape != (1, 20480):
            logger.warning(f"Unexpected SV embedding shape: {sv_emb.shape}, expected (1, 20480)")
            # Try to reshape if possible
            if sv_emb.size == 20480:
                sv_emb = sv_emb.reshape(1, 20480)
            else:
                logger.error(f"Cannot reshape SV embedding with {sv_emb.size} elements to (1, 20480)")
                return None
        
        return sv_emb.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Failed to extract speaker embedding: {e}")
        return None


def average_sv_embeddings(sv_embs: list[np.ndarray]) -> Optional[np.ndarray]:
    """
    Average multiple speaker embeddings (for multi-reference audio).
    
    Args:
        sv_embs: List of speaker embeddings, each of shape (1, 20480)
        
    Returns:
        Averaged speaker embedding of shape (1, 20480), or None if input is empty.
    """
    if not sv_embs:
        return None
    
    try:
        # Filter out None values
        valid_embs = [emb for emb in sv_embs if emb is not None]
        
        if not valid_embs:
            return None
        
        if len(valid_embs) == 1:
            return valid_embs[0]
        
        # Stack and compute mean along batch dimension
        stacked = np.stack(valid_embs, axis=0)  # (N, 1, 20480)
        mean_emb = np.mean(stacked, axis=0)  # (1, 20480)
        
        return mean_emb.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Failed to average speaker embeddings: {e}")
        return None
