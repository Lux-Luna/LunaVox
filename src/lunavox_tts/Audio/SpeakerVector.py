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
    Extract Kaldi-style fbank features from 16kHz waveform.
    
    This is a simplified NumPy-based implementation that approximates Kaldi's fbank.
    For production use, consider using torchaudio.compliance.kaldi.fbank or porting
    the full kaldi.py implementation.
    
    Args:
        waveform_16k: Audio waveform at 16kHz, shape (n_samples,)
        num_mel_bins: Number of mel filterbank bins (default: 80)
        
    Returns:
        Fbank features, shape (n_frames, num_mel_bins)
    """
    try:
        import torch
        import torchaudio.compliance.kaldi as kaldi
        
        # Convert to torch tensor and ensure correct shape
        if isinstance(waveform_16k, np.ndarray):
            # Ensure 2D: (1, n_samples) for Kaldi compliance
            if waveform_16k.ndim == 1:
                waveform_16k = waveform_16k[np.newaxis, :]
            elif waveform_16k.ndim == 2 and waveform_16k.shape[0] > 1:
                # Take first channel if multi-channel
                waveform_16k = waveform_16k[0:1, :]
            
            waveform_tensor = torch.from_numpy(waveform_16k).float()
        else:
            waveform_tensor = waveform_16k
        
        # Extract fbank features using Kaldi-compatible function
        # Parameters match GPT-SoVITS/GPT_SoVITS/eres2net/kaldi.py::fbank defaults
        fbank = kaldi.fbank(
            waveform_tensor,
            num_mel_bins=num_mel_bins,
            sample_frequency=16000,
            dither=0.0,  # No dithering for consistency
            window_type='povey',
            frame_length=25.0,
            frame_shift=10.0,
        )
        
        return fbank.numpy()
        
    except ImportError:
        logger.error(
            "torchaudio is required for fbank extraction. "
            "Please install it: pip install torchaudio"
        )
        raise
    except Exception as e:
        logger.error(f"Failed to extract fbank features: {e}")
        raise


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
        # Default location within package
        from pathlib import Path
        pkg_root = Path(__file__).resolve().parents[1]
        model_path = str(pkg_root / "Data" / "sv" / "eres2netv2.onnx")
    
    if not os.path.exists(model_path):
        logger.error(
            f"Speaker embedding model not found at {model_path}. "
            f"Please export ERes2NetV2 to ONNX first. "
            f"See documentation for manual export instructions."
        )
        return False
    
    try:
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        _sv_model = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=sess_options
        )
        _sv_model_path = model_path
        logger.info(f"✓ Loaded speaker embedding model from {model_path}")
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
        
        # Extract fbank features (n_frames, 80)
        fbank_feat = _get_fbank_features(waveform_16k, num_mel_bins=80)
        
        # Add batch dimension and convert to (B, T, F) format
        fbank_feat = np.expand_dims(fbank_feat, axis=0)  # (1, n_frames, 80)
        
        # Run ONNX inference
        # Expected input: (B, T, F) where B=batch, T=time frames, F=80 mel bins
        # Expected output: (B, 20480) flattened speaker embedding
        input_name = _sv_model.get_inputs()[0].name
        output_name = _sv_model.get_outputs()[0].name
        
        sv_emb = _sv_model.run([output_name], {input_name: fbank_feat.astype(np.float32)})[0]
        
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

