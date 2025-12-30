import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

_tokenizer: Optional[Tokenizer] = None
_ort_session: Optional[ort.InferenceSession] = None
_model_dir: Optional[Path] = None

def _resolve_bert_paths() -> tuple[Path, Path]:
    """
    Resolve paths for RoBERTa ONNX model and tokenizer.
    Returns (model_path, tokenizer_path).
    """
    global _model_dir
    
    # Define local storage path
    from ..Utils.EnvManager import env_manager
    base_dir = env_manager.repo_root / "RoBERTa"
    
    from ..Utils.ResourceManager import resource_manager
    resource_manager.ensure_roberta()

    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        
    _model_dir = base_dir
    
    model_path = base_dir / "RoBERTa.onnx"
    tokenizer_path = base_dir / "roberta_tokenizer" / "tokenizer.json"
    
    return model_path, tokenizer_path

def _ensure_model_exists():
    model_path, tokenizer_path = _resolve_bert_paths()
    
    if not (model_path.exists() and tokenizer_path.exists()):
        logger.error(f"Chinese RoBERTa ONNX model or tokenizer not found at {model_path.parent}.")
        logger.error("Please ensure 'RoBERTa.onnx' is in the root 'RoBERTa' folder and 'tokenizer.json' is in 'RoBERTa/roberta_tokenizer'.")
        raise FileNotFoundError(f"Missing RoBERTa components in {model_path.parent}")

def _load_model() -> None:
    global _tokenizer, _ort_session
    
    if _tokenizer is not None and _ort_session is not None:
        return

    _ensure_model_exists()
    model_path, tokenizer_path = _resolve_bert_paths()

    try:
        _tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        _ort_session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=sess_options
        )
    except Exception as e:
        logger.error(f"Failed to load RoBERTa ONNX session: {e}")
        raise RuntimeError(f"RoBERTa load failed: {e}")

def compute_bert_phone_features(norm_text: str, word2ph: List[int], return_tensor: bool = False) -> np.ndarray:
    """
    Compute BERT features for Chinese text using ONNX Runtime.
    """
    if not norm_text:
        return np.zeros((sum(word2ph), 1024), dtype=np.float32)
    if len(word2ph) != len(norm_text):
        return np.zeros((sum(word2ph), 1024), dtype=np.float32)

    try:
        _load_model()
    except Exception as e:
        logger.warning(f"Chinese BERT features unavailable: {e}. Returning zeros.")
        return np.zeros((sum(word2ph), 1024), dtype=np.float32)

    assert _tokenizer is not None and _ort_session is not None

    # Tokenize
    encoding = _tokenizer.encode(norm_text)
    input_ids = encoding.ids
    attention_mask = encoding.attention_mask
    
    # Prepare inputs
    # Shape: (1, seq_len)
    inputs = {
        'input_ids': np.array([input_ids], dtype=np.int64),
        'attention_mask': np.array([attention_mask], dtype=np.int64),
        'token_type_ids': np.zeros((1, len(input_ids)), dtype=np.int64) 
    }
    
    # The RoBERTa.onnx usually takes 'input_ids', 'attention_mask', 'token_type_ids'
    # And outputs 'last_hidden_state' or similar. 
    # Let's verify input names dynamically if possible, or assume standard BERT export.
    
    # The model *does* take 'repeats'. This means it's a custom exported model that does the repeat logic inside ONNX.
    
    inputs['repeats'] = np.array(word2ph, dtype=np.int64)
    
    # If the model expects 'token_type_ids', we should provide it. 
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']

    outputs = _ort_session.run(None, inputs)
    text_bert = outputs[0].astype(np.float32)
    
    return text_bert
