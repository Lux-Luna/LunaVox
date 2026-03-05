
"""
ResourceManager - Centralized repository for shared model lifecycle management.

This module consolidates the ownership of global singleton resources like HuBERT,
preventing split responsibilities between ModelManager and RuntimeManager.
"""
import logging
import gc
import onnxruntime
import os
from typing import Optional, List
from onnxruntime import InferenceSession

from .Core.Model import get_default_sess_options, load_session_with_fp16_conversion

logger = logging.getLogger(__name__)

class GlobalResourceManager:
    """
    Manages global shared resources (HuBERT, etc.) that are not tied to specific characters.
    """
    _instance: Optional["GlobalResourceManager"] = None
    
    def __new__(cls) -> "GlobalResourceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.cn_hubert: Optional[InferenceSession] = None

    def load_cn_hubert(self, providers: List[str] = None) -> bool:
        """
        Load the Chinese HuBERT model for SSL feature extraction.
        
        Args:
            providers: List of execution providers (e.g. ['CUDAExecutionProvider']). 
                       If None, will be resolved automatically.
        """
        if self.cn_hubert is not None:
             return True

        from .Utils.AssetManager import asset_manager
        asset_manager.ensure_base()
        asset_manager.ensure_extractor()
        
        model_path = os.getenv("HUBERT_MODEL_PATH")
        if not (model_path and os.path.isfile(model_path)):
            potential_path = asset_manager.tts_data_dir / "chinese-hubert-base" / "chinese-hubert-base.onnx"
            if potential_path.is_file():
                model_path = str(potential_path)
            else:
                logger.error("Chinese HuBERT model not found in TTSData.")
                return False

        try:
            # Default providers if not supplied
            if providers is None:
                 from .Core.Model import resolve_providers
                 providers = resolve_providers()

            hubert_dir = os.path.dirname(model_path)
            hubert_fp16 = os.path.join(hubert_dir, "chinese-hubert-base_weights_fp16.bin")
            
            if os.path.exists(hubert_fp16):
                 self.cn_hubert = load_session_with_fp16_conversion(
                    model_path, hubert_fp16, providers, get_default_sess_options()
                )
            else:
                self.cn_hubert = onnxruntime.InferenceSession(
                    model_path, providers=providers, sess_options=get_default_sess_options()
                )
            logger.debug("Successfully loaded CN_HuBERT model via ResourceManager.")
            return True
        except Exception as e:
            logger.error(f"Failed to load ONNX model '{model_path}': {e}")
            return False

    def unload_cn_hubert(self) -> None:
        """Unload HuBERT model and release memory."""
        if self.cn_hubert is not None:
            logger.info("Unloading HuBERT model...")
            self.cn_hubert = None
            gc.collect()
            logger.info("✓ HuBERT model unloaded.")

    def is_hubert_loaded(self) -> bool:
        return self.cn_hubert is not None

# Global Singleton
resource_manager = GlobalResourceManager()
