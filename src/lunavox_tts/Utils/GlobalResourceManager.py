"""
GlobalResourceManager - Centralized lifecycle management for heavy singleton models.

Provides unified load/unload interfaces for:
- HuBERT (SSL content extraction)
- BERT (Chinese semantic features)
- Speaker Vector (SV) (Timbre extraction)

This enables accurate memory benchmarking by ensuring complete resource cleanup
between test runs.
"""
import gc
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GlobalResourceManager:
    """
    Singleton manager for heavyweight global models.
    
    Usage:
        from lunavox_tts.Utils.GlobalResourceManager import global_resource_manager
        global_resource_manager.cleanup_all()  # Clean everything for fresh measurement
    """
    
    _instance: Optional["GlobalResourceManager"] = None
    
    def __new__(cls) -> "GlobalResourceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    # =========================================================================
    # HuBERT Management
    # =========================================================================
    
    def load_hubert(self) -> bool:
        """Load HuBERT model via ModelManager."""
        from ..ModelManager import model_manager
        return model_manager.load_cn_hubert()
    
    def unload_hubert(self) -> None:
        """Unload HuBERT model and release memory."""
        from ..ModelManager import model_manager
        if model_manager.cn_hubert is not None:
            logger.info("Unloading HuBERT model...")
            model_manager.cn_hubert = None
            gc.collect()
            logger.info("✓ HuBERT model unloaded.")
    
    def is_hubert_loaded(self) -> bool:
        """Check if HuBERT is currently loaded."""
        from ..ModelManager import model_manager
        return model_manager.cn_hubert is not None
    
    # =========================================================================
    # Chinese BERT Management
    # =========================================================================
    
    def load_zh_bert(self) -> bool:
        """Load Chinese RoBERTa BERT model."""
        try:
            from ..Languages.Chinese.ZhBert import _load_model
            _load_model()
            return True
        except Exception as e:
            logger.error(f"Failed to load ZhBERT: {e}")
            return False
    
    def unload_zh_bert(self) -> None:
        """Unload Chinese RoBERTa BERT model and release memory."""
        from ..Languages.Chinese import ZhBert
        if ZhBert._ort_session is not None or ZhBert._tokenizer is not None:
            logger.info("Unloading Chinese BERT model...")
            ZhBert._ort_session = None
            ZhBert._tokenizer = None
            ZhBert._model_dir = None
            gc.collect()
            logger.info("✓ Chinese BERT model unloaded.")
    
    def is_zh_bert_loaded(self) -> bool:
        """Check if Chinese BERT is currently loaded."""
        from ..Languages.Chinese import ZhBert
        return ZhBert._ort_session is not None
    
    def unload_all_bert(self) -> None:
        """Unload all language BERT models."""
        self.unload_zh_bert()
        # Future: add English BERT, Japanese BERT if needed
    
    # =========================================================================
    # Speaker Vector (SV) Management
    # =========================================================================
    
    def load_sv(self) -> bool:
        """Load Speaker Vector model."""
        from ..Resources.Audio.SpeakerVector import load_sv_model
        return load_sv_model()
    
    def unload_sv(self) -> None:
        """Unload Speaker Vector model and release memory."""
        from ..Resources.Audio import SpeakerVector
        if SpeakerVector._sv_model is not None:
            logger.info("Unloading Speaker Vector model...")
            SpeakerVector._sv_model = None
            SpeakerVector._sv_model_path = None
            gc.collect()
            logger.info("✓ Speaker Vector model unloaded.")
    
    def is_sv_loaded(self) -> bool:
        """Check if Speaker Vector model is currently loaded."""
        from ..Resources.Audio import SpeakerVector
        return SpeakerVector._sv_model is not None
    
    # =========================================================================
    # Character Model Management (delegated)
    # =========================================================================
    
    def unload_all_characters(self) -> None:
        """Unload all cached character models."""
        from ..ModelManager import model_manager
        logger.info("Unloading all character models...")
        model_manager.clean_cache()
        gc.collect()
        logger.info("✓ All character models unloaded.")
    
    # =========================================================================
    # Aggregate Cleanup
    # =========================================================================
    
    def cleanup_all(self) -> None:
        """
        Release ALL heavy resources for clean memory measurement.
        
        Call this between benchmark runs to ensure each test starts
        from a pristine state with no residual models.
        """
        import time
        
        logger.info("=== GlobalResourceManager: Full Cleanup ===")
        self.unload_hubert()
        self.unload_all_bert()
        self.unload_sv()
        self.unload_all_characters()
        self.clear_api_state()
        self.clear_frontend_cache()
        
        # Allow async operations to complete
        time.sleep(0.1)
        
        # Multiple GC passes for ONNX C++ destructor chain
        gc.collect()
        gc.collect()
        gc.collect()
        
        self.try_empty_vram()
        logger.info("=== Cleanup Complete ===")
    
    def clear_api_state(self) -> None:
        """
        Clear API-level state that holds ReferenceAudio objects.
        
        This prevents memory leaks from orphaned feature arrays when
        switching characters or running benchmarks.
        """
        try:
            from ..API.state import clear_all_reference_audio
            clear_all_reference_audio()
            logger.debug("✓ API reference audio state cleared.")
        except ImportError:
            pass
        
        try:
            from ..Resources.Audio.ReferenceAudio import ReferenceAudio
            ReferenceAudio.clear_cache()
            logger.debug("✓ ReferenceAudio prompt cache cleared.")
        except (ImportError, AttributeError):
            pass
    
    def clear_frontend_cache(self) -> None:
        """
        Clear cached frontend instances from LanguageRegistry.
        
        This releases heavy dependencies like jieba (Chinese) and
        pyopenjtalk (Japanese) that get cached after first use.
        """
        try:
            from ..Core.Frontend.registry import language_registry
            language_registry.clear_instances()
            logger.debug("✓ Frontend instances cleared.")
        except ImportError:
            pass
    
    def try_empty_vram(self) -> None:
        """
        Attempt to release GPU memory.
        
        Note: LunaVox uses ONNX Runtime, not PyTorch. ONNX RT manages
        its own GPU memory pools. We only call gc.collect() here.
        Importing torch would add ~360MB RAM overhead for no benefit.
        """
        gc.collect()
    
    def get_loaded_resources(self) -> dict:
        """Return a dict of which resources are currently loaded."""
        return {
            "hubert": self.is_hubert_loaded(),
            "zh_bert": self.is_zh_bert_loaded(),
            "sv": self.is_sv_loaded(),
        }


# Global singleton instance
global_resource_manager = GlobalResourceManager()
