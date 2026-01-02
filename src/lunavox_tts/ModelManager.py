"""
ModelManager - High-level facade for model lifecycle management.

Orchestrates model registration, loading (via ModelLoader), 
and caching (via LRUCacheDict).
"""
import atexit
import os
import gc
import logging
import onnxruntime
from dataclasses import dataclass
from typing import Optional, List, Dict
from onnxruntime import InferenceSession

from .Utils.Utils import LRUCacheDict
from .Utils.PerformanceMonitor import monitor
from .Core.Model import (
    model_loader,
    model_registry,
    get_model_spec,
    detect_model_version,
    get_default_sess_options,
    resolve_providers,
    load_session_with_fp16_conversion
)

logger = logging.getLogger(__name__)


@dataclass
class GSVModel:
    """Legacy container for compatibility with current engine."""
    T2S_ENCODER: InferenceSession
    T2S_FIRST_STAGE_DECODER: InferenceSession
    T2S_STAGE_DECODER: InferenceSession
    VITS: InferenceSession
    PROMPT_ENCODER: Optional[InferenceSession] = None


class ModelManager:
    """
    Facade for model management. 
    Maintains a cache of active model sessions and handles resource lifecycle.
    """
    
    def __init__(self):
        capacity_str = os.getenv('Max_Cached_Character_Models', '3')
        self.character_to_model: Dict[str, Dict[str, InferenceSession]] = LRUCacheDict(
            capacity=int(capacity_str)
        )
        self.providers = resolve_providers()
        self.cn_hubert: Optional[InferenceSession] = None

    def load_cn_hubert(self) -> bool:
        """Load the Chinese HuBERT model (used for SSL feature extraction)."""
        from .Utils.ResourceManager import resource_manager
        resource_manager.ensure_base()
        resource_manager.ensure_extractor()
        
        model_path = os.getenv("HUBERT_MODEL_PATH")
        if not (model_path and os.path.isfile(model_path)):
            potential_path = resource_manager.tts_data_dir / "chinese-hubert-base" / "chinese-hubert-base.onnx"
            if potential_path.is_file():
                model_path = str(potential_path)
            else:
                logger.error("Chinese HuBERT model not found in TTSData.")
                return False

        try:
            hubert_dir = os.path.dirname(model_path)
            hubert_fp16 = os.path.join(hubert_dir, "chinese-hubert-base_weights_fp16.bin")
            
            if os.path.exists(hubert_fp16):
                 self.cn_hubert = load_session_with_fp16_conversion(
                    model_path, hubert_fp16, self.providers, get_default_sess_options()
                )
            else:
                self.cn_hubert = onnxruntime.InferenceSession(
                    model_path, providers=self.providers, sess_options=get_default_sess_options()
                )
            logger.debug("Successfully loaded CN_HuBERT model.")
            return True
        except Exception as e:
            logger.error(f"Failed to load ONNX model '{model_path}': {e}")
            return False

    def unload_cn_hubert(self) -> None:
        if self.cn_hubert is not None:
            logger.info("Unloading HuBERT model...")
            self.cn_hubert = None
            gc.collect()
            logger.info("✓ HuBERT model unloaded.")

    def unload_sv_model(self) -> None:
        """Unload Speaker Vector extractor model."""
        from .Resources.Audio import SpeakerVector
        if hasattr(SpeakerVector, "_sv_model") and SpeakerVector._sv_model is not None:
            logger.info("Unloading Speaker Vector model...")
            SpeakerVector._sv_model = None
            gc.collect()
            logger.info("✓ Speaker Vector model unloaded.")

    def unload_prompt_encoder(self, character_name: str) -> None:
        name = character_name.lower()
        if name in self.character_to_model:
            model_dict = self.character_to_model[name]
            if model_dict.get("PROMPT_ENCODER") is not None:
                logger.info(f"Unloading Prompt Encoder for '{character_name}'...")
                model_dict["PROMPT_ENCODER"] = None
                gc.collect()
                logger.info("✓ Prompt Encoder unloaded.")

    def get(self, character_name: str, skip_prompt_encoder: bool = False) -> Optional[GSVModel]:
        """Retrieve a character's model sessions, loading them if necessary."""
        name = character_name.lower()
        
        # Check cache
        if name in self.character_to_model:
            m = self.character_to_model[name]
            return GSVModel(
                T2S_ENCODER=m["T2S_ENCODER"],
                T2S_FIRST_STAGE_DECODER=m["T2S_FIRST_STAGE_DECODER"],
                T2S_STAGE_DECODER=m["T2S_STAGE_DECODER"],
                VITS=m["VITS"],
                PROMPT_ENCODER=m.get("PROMPT_ENCODER")
            )
        
        # Load if registered
        entry = model_registry.get(name)
        if entry:
            if self.load_character(name, entry.path, skip_prompt_encoder=skip_prompt_encoder):
                return self.get(name)
        
        return None

    def has_character(self, character_name: str) -> bool:
        return model_registry.has(character_name)

    def load_character(self, character_name: str, model_dir: str, skip_prompt_encoder: bool = False) -> bool:
        """Load all model components for a character."""
        import time
        t_start = time.perf_counter()
        name = character_name.lower()
        
        # 1. Check if already loaded with SAME model_dir BEFORE updating registry
        existing_entry = model_registry.get(name)
        already_loaded_same = (
            name in self.character_to_model and
            existing_entry and 
            existing_entry.path == model_dir
        )
        
        # 2. Detect version and register (this updates registry entry)
        version = detect_model_version(model_dir)
        spec = get_model_spec(version)
        entry = model_registry.register(name, model_dir, force_version=version)
        
        if already_loaded_same:
             m = self.character_to_model[name]
             # Handle upgrade (Persona -> Reference mode)
             if not skip_prompt_encoder and spec.prompt_encoder and m.get("PROMPT_ENCODER") is None:
                 logger.info(f"Upgrading character '{character_name}': Loading missing Prompt Encoder...")
             else:
                 return True
        
        # If loaded with different model_dir, force reload
        if name in self.character_to_model and not already_loaded_same:
            logger.info(f"Character '{character_name}' model changed from previous, reloading...")

        # 3. Resource Pre-check
        from .Utils.ResourceManager import resource_manager
        is_v2pp = version in ('v2ProPlus', 'v2pp')
        resource_manager.ensure_base()
        if is_v2pp:
            resource_manager.ensure_v2pp(skip_prompt_encoder=skip_prompt_encoder)
        
        # 4. Actual Loading via ModelLoader
        model_loader.refresh_providers()
        self.providers = model_loader.providers
        
        skip = set()
        if skip_prompt_encoder:
            skip.add("PROMPT_ENCODER")
            
        try:
            model_dict = model_loader.load_all(model_dir, spec, skip_components=skip)
            self.character_to_model[name] = model_dict
            model_registry.mark_loaded(name, set(model_dict.keys()))
            
            t_end = time.perf_counter()
            duration = t_end - t_start
            
            logger.info(f"✓ Character '{character_name.capitalize()}' loaded: type={version}, providers={self.providers}")
            monitor.log_metric(f"Load time ({name})", f"{duration:.2f}", "s")
            return True
        except Exception as e:
            logger.error(f"Error loading character '{character_name}': {e}", exc_info=True)
            return False

    def get_character_version(self, character_name: str) -> str:
        entry = model_registry.get(character_name)
        return entry.version if entry else 'v2'

    def remove_character(self, character_name: str) -> None:
        name = character_name.lower()
        if name in self.character_to_model:
            del self.character_to_model[name]
        model_registry.unregister(name)
        gc.collect()

    def clean_cache(self) -> None:
        self.character_to_model.clear()


model_manager: ModelManager = ModelManager()
atexit.register(model_manager.unload_cn_hubert)
