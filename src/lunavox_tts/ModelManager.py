import atexit
import gc
from dataclasses import dataclass
import os
import logging
import onnxruntime
from onnxruntime import InferenceSession
from typing import Optional, List
import numpy as np

from .Utils.Shared import context
from .Utils.Utils import LRUCacheDict
from .Utils.PerformanceMonitor import monitor
from .Core.Model import (
    get_default_sess_options,
    resolve_providers,
    load_session_with_fp16_conversion
)

logger = logging.getLogger(__name__)


class _GSVModelFile:
    T2S_ENCODER_FP16: str = 't2s_encoder_fp16.onnx'
    T2S_FIRST_STAGE_DECODER_FP16: str = 't2s_first_stage_decoder_fp16.onnx'
    T2S_STAGE_DECODER_FP16: str = 't2s_stage_decoder_fp16.onnx'
    VITS_FP16: str = 'vits_fp16.onnx'
    
    T2S_ENCODER_FP32: str = 't2s_encoder_fp32.onnx'
    T2S_FIRST_STAGE_DECODER_FP32: str = 't2s_first_stage_decoder_fp32.onnx'
    T2S_STAGE_DECODER_FP32: str = 't2s_stage_decoder_fp32.onnx'
    VITS_FP32: str = 'vits_fp32.onnx'

    # Binaries for weight conversion (CPU mode)
    T2S_DECODER_WEIGHT_FP16: str = 't2s_shared_fp16.bin'
    VITS_WEIGHT_FP16: str = 'vits_fp16.bin'
    PROMPT_ENCODER_WEIGHT_FP16: str = 'prompt_encoder_fp16.bin'

    PROMPT_ENCODER_FP16: str = 'prompt_encoder_fp16.onnx'
    PROMPT_ENCODER_FP32: str = 'prompt_encoder_fp32.onnx'


@dataclass
class GSVModel:
    T2S_ENCODER: InferenceSession
    T2S_FIRST_STAGE_DECODER: InferenceSession
    T2S_STAGE_DECODER: InferenceSession
    VITS: InferenceSession
    PROMPT_ENCODER: Optional[InferenceSession] = None


class ModelManager:
    def __init__(self):
        capacity_str = os.getenv('Max_Cached_Character_Models', '3')
        self.character_to_model: dict[str, dict[str, InferenceSession]] = LRUCacheDict(
            capacity=int(capacity_str))
        self.model_paths: dict[str, str] = {}  # Persistence dict for model paths
        self.character_versions: dict[str, str] = {}  # Store model versions
        self.providers = resolve_providers()

        self.cn_hubert: Optional[InferenceSession] = None

    def load_cn_hubert(self) -> bool:
        from .Utils.ResourceManager import resource_manager
        resource_manager.ensure_tts_data()
        
        model_path: Optional[str] = os.getenv("HUBERT_MODEL_PATH")
        
        # If env var not set or invalid, check default location in TTSData folder
        if not (model_path and os.path.isfile(model_path)):
            # Use absolute path from resource_manager
            potential_path = resource_manager.tts_data_dir / "chinese-hubert-base" / "chinese-hubert-base.onnx"
            if potential_path.is_file():
                model_path = str(potential_path)
            else:
                logger.error("Chinese HuBERT model not found in TTSData.")
                return False
        logger.debug(f"Found existing Chinese HuBERT model at: {os.path.abspath(model_path)}")

        try:
            # Check for FP16 weights for HuBERT
            # Assuming standard naming if we want to support patching here too.
            # But for now, stick to standard loading unless requested.
            hubert_dir = os.path.dirname(model_path)
            hubert_fp16 = os.path.join(hubert_dir, "chinese-hubert-base_weights_fp16.bin")
            
            if os.path.exists(hubert_fp16):
                 self.cn_hubert = load_session_with_fp16_conversion(
                    model_path, hubert_fp16, self.providers, get_default_sess_options()
                )
            else:
                self.cn_hubert = onnxruntime.InferenceSession(model_path,
                                                          providers=self.providers,
                                                          sess_options=get_default_sess_options())
            logger.debug("Successfully loaded CN_HuBERT model.")
            return True
        except Exception as e:
            logger.error(
                f"Error: Failed to load ONNX model '{model_path}'.\n"
                f"Details: {e}"
            )
        return False

    def unload_cn_hubert(self) -> None:
        """
        Unload the Chinese HuBERT model to free up memory/VRAM.
        Useful when in Persona mode where HuBERT is no longer needed.
        """
        if self.cn_hubert is not None:
            logger.info("Unloading HuBERT model...")
            del self.cn_hubert
            self.cn_hubert = None
            import gc
            gc.collect()
            logger.info("✓ HuBERT model unloaded.")

    def unload_sv_model(self) -> None:
        """
        Unload the Speaker Vector model to free up memory/VRAM.
        Useful when in Persona mode where SV extraction is no longer needed.
        """
        from .Resources.Audio import SpeakerVector
        if hasattr(SpeakerVector, "_sv_model") and SpeakerVector._sv_model is not None:
            logger.info("Unloading Speaker Vector model...")
            SpeakerVector._sv_model = None
            import gc
            gc.collect()
            logger.info("✓ Speaker Vector model unloaded.")

    def unload_prompt_encoder(self, character_name: str) -> None:
        """
        Unload the Prompt Encoder session for a character to free memory/VRAM.
        Useful in Persona mode where global embeddings are already cached.
        """
        character_name = character_name.lower()
        if character_name in self.character_to_model:
            model_dict = self.character_to_model[character_name]
            if model_dict.get("PROMPT_ENCODER") is not None:
                logger.info(f"Unloading Prompt Encoder for '{character_name}'...")
                del model_dict["PROMPT_ENCODER"]
                model_dict["PROMPT_ENCODER"] = None
                gc.collect()
                logger.info("✓ Prompt Encoder unloaded.")

    def get(self, character_name: str, skip_prompt_encoder: bool = False) -> Optional[GSVModel]:
        if character_name in self.character_to_model:
            model_map = self.character_to_model[character_name]
            return GSVModel(
                T2S_ENCODER=model_map["T2S_ENCODER"],
                T2S_FIRST_STAGE_DECODER=model_map["T2S_FIRST_STAGE_DECODER"],
                T2S_STAGE_DECODER=model_map["T2S_STAGE_DECODER"],
                VITS=model_map["VITS"],
                PROMPT_ENCODER=model_map.get("PROMPT_ENCODER")
            )
        if character_name in self.model_paths:
            model_dir = self.model_paths[character_name]
            if self.load_character(character_name, model_dir, skip_prompt_encoder=skip_prompt_encoder):
                return self.get(character_name)
            else:
                del self.model_paths[character_name]
                return None
        return None

    def has_character(self, character_name: str) -> bool:
        character_name = character_name.lower()
        return character_name in self.model_paths

    def load_character(self, character_name: str, model_dir: str, skip_prompt_encoder: bool = False) -> bool:
        """
        Load a character's TTS models.
        
        Args:
            character_name: Name of the character.
            model_dir: Path to the model directory.
            skip_prompt_encoder: If True, skip loading the prompt_encoder.
                                 Useful when using Personas with cached global_emb.
        """
        import time
        t_start = time.perf_counter()
        character_name = character_name.lower()
        is_v2pp_attempt = "v2_pro_plus" in model_dir or "v2pp" in model_dir.lower()
        
        if character_name in self.character_to_model:
            model_dict = self.character_to_model[character_name]
            # If we transition from Persona -> Reference mode, we might need to load the missing Prompt Encoder
            if is_v2pp_attempt and not skip_prompt_encoder and model_dict.get("PROMPT_ENCODER") is None:
                logger.info(f"Upgrading character '{character_name}': Loading missing Prompt Encoder...")
            else:
                logger.debug(f"Character '{character_name}' is already in cache; no need to reload.")
                return True
        else:
            model_dict = {}

        from .Utils.ResourceManager import resource_manager
        resource_manager.ensure_character_data(v2pp=is_v2pp_attempt, skip_prompt_encoder=skip_prompt_encoder)

        # Load model version metadata
        model_version = 'v2'  # Default
        model_info_path = os.path.join(model_dir, 'model_info.json')
        if os.path.exists(model_info_path):
            try:
                import json
                with open(model_info_path, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                    model_version = model_info.get('version', 'v2')
                logger.debug(f"Loaded model version metadata: {model_version}")
            except Exception as e:
                logger.warning(f"Failed to load model metadata, defaulting to v2: {e}")
        else:
            logger.debug(f"No model_info.json found, assuming v2")

        from .Utils.EnvManager import env_manager
        # Refresh providers to reflect any mode changes (CPU/GPU switch)
        self.providers = resolve_providers()

        # Define model files and their corresponding weights
        model_load_plan = [
            ("T2S_ENCODER", _GSVModelFile.T2S_ENCODER_FP32, None),
            ("T2S_FIRST_STAGE_DECODER", _GSVModelFile.T2S_FIRST_STAGE_DECODER_FP32, _GSVModelFile.T2S_DECODER_WEIGHT_FP16),
            ("T2S_STAGE_DECODER", _GSVModelFile.T2S_STAGE_DECODER_FP32, _GSVModelFile.T2S_DECODER_WEIGHT_FP16),
            ("VITS", _GSVModelFile.VITS_FP32, _GSVModelFile.VITS_WEIGHT_FP16),
        ]
        
        # Only load PROMPT_ENCODER if v2ProPlus AND not skipping (persona mode with cached ge)
        if model_version == 'v2ProPlus' and not skip_prompt_encoder:
            model_load_plan.append(("PROMPT_ENCODER", _GSVModelFile.PROMPT_ENCODER_FP32, _GSVModelFile.PROMPT_ENCODER_WEIGHT_FP16))
        elif skip_prompt_encoder:
            logger.info("Skipping prompt_encoder loading (Persona has cached global embeddings)")

        try:
            total_steps = len(model_load_plan)
            for i, (key, onnx_file, bin_file) in enumerate(model_load_plan):
                # Skip if already loaded (upgrade scenario)
                if key in model_dict and model_dict[key] is not None:
                    continue

                # Simple progress hint
                logger.info(f"Loading character model component '{key}'... ({i+1}/{total_steps})")
                
                onnx_path = os.path.join(model_dir, onnx_file)
                bin_path = os.path.join(model_dir, bin_file) if bin_file else None
                
                # GPU/CPU Classic Path: Prioritize in-memory patching for memory efficiency
                if bin_path and os.path.exists(bin_path) and os.path.exists(onnx_path):
                    logger.debug(f"Loading {key} with in-memory FP16 patching...")
                    model_dict[key] = load_session_with_fp16_conversion(
                        onnx_path, bin_path, self.providers, get_default_sess_options()
                    )
                else:
                    # Fallback to standard loading
                    if os.path.exists(onnx_path):
                        logger.debug(f"Loading {key} from standard FP32 ONNX: {onnx_file}")
                        model_dict[key] = onnxruntime.InferenceSession(
                            onnx_path, providers=self.providers, sess_options=get_default_sess_options()
                        )
                    elif key == "PROMPT_ENCODER":
                        logger.warning(f"Skipping PROMPT_ENCODER (Optional) as file not found: {onnx_path}")
                        continue # Optional
                    else:
                        raise FileNotFoundError(f"Required model file not found: {onnx_file}")

            is_v2pp = model_dict.get("PROMPT_ENCODER") is not None
            if is_v2pp:
                model_version = 'v2ProPlus'

            t_end = time.perf_counter()
            duration = t_end - t_start
            
            # Calculate total model size on disk
            total_size_mb = 0
            for filename in os.listdir(model_dir):
                if filename.endswith(".onnx") or filename.endswith(".bin"):
                    total_size_mb += os.path.getsize(os.path.join(model_dir, filename))
            total_size_mb /= (1024 * 1024)

            logger.info(
                f"✓ Character '{character_name.capitalize()}' loaded successfully.\n"
                f"  - Model Type: {model_version}\n"
                f"  - Providers: {self.providers}\n"
                f"  - Total Size: {total_size_mb:.2f} MB"
            )
            monitor.log_metric(f"Load time ({character_name})", f"{duration:.2f}", "s")
            monitor.log_metric(f"Model Size ({character_name})", f"{total_size_mb:.2f}", "MB")

        except Exception as e:
            logger.error(f"Error loading character '{character_name}': {e}", exc_info=True)
            return False

        self.character_to_model[character_name] = model_dict
        self.model_paths[character_name] = model_dir
        self.character_versions[character_name] = model_version

        if not context.current_speaker:
            context.current_speaker = character_name
            
        gc.collect()

        return True

    def get_character_version(self, character_name: str) -> str:
        """Get the model version for a character (v2, v2Pro, v2ProPlus)."""
        character_name = character_name.lower()
        return self.character_versions.get(character_name, 'v2')

    def remove_character(self, character_name: str) -> None:
        character_name = character_name.lower()
        if character_name in self.character_to_model:
            del self.character_to_model[character_name]
        if character_name in self.character_versions:
            del self.character_versions[character_name]
        gc.collect()
        logger.debug(f"Character {character_name.capitalize()} removed successfully.")

    def clean_cache(self) -> None:
        pass


model_manager: ModelManager = ModelManager()
