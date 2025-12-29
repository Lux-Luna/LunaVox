import atexit
import gc
from dataclasses import dataclass
import os
import logging
import onnxruntime
from onnxruntime import InferenceSession
from typing import Optional
import numpy as np
# from importlib.resources import files
from huggingface_hub import hf_hub_download

from .Utils.Shared import context
# from .Utils.Constants import PACKAGE_NAME
from .Utils.Utils import LRUCacheDict

logger = logging.getLogger(__name__)

SESS_OPTIONS = onnxruntime.SessionOptions()
SESS_OPTIONS.log_severity_level = 3
SESS_OPTIONS.add_session_config_entry("session.use_env_allocators", "1")

_DEFAULT_PROVIDER_ORDER: list[str] = [
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "ROCMExecutionProvider",
    "CPUExecutionProvider",
]


def _resolve_providers() -> list[str]:
    from .Utils.EnvManager import env_manager
    
    # 1. Check persistence/user requested mode
    target_mode = env_manager.get_mode()
    available = set(onnxruntime.get_available_providers())
    
    # If user explicitly wants CPU, we only return CPU provider
    if target_mode == "cpu":
        logger.info("LunaVox is running in CPU mode as configured.")
        return ["CPUExecutionProvider"]

    # 2. Handle GPU/Auto mode
    env_value = os.getenv("LUNAVOX_ORT_PROVIDERS")
    if env_value:
        requested = [item.strip() for item in env_value.split(",") if item.strip()]
        resolved = [provider for provider in requested if provider in available]
        if resolved:
            logger.info("Using ONNXRuntime providers from LUNAVOX_ORT_PROVIDERS: %s", ",".join(resolved))
            return resolved
        logger.warning(
            "Requested providers '%s' are not available in this environment. Falling back to auto detection.",
            env_value,
        )
    
    # Filter preferred providers by availability
    resolved = [provider for provider in _DEFAULT_PROVIDER_ORDER if provider in available]
    if resolved:
        logger.info("Auto-detected ONNXRuntime providers: %s", ",".join(resolved))
        return resolved
    
    logger.info("No preferred providers available or found; falling back to CPUExecutionProvider.")
    return ["CPUExecutionProvider"]


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
    T2S_DECODER_WEIGHT_FP32: str = 't2s_shared_fp32.bin'
    T2S_DECODER_WEIGHT_FP16: str = 't2s_shared_fp16.bin'
    VITS_WEIGHT_FP32: str = 'vits_fp32.bin'
    VITS_WEIGHT_FP16: str = 'vits_fp16.bin'


@dataclass
class GSVModel:
    T2S_ENCODER: InferenceSession
    T2S_FIRST_STAGE_DECODER: InferenceSession
    T2S_STAGE_DECODER: InferenceSession
    VITS: InferenceSession


def download_model(filename: str, repo_id: str = 'Lux-Luna/LunaVox') -> Optional[str]:
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )
        return model_path
    except Exception as e:
        logger.error(f"Failed to download model {filename}: {str(e)}", exc_info=True)


def convert_bin_to_fp32(fp16_bin_path: str, output_fp32_bin_path: str) -> None:
    """Converts FP16 binary weight file to FP32 for better CPU performance."""
    fp16_array = np.fromfile(fp16_bin_path, dtype=np.float16)
    fp32_array = fp16_array.astype(np.float32)
    fp32_array.tofile(output_fp32_bin_path)


def convert_bins_to_fp32(model_dir: str) -> None:
    """Scans for FP16 weights and converts to FP32 if missing."""
    fp16_fp32_pairs = [
        (_GSVModelFile.T2S_DECODER_WEIGHT_FP16, _GSVModelFile.T2S_DECODER_WEIGHT_FP32),
        (_GSVModelFile.VITS_WEIGHT_FP16, _GSVModelFile.VITS_WEIGHT_FP32),
    ]

    for fp16_name, fp32_name in fp16_fp32_pairs:
        fp16_bin = os.path.normpath(os.path.join(model_dir, fp16_name))
        fp32_bin = os.path.normpath(os.path.join(model_dir, fp32_name))

        if os.path.exists(fp16_bin) and not os.path.exists(fp32_bin):
            logger.info(f"Converting weights {fp16_name} to FP32 for CPU performance...")
            convert_bin_to_fp32(fp16_bin, fp32_bin)

    logger.info("Successfully checked/generated FP32 weights.")


class ModelManager:
    def __init__(self):
        capacity_str = os.getenv('Max_Cached_Character_Models', '3')
        self.character_to_model: dict[str, dict[str, InferenceSession]] = LRUCacheDict(
            capacity=int(capacity_str))
        self.character_model_paths: dict[str, str] = {}  # 创建一个持久化字典来存储角色模型路径
        self.character_versions: dict[str, str] = {}  # 存储每个角色的模型版本
        self.providers = _resolve_providers()

        self.cn_hubert: Optional[InferenceSession] = None

    def load_cn_hubert(self) -> bool:
        model_path: Optional[str] = os.getenv("HUBERT_MODEL_PATH")
        if not (model_path and os.path.isfile(model_path)):
            logger.info("Chinese HuBERT model not found locally. Starting download of 'chinese-hubert-base.onnx'...")
            model_path = download_model('chinese-hubert-base.onnx')
            logger.info(f"Chinese HuBERT model download completed. Saved to: {os.path.abspath(model_path)}")
        if not model_path:
            return False
        logger.info(f"Found existing Chinese HuBERT model at: {os.path.abspath(model_path)}")

        try:
            self.cn_hubert = onnxruntime.InferenceSession(model_path,
                                                          providers=self.providers,
                                                          sess_options=SESS_OPTIONS)
            logger.info("Successfully loaded CN_HuBERT model.")
            return True
        except Exception as e:
            logger.error(
                f"Error: Failed to load ONNX model '{model_path}'.\n"
                f"Details: {e}"
            )
        return False

    def get(self, character_name: str) -> Optional[GSVModel]:
        if character_name in self.character_to_model:
            model_map = self.character_to_model[character_name]
            return GSVModel(
                T2S_ENCODER=model_map["T2S_ENCODER"],
                T2S_FIRST_STAGE_DECODER=model_map["T2S_FIRST_STAGE_DECODER"],
                T2S_STAGE_DECODER=model_map["T2S_STAGE_DECODER"],
                VITS=model_map["VITS"]
            )
        if character_name in self.character_model_paths:
            model_dir = self.character_model_paths[character_name]
            if self.load_character(character_name, model_dir):
                return self.get(character_name)
            else:
                del self.character_model_paths[character_name]  # 如果重载失败，可以考虑从路径记录中移除，防止反复失败
                return None
        return None

    def has_character(self, character_name: str) -> bool:
        character_name = character_name.lower()
        return character_name in self.character_model_paths

    def load_character(self, character_name: str, model_dir: str) -> bool:
        character_name = character_name.lower()
        if character_name in self.character_to_model:
            logger.info(f"Character '{character_name}' is already in cache; no need to reload.")
            _ = self.character_to_model[character_name]  # 访问一次以更新其在LRU缓存中的位置
            return True
        
        # Load model version metadata
        model_version = 'v2'  # Default
        model_info_path = os.path.join(model_dir, 'model_info.json')
        if os.path.exists(model_info_path):
            try:
                import json
                with open(model_info_path, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                    model_version = model_info.get('version', 'v2')
                logger.info(f"Loaded model version metadata: {model_version}")
            except Exception as e:
                logger.warning(f"Failed to load model metadata, defaulting to v2: {e}")
        else:
            logger.info(f"No model_info.json found, assuming v2")

        model_dict: dict[str, InferenceSession] = {}
        
        from .Utils.EnvManager import env_manager
        # Refresh providers to reflect any mode changes (CPU/GPU switch)
        self.providers = _resolve_providers()
        is_cpu_mode = env_manager.get_mode() == "cpu"

        if is_cpu_mode:
            # CPU mode: Try to use FP32 files and ensure converted weights
            convert_bins_to_fp32(model_dir)
            if os.path.exists(os.path.join(model_dir, _GSVModelFile.T2S_ENCODER_FP32)):
                logger.info("CPU Mode: Using FP32 models for better performance.")
                files_to_load = {
                    "T2S_ENCODER": _GSVModelFile.T2S_ENCODER_FP32,
                    "T2S_FIRST_STAGE_DECODER": _GSVModelFile.T2S_FIRST_STAGE_DECODER_FP32,
                    "T2S_STAGE_DECODER": _GSVModelFile.T2S_STAGE_DECODER_FP32,
                    "VITS": _GSVModelFile.VITS_FP32,
                }
            else:
                logger.warning("CPU Mode: FP32 models not found, falling back to FP16 models.")
                files_to_load = {
                    "T2S_ENCODER": _GSVModelFile.T2S_ENCODER_FP16,
                    "T2S_FIRST_STAGE_DECODER": _GSVModelFile.T2S_FIRST_STAGE_DECODER_FP16,
                    "T2S_STAGE_DECODER": _GSVModelFile.T2S_STAGE_DECODER_FP16,
                    "VITS": _GSVModelFile.VITS_FP16,
                }
        else:
            # GPU/Auto mode: Prefer FP16 models for speed/memory efficiency
            if os.path.exists(os.path.join(model_dir, _GSVModelFile.T2S_ENCODER_FP16)):
                logger.info("GPU Mode: Using FP16 models.")
                files_to_load = {
                    "T2S_ENCODER": _GSVModelFile.T2S_ENCODER_FP16,
                    "T2S_FIRST_STAGE_DECODER": _GSVModelFile.T2S_FIRST_STAGE_DECODER_FP16,
                    "T2S_STAGE_DECODER": _GSVModelFile.T2S_STAGE_DECODER_FP16,
                    "VITS": _GSVModelFile.VITS_FP16,
                }
            else:
                logger.info("GPU Mode: FP16 models not found, using FP32 models.")
                files_to_load = {
                    "T2S_ENCODER": _GSVModelFile.T2S_ENCODER_FP32,
                    "T2S_FIRST_STAGE_DECODER": _GSVModelFile.T2S_FIRST_STAGE_DECODER_FP32,
                    "T2S_STAGE_DECODER": _GSVModelFile.T2S_STAGE_DECODER_FP32,
                    "VITS": _GSVModelFile.VITS_FP32,
                }

        for key, filename in files_to_load.items():
            model_path: str = os.path.join(model_dir, filename)
            model_path = os.path.normpath(model_path)
            try:
                # In CPU mode, if we are loading an FP32 model that expects a .bin file, 
                # ORT will pick up the converted _fp32.bin automatically if it's in the same dir.
                model_dict[key] = onnxruntime.InferenceSession(model_path,
                                                                      providers=self.providers,
                                                                      sess_options=SESS_OPTIONS)
                logger.info(f"Model loaded successfully: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load ONNX model '{model_path}' for key '{key}': {e}", exc_info=True)
                return False

        self.character_to_model[character_name] = model_dict
        self.character_model_paths[character_name] = model_dir
        self.character_versions[character_name] = model_version

        if not context.current_speaker:
            context.current_speaker = character_name

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
        logger.info(f"Character {character_name.capitalize()} removed successfully.")

    def clean_cache(self) -> None:
        """Deletes temporary FP32 weight files created for CPU mode."""
        temp_weights: list[str] = [_GSVModelFile.T2S_DECODER_WEIGHT_FP32, _GSVModelFile.VITS_WEIGHT_FP32]
        deleted_any: bool = False
        try:
            for character, model_dir in self.character_model_paths.items():
                for filename in temp_weights:
                    filepath: str = os.path.join(model_dir, filename)
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        deleted_any = True
            if deleted_any:
                logger.info("Temporary FP32 weight files have been cleaned up.")
        except Exception as e:
            logger.error(f"Failed to delete temporary weight file: {e}")


model_manager: ModelManager = ModelManager()
atexit.register(model_manager.clean_cache)
