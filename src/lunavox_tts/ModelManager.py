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
    available = set(onnxruntime.get_available_providers())
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
    resolved = [provider for provider in _DEFAULT_PROVIDER_ORDER if provider in available]
    if resolved:
        logger.info("Auto-detected ONNXRuntime providers: %s", ",".join(resolved))
        return resolved
    logger.info("No preferred providers available; falling back to CPUExecutionProvider.")
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


@dataclass
class GSVModel:
    T2S_ENCODER: InferenceSession
    T2S_FIRST_STAGE_DECODER: InferenceSession
    T2S_STAGE_DECODER: InferenceSession
    VITS: InferenceSession


def download_model(filename: str, repo_id: str = 'Lux-Luna/LunaVox') -> Optional[str]:
    try:
        # package_root = files(PACKAGE_NAME)
        # model_dir = str(package_root / "Data")
        # os.makedirs(model_dir, exist_ok=True)

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            # cache_dir=model_dir,
        )
        return model_path

    except Exception as e:
        logger.error(f"Failed to download model {filename}: {str(e)}", exc_info=True)


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
        
        # Check for FP16 models first
        if os.path.exists(os.path.join(model_dir, _GSVModelFile.T2S_ENCODER_FP16)):
            logger.info("Using FP16 models.")
            files_to_load = {
                "T2S_ENCODER": _GSVModelFile.T2S_ENCODER_FP16,
                "T2S_FIRST_STAGE_DECODER": _GSVModelFile.T2S_FIRST_STAGE_DECODER_FP16,
                "T2S_STAGE_DECODER": _GSVModelFile.T2S_STAGE_DECODER_FP16,
                # Try FP32 VITS first for stability, fallback to FP16 if not found
                "VITS": _GSVModelFile.VITS_FP32 if os.path.exists(os.path.join(model_dir, _GSVModelFile.VITS_FP32)) else _GSVModelFile.VITS_FP16,
            }
        else:
            logger.info("Using FP32 models.")
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
                model_dict[key] = onnxruntime.InferenceSession(model_path,
                                                                      providers=self.providers,
                                                                      sess_options=SESS_OPTIONS)
                logger.info(f"Model loaded successfully: {model_path}")
            except Exception as e:
                print(f"DEBUG: Error: Failed to load ONNX model '{model_path}'.\nDetails: {e}", flush=True)
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
        pass


model_manager: ModelManager = ModelManager()
atexit.register(model_manager.clean_cache)
