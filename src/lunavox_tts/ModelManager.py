import atexit
import gc
from dataclasses import dataclass
import os
import logging
import onnxruntime
from onnxruntime import InferenceSession
from typing import Optional, List
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


def download_model(filename: str, repo_id: str = 'Lux-Luna/LunaVox') -> Optional[str]:
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )
        return model_path
    except Exception as e:
        logger.error(f"Failed to download model {filename}: {str(e)}", exc_info=True)


def load_session_with_fp16_conversion(
    onnx_path: str,
    fp16_bin_path: str,
    providers: List[str],
    sess_options: Optional[onnxruntime.SessionOptions] = None
) -> InferenceSession:
    """
    Reads ONNX and FP16 weights, converts to FP32 in memory, 
    injects into ONNX model, and creates InferenceSession without temp files.
    """
    import onnx

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX Model not found: {onnx_path}")
    if not os.path.exists(fp16_bin_path):
        raise FileNotFoundError(f"FP16 Weight file not found: {fp16_bin_path}")

    model_proto = onnx.load(onnx_path, load_external_data=False)
    fp16_data = np.fromfile(fp16_bin_path, dtype=np.float16)
    fp32_data = fp16_data.astype(np.float32)
    fp32_bytes = fp32_data.tobytes()

    # Iterate and patch external data initializers
    for tensor in model_proto.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            offset = 0
            length = 0
            for entry in tensor.external_data:
                if entry.key == 'offset':
                    offset = int(entry.value)
                elif entry.key == 'length':
                    length = int(entry.value)

            if offset + length > len(fp32_bytes):
                logger.warning(
                    f"Tensor {tensor.name} requested a data range that exceeds the size of the provided bin file. "
                    f"Offset: {offset}, Length: {length}, Buffer: {len(fp32_bytes)}"
                )
                continue

            tensor_data = fp32_bytes[offset: offset + length]
            tensor.raw_data = tensor_data

            del tensor.external_data[:]
            tensor.data_location = onnx.TensorProto.DEFAULT

    try:
        session = InferenceSession(
            model_proto.SerializeToString(),
            providers=providers,
            sess_options=sess_options
        )
        return session
    except Exception as e:
        logger.error(f"Failed to load in-memory model {os.path.basename(onnx_path)}: {e}")
        raise e


class ModelManager:
    def __init__(self):
        capacity_str = os.getenv('Max_Cached_Character_Models', '3')
        self.character_to_model: dict[str, dict[str, InferenceSession]] = LRUCacheDict(
            capacity=int(capacity_str))
        self.character_model_paths: dict[str, str] = {}  # Persistence dict for model paths
        self.character_versions: dict[str, str] = {}  # Store model versions
        self.providers = _resolve_providers()

        self.cn_hubert: Optional[InferenceSession] = None

    def load_cn_hubert(self) -> bool:
        model_path: Optional[str] = os.getenv("HUBERT_MODEL_PATH")
        
        # If env var not set or invalid, check default location in Data folder
        if not (model_path and os.path.isfile(model_path)):
            # Try the new folder structure first
            potential_path = os.path.join("Data", "chinese-hubert-base", "chinese-hubert-base.onnx")
            if os.path.isfile(potential_path):
                model_path = potential_path
            else:
                logger.info("Chinese HuBERT model not found locally. Starting download of 'chinese-hubert-base.onnx'...")
                model_path = download_model('chinese-hubert-base.onnx')
                logger.info(f"Chinese HuBERT model download completed. Saved to: {os.path.abspath(model_path)}")
        
        if not model_path:
            return False
        logger.info(f"Found existing Chinese HuBERT model at: {os.path.abspath(model_path)}")

        try:
            # Check for FP16 weights for HuBERT
            # Assuming standard naming if we want to support patching here too.
            # But for now, stick to standard loading unless requested.
            hubert_dir = os.path.dirname(model_path)
            hubert_fp16 = os.path.join(hubert_dir, "chinese-hubert-base_weights_fp16.bin")
            
            if os.path.exists(hubert_fp16):
                 self.cn_hubert = load_session_with_fp16_conversion(
                    model_path, hubert_fp16, self.providers, SESS_OPTIONS
                )
            else:
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
                VITS=model_map["VITS"],
                PROMPT_ENCODER=model_map.get("PROMPT_ENCODER")
            )
        if character_name in self.character_model_paths:
            model_dir = self.character_model_paths[character_name]
            if self.load_character(character_name, model_dir):
                return self.get(character_name)
            else:
                del self.character_model_paths[character_name]
                return None
        return None

    def has_character(self, character_name: str) -> bool:
        character_name = character_name.lower()
        return character_name in self.character_model_paths

    def load_character(self, character_name: str, model_dir: str) -> bool:
        character_name = character_name.lower()
        if character_name in self.character_to_model:
            logger.info(f"Character '{character_name}' is already in cache; no need to reload.")
            _ = self.character_to_model[character_name]
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

        # Define model files and their corresponding weights
        model_load_plan = [
            ("T2S_ENCODER", _GSVModelFile.T2S_ENCODER_FP32, None),
            ("T2S_FIRST_STAGE_DECODER", _GSVModelFile.T2S_FIRST_STAGE_DECODER_FP32, _GSVModelFile.T2S_DECODER_WEIGHT_FP16),
            ("T2S_STAGE_DECODER", _GSVModelFile.T2S_STAGE_DECODER_FP32, _GSVModelFile.T2S_DECODER_WEIGHT_FP16),
            ("VITS", _GSVModelFile.VITS_FP32, _GSVModelFile.VITS_WEIGHT_FP16),
        ]
        
        if model_version == 'v2ProPlus':
            model_load_plan.append(("PROMPT_ENCODER", _GSVModelFile.PROMPT_ENCODER_FP32, _GSVModelFile.PROMPT_ENCODER_WEIGHT_FP16))

        try:
            for key, onnx_file, bin_file in model_load_plan:
                onnx_path = os.path.join(model_dir, onnx_file)
                
                # Check for native FP16 model first (if GPU and available)
                # But prioritize skeleton + bin if bin exists as per user request
                bin_path = os.path.join(model_dir, bin_file) if bin_file else None
                
                if bin_path and os.path.exists(bin_path) and os.path.exists(onnx_path):
                    # In-memory patching
                    logger.info(f"Loading {key} with in-memory FP16 patching...")
                    model_dict[key] = load_session_with_fp16_conversion(
                        onnx_path, bin_path, self.providers, SESS_OPTIONS
                    )
                else:
                    # Fallback to standard loading
                    # Check if there's an FP16 version of the ONNX file
                    fp16_onnx_name = getattr(_GSVModelFile, f"{key}_FP16", None)
                    fp16_onnx_path = os.path.join(model_dir, fp16_onnx_name) if fp16_onnx_name else None
                    
                    if fp16_onnx_path and os.path.exists(fp16_onnx_path):
                        logger.info(f"Loading {key} from native FP16 ONNX: {fp16_onnx_name}")
                        model_dict[key] = onnxruntime.InferenceSession(
                            fp16_onnx_path, providers=self.providers, sess_options=SESS_OPTIONS
                        )
                    elif os.path.exists(onnx_path):
                        logger.info(f"Loading {key} from standard FP32 ONNX: {onnx_file}")
                        model_dict[key] = onnxruntime.InferenceSession(
                            onnx_path, providers=self.providers, sess_options=SESS_OPTIONS
                        )
                    elif key == "PROMPT_ENCODER":
                        continue # Optional
                    else:
                        raise FileNotFoundError(f"Required model file not found: {onnx_file}")

            is_v2pp = model_dict.get("PROMPT_ENCODER") is not None
            if is_v2pp:
                model_version = 'v2ProPlus'

            logger.info(
                f"Character '{character_name.capitalize()}' loaded successfully.\n"
                f"- Model Path: {model_dir}\n"
                f"- Model Type: {model_version}\n"
                f"- Providers: {self.providers}"
            )

        except Exception as e:
            logger.error(f"Error loading character '{character_name}': {e}", exc_info=True)
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
