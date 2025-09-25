import os
from huggingface_hub import hf_hub_download
from typing import Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

files_list = [
    "prompt.wav",
    "prompt_wav.json",
    "t2s_encoder_fp32.bin",
    "t2s_encoder_fp32.onnx",
    "t2s_first_stage_decoder_fp32.onnx",
    "t2s_shared_fp16.bin",
    "t2s_stage_decoder_fp32.onnx",
    "vits_fp16.bin",
    "vits_fp32.onnx",
]


def download_predefined_character_model(character_name: str, save_path: Optional[str] = None) -> str:
    repo_id: str = "Lux-Luna/LunaVox"

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    logger.info(f"🚀 Starting download of model for character '{character_name}'. This may take a few moments... ⏳")

    cache_dir_path: Optional[str] = None  # 用来记录缓存目录

    for filename in files_list:
        repo_path = f"character_model/{character_name}/{filename}"
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=repo_path,
                cache_dir=save_path  # 下载到指定目录
            )
            if cache_dir_path is None:
                cache_dir_path = str(Path(local_path).parent)
            logger.info(f"✅ Downloaded {filename}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to download {filename}: {e}")

    logger.info(f"🎉 All model files for '{character_name}' have been downloaded to '{save_path or cache_dir_path}' 📂")

    return save_path or cache_dir_path
