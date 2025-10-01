from .VITSConverter import VITSConverter
from .T2SConverter import T2SModelConverter
from .EncoderConverter import EncoderConverter
from ...Utils.Constants import PACKAGE_NAME

import logging
from typing import Optional, Tuple
import os
import shutil
import traceback
import importlib.resources
import contextlib
from pathlib import Path

logger = logging.getLogger()

CACHE_DIR = os.path.join(os.getcwd(), "Cache")
_ENCODER_RESOURCE_PATH = "Data/v2/Models/t2s_encoder_fp32.onnx"
_STAGE_DECODER_RESOURCE_PATH = "Data/v2/Models/t2s_stage_decoder_fp32.onnx"
_FIRST_STAGE_DECODER_RESOURCE_PATH = "Data/v2/Models/t2s_first_stage_decoder_fp32.onnx"
_VITS_RESOURCE_PATH = "Data/v2/Models/vits_fp32.onnx"
_T2S_KEYS_RESOURCE_PATH = "Data/v2/Keys/t2s_onnx_keys.txt"
_VITS_KEYS_RESOURCE_PATH = "Data/v2/Keys/vits_onnx_keys.txt"


def _ensure_v2_resources_installed() -> None:
    """Verify base ONNX templates and key lists exist inside LunaVox package data.

    This function only validates resources within `src/lunavox_tts/Data/v2/...` and
    does not copy from any external repositories or backups.
    """
    try:
        pkg_root = Path(__file__).resolve().parents[2]  # .../src/lunavox_tts
        models_dir = pkg_root / "Data" / "v2" / "Models"
        keys_dir = pkg_root / "Data" / "v2" / "Keys"

        required_models = [
            "t2s_encoder_fp32.onnx",
            "t2s_stage_decoder_fp32.onnx",
            "t2s_first_stage_decoder_fp32.onnx",
            "vits_fp32.onnx",
        ]
        required_keys = [
            "t2s_onnx_keys.txt",
            "vits_onnx_keys.txt",
        ]

        missing: list[str] = []
        for name in required_models:
            if not (models_dir / name).exists():
                missing.append(os.fspath(models_dir / name))
        for name in required_keys:
            if not (keys_dir / name).exists():
                missing.append(os.fspath(keys_dir / name))

        if missing:
            raise FileNotFoundError(
                "Missing required LunaVox base resources: "
                + "; ".join(missing)
                + ". Ensure these files exist under 'src/lunavox_tts/Data/v2' or reinstall LunaVox."
            )

    except Exception as e:
        logger.error(f"Failed to verify v2 resources: {e}")
        raise


def find_ckpt_and_pth(directory: str) -> Tuple[Optional[str], Optional[str]]:
    ckpt_path: Optional[str] = None
    pth_path: Optional[str] = None
    for filename in os.listdir(directory):
        full_path: str = os.path.join(directory, filename)
        if filename.endswith(".ckpt") and ckpt_path is None:
            ckpt_path = full_path
        elif filename.endswith(".pth") and pth_path is None:
            pth_path = full_path
        if ckpt_path and pth_path:
            break
    return ckpt_path, pth_path


def remove_folder(folder: str) -> None:
    try:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            logger.info(f"🧹 Folder cleaned: {folder}")
    except Exception as e:
        logger.error(f"❌ Failed to clean folder {folder}: {e}")


def convert(torch_ckpt_path: str,
            torch_pth_path: str,
            output_dir: str):
    # 确保缓存和输出目录存在
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if len(os.listdir(output_dir)) > 0:
        logger.warning(f"The output directory {output_dir} is not empty!")

    try:
        # Ensure required resources are present inside the package before resolving them
        _ensure_v2_resources_installed()
        with contextlib.ExitStack() as stack:
            files = importlib.resources.files(PACKAGE_NAME)

            encoder_onnx_path = stack.enter_context(importlib.resources.as_file(files.joinpath(_ENCODER_RESOURCE_PATH)))
            stage_decoder_path = stack.enter_context(
                importlib.resources.as_file(files.joinpath(_STAGE_DECODER_RESOURCE_PATH)))
            first_stage_decoder_path = stack.enter_context(
                importlib.resources.as_file(files.joinpath(_FIRST_STAGE_DECODER_RESOURCE_PATH)))
            vits_onnx_path = stack.enter_context(importlib.resources.as_file(files.joinpath(_VITS_RESOURCE_PATH)))
            t2s_keys_path = stack.enter_context(importlib.resources.as_file(files.joinpath(_T2S_KEYS_RESOURCE_PATH)))
            vits_keys_path = stack.enter_context(importlib.resources.as_file(files.joinpath(_VITS_KEYS_RESOURCE_PATH)))

            converter_1 = T2SModelConverter(
                torch_ckpt_path=torch_ckpt_path,
                stage_decoder_onnx_path=str(stage_decoder_path),
                first_stage_decoder_onnx_path=str(first_stage_decoder_path),
                key_list_file=str(t2s_keys_path),
                output_dir=output_dir,
                cache_dir=CACHE_DIR,
            )
            converter_2 = VITSConverter(
                torch_pth_path=torch_pth_path,
                vits_onnx_path=str(vits_onnx_path),
                key_list_file=str(vits_keys_path),
                output_dir=output_dir,
                cache_dir=CACHE_DIR,
            )
            converter_3 = EncoderConverter(
                ckpt_path=torch_ckpt_path,
                pth_path=torch_pth_path,
                onnx_input_path=str(encoder_onnx_path),
                output_dir=output_dir,
            )

            try:
                converter_1.run_full_process()
                converter_2.run_full_process()
                converter_3.convert()
                logger.info(f"🎉 Conversion successful! Saved to: {os.path.abspath(output_dir)}\n")
            except Exception:
                logger.error(f"❌ A critical error occurred during the conversion process")
                logger.error(traceback.format_exc())
                remove_folder(output_dir)  # 只在失败时清理输出目录

    finally:
        # 无论成功还是失败，都尝试清理缓存目录
        remove_folder(CACHE_DIR)
