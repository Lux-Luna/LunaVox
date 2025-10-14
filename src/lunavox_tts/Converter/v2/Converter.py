from .VITSConverter import VITSConverter
from .T2SConverter import T2SModelConverter
from .EncoderConverter import EncoderConverter
from ..version_detector import detect_version, is_v2pro_variant
from ...Utils.Constants import PACKAGE_NAME

import logging
from typing import Optional, Tuple
import os
import shutil
import traceback
import importlib.resources
import contextlib
from pathlib import Path
import json

logger = logging.getLogger()

CACHE_DIR = os.path.join(os.getcwd(), "Cache")

# Resource paths - T2S models are shared across v2/v2Pro/v2ProPlus
_ENCODER_RESOURCE_PATH = "Data/v2/Models/t2s_encoder_fp32.onnx"
_STAGE_DECODER_RESOURCE_PATH = "Data/v2/Models/t2s_stage_decoder_fp32.onnx"
_FIRST_STAGE_DECODER_RESOURCE_PATH = "Data/v2/Models/t2s_first_stage_decoder_fp32.onnx"
_T2S_KEYS_RESOURCE_PATH = "Data/v2/Keys/t2s_onnx_keys.txt"

# Version-specific VITS paths
_VERSION_PATHS = {
    'v2': {
        'vits_onnx': "Data/v2/Models/vits_fp32.onnx",
        'vits_keys': "Data/v2/Keys/vits_onnx_keys.txt",
    },
    'v2Pro': {
        'vits_onnx': "Data/v2Pro/Models/vits_fp32.onnx",
        'vits_keys': "Data/v2Pro/Keys/vits_onnx_keys.txt",
    },
    'v2ProPlus': {
        'vits_onnx': "Data/v2ProPlus/Models/vits_fp32.onnx",
        'vits_keys': "Data/v2ProPlus/Keys/vits_onnx_keys.txt",
    },
}


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

    # Detect model version
    model_version = detect_version(torch_pth_path)
    logger.info(f"📦 Detected model version: {model_version}")
    
    # Validate version-specific resources
    if model_version not in _VERSION_PATHS:
        logger.error(f"Unsupported model version: {model_version}. Defaulting to v2.")
        model_version = 'v2'
    
    # Get version-specific paths
    version_config = _VERSION_PATHS[model_version]
    vits_resource_path = version_config['vits_onnx']
    vits_keys_resource_path = version_config['vits_keys']
    
    # Check if v2Pro/v2ProPlus ONNX templates exist and validate metadata
    if is_v2pro_variant(model_version):
        pkg_root = Path(__file__).resolve().parents[2]  # .../src/lunavox_tts
        # vits_resource_path already includes "Data/" prefix
        vits_template_path = pkg_root / vits_resource_path
        if not vits_template_path.exists():
            logger.error(
                f"❌ ONNX template for {model_version} not found at {vits_template_path}. "
                f"Please export the {model_version} VITS model to ONNX first. "
                f"See documentation for manual export instructions."
            )
            logger.warning(f"Falling back to v2 conversion (will NOT work correctly for {model_version} inference!)")
            model_version = 'v2'
            version_config = _VERSION_PATHS['v2']
            vits_resource_path = version_config['vits_onnx']
            vits_keys_resource_path = version_config['vits_keys']
        else:
            # Validate template metadata if meta.json exists
            meta_json_path = vits_template_path.parent / "meta.json"
            if meta_json_path.exists():
                try:
                    with open(meta_json_path, 'r', encoding='utf-8') as f:
                        template_meta = json.load(f)
                    
                    # Validate version matches
                    template_version = template_meta.get('version', 'unknown')
                    if template_version != model_version:
                        logger.error(
                            f"❌ Template metadata mismatch: "
                            f"Detected {model_version} but template is for {template_version}. "
                            f"Please re-export the ONNX template with correct version."
                        )
                        raise ValueError(f"Template version mismatch: {template_version} != {model_version}")
                    
                    # Validate architecture configuration from PTH
                    import torch
                    pth_state = torch.load(torch_pth_path, map_location='cpu', weights_only=False)
                    pth_config = pth_state.get('config', {})
                    
                    if hasattr(pth_config, 'model'):
                        model_config = pth_config.model
                    else:
                        model_config = pth_config.get('model', {})
                    
                    pth_upsample_initial = getattr(model_config, 'upsample_initial_channel', 
                                                   model_config.get('upsample_initial_channel', 512))
                    pth_upsample_kernels = getattr(model_config, 'upsample_kernel_sizes',
                                                   model_config.get('upsample_kernel_sizes', []))
                    
                    template_arch = template_meta.get('arch', {})
                    template_upsample_initial = template_arch.get('upsample_initial_channel', 512)
                    template_upsample_kernels = template_arch.get('upsample_kernel_sizes', [])
                    
                    # Check architecture match
                    if model_version == 'v2ProPlus':
                        if template_upsample_initial != 768:
                            logger.error(
                                f"❌ Architecture mismatch for v2ProPlus: "
                                f"Template has upsample_initial_channel={template_upsample_initial}, expected 768. "
                                f"You are using a v2 or v2Pro template with v2ProPlus weights. "
                                f"Please export the correct v2ProPlus ONNX template."
                            )
                            raise ValueError("Template architecture mismatch for v2ProPlus")
                        
                        if template_upsample_kernels and template_upsample_kernels[0] != 20:
                            logger.error(
                                f"❌ Architecture mismatch for v2ProPlus: "
                                f"Template has upsample_kernel_sizes={template_upsample_kernels}, expected [20,16,8,2,2]. "
                                f"Please export the correct v2ProPlus ONNX template."
                            )
                            raise ValueError("Template kernel size mismatch for v2ProPlus")
                    
                    elif model_version == 'v2Pro':
                        if template_upsample_initial != 512:
                            logger.error(
                                f"❌ Architecture mismatch for v2Pro: "
                                f"Template has upsample_initial_channel={template_upsample_initial}, expected 512. "
                                f"Please export the correct v2Pro ONNX template."
                            )
                            raise ValueError("Template architecture mismatch for v2Pro")
                    
                    logger.info(f"✓ Template metadata validated for {model_version}")
                    logger.info(f"  Architecture: upsample_initial={template_upsample_initial}, "
                              f"kernels={template_upsample_kernels}")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse meta.json: {e}. Proceeding without validation.")
                except Exception as e:
                    logger.warning(f"Failed to validate template metadata: {e}. Proceeding with caution.")
            else:
                logger.warning(f"No meta.json found for {model_version} template. "
                             f"Skipping architecture validation. "
                             f"If conversion fails, please re-export the ONNX template.")

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
            vits_onnx_path = stack.enter_context(importlib.resources.as_file(files.joinpath(vits_resource_path)))
            t2s_keys_path = stack.enter_context(importlib.resources.as_file(files.joinpath(_T2S_KEYS_RESOURCE_PATH)))
            vits_keys_path = stack.enter_context(importlib.resources.as_file(files.joinpath(vits_keys_resource_path)))

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
                model_version=model_version,
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
                
                # Save model version metadata
                model_info = {
                    'version': model_version,
                    'created_at': str(Path(output_dir).stat().st_mtime),
                }
                model_info_path = os.path.join(output_dir, 'model_info.json')
                with open(model_info_path, 'w', encoding='utf-8') as f:
                    json.dump(model_info, f, indent=2)
                logger.info(f"✓ Saved model metadata: {model_version}")
                
                logger.info(f"🎉 Conversion successful! Saved to: {os.path.abspath(output_dir)}\n")
            except Exception:
                logger.error(f"❌ A critical error occurred during the conversion process")
                logger.error(traceback.format_exc())
                remove_folder(output_dir)  # 只在失败时清理输出目录

    finally:
        # 无论成功还是失败，都尝试清理缓存目录
        remove_folder(CACHE_DIR)
