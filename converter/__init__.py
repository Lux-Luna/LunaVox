"""
LunaVox Model Converter

Converts PyTorch GPT-SoVITS models to ONNX format for use with LunaVox TTS.

Usage:
    from converter import convert
    
    convert(
        ckpt_path="path/to/s1bert.ckpt",
        pth_path="path/to/s2G.pth",
        output_dir="output/model",
        format="fp16"
    )
"""
import os
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Literal

logger = logging.getLogger(__name__)

# Package data paths (relative to this file)
_DATA_DIR = Path(__file__).parent / "Data"
_LUNAVOX_DATA_DIR = Path(__file__).parent.parent / "src" / "lunavox_tts" / "Data"

# Version-specific resource paths
_RESOURCE_PATHS = {
    'v2': {
        't2s_encoder': "v2/Models/t2s_encoder_fp32.onnx",
        't2s_stage_decoder': "v2/Models/t2s_stage_decoder_fp32.onnx",
        't2s_first_stage_decoder': "v2/Models/t2s_first_stage_decoder_fp32.onnx",
        't2s_keys': "v2/Keys/t2s_onnx_keys.txt",
        'vits_onnx': "v2/Models/vits_fp32.onnx",
        'vits_keys': "v2/Keys/vits_onnx_keys.txt",
    },
    'v2Pro': {
        't2s_encoder': "v2/Models/t2s_encoder_fp32.onnx",
        't2s_stage_decoder': "v2/Models/t2s_stage_decoder_fp32.onnx",
        't2s_first_stage_decoder': "v2/Models/t2s_first_stage_decoder_fp32.onnx",
        't2s_keys': "v2/Keys/t2s_onnx_keys.txt",
        'vits_onnx': "v2Pro/Models/vits_fp32.onnx",
        'vits_keys': "v2Pro/Keys/vits_onnx_keys.txt",
    },
    'v2ProPlus': {
        't2s_encoder': "v2/Models/t2s_encoder_fp32.onnx",
        't2s_stage_decoder': "v2/Models/t2s_stage_decoder_fp32.onnx",
        't2s_first_stage_decoder': "v2/Models/t2s_first_stage_decoder_fp32.onnx",
        't2s_keys': "v2/Keys/t2s_onnx_keys.txt",
        'vits_onnx': "v2ProPlus/Models/vits_fp32.onnx",
        'vits_keys': "v2ProPlus/Keys/vits_onnx_keys.txt",
        'prompt_encoder': "v2ProPlus/Models/prompt_encoder_fp32.onnx",
        'prompt_encoder_keys': "v2ProPlus/Keys/prompt_encoder_weights.txt",
    },
}


def _get_data_dir() -> Path:
    """Get the data directory containing ONNX templates."""
    if _DATA_DIR.exists():
        return _DATA_DIR
    elif _LUNAVOX_DATA_DIR.exists():
        return _LUNAVOX_DATA_DIR
    else:
        raise FileNotFoundError(
            "ONNX template data not found. Please ensure the converter/Data or "
            "src/lunavox_tts/Data directory exists with required templates."
        )


def _get_resource_path(version: str, resource_key: str) -> Path:
    """Get the full path to a resource file."""
    data_dir = _get_data_dir()
    paths = _RESOURCE_PATHS.get(version, _RESOURCE_PATHS['v2'])
    relative_path = paths.get(resource_key)
    if not relative_path:
        raise KeyError(f"Resource '{resource_key}' not defined for version '{version}'")
    return data_dir / relative_path


def convert(
    ckpt_path: str,
    pth_path: str,
    output_dir: str,
    format: Literal["fp16"] = "fp16",
    model_version: Optional[str] = None,
) -> None:
    """
    Convert PyTorch GPT-SoVITS models to ONNX format.
    
    Args:
        ckpt_path: Path to the T2S model (.ckpt) file
        pth_path: Path to the VITS model (.pth) file
        output_dir: Directory to save converted models
        format: Output format - "fp16" (FP16 weights + FP32 skeleton)
        model_version: Model version override (auto-detected if None)
    """
    from .utils.version_detector import detect_version, ensure_torch
    from .core.t2s_converter import T2SConverter, EncoderConverter
    from .core.vits_converter import VITSConverter
    
    ensure_torch()
    
    # Create directories
    cache_dir = os.path.join(os.getcwd(), "Cache")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Detect model version
    version = model_version or detect_version(pth_path)
    logger.info(f"📦 Model version: {version}")
    
    # Get resource paths
    try:
        encoder_template = _get_resource_path(version, 't2s_encoder')
        stage_decoder_template = _get_resource_path(version, 't2s_stage_decoder')
        first_stage_template = _get_resource_path(version, 't2s_first_stage_decoder')
        t2s_keys = _get_resource_path(version, 't2s_keys')
        vits_template = _get_resource_path(version, 'vits_onnx')
        vits_keys = _get_resource_path(version, 'vits_keys')
    except FileNotFoundError as e:
        logger.error(f"Missing template files: {e}")
        raise
    
    # Convert T2S Encoder
    logger.info("🔄 Converting T2S Encoder...")
    encoder_conv = EncoderConverter(
        ckpt_path=ckpt_path,
        pth_path=pth_path,
        onnx_template_path=str(encoder_template),
        output_dir=output_dir,
    )
    encoder_conv.convert(format=format)
    
    # Convert T2S Decoders
    logger.info("🔄 Converting T2S Decoders...")
    t2s_conv = T2SConverter(
        torch_ckpt_path=ckpt_path,
        stage_decoder_onnx_path=str(stage_decoder_template),
        first_stage_decoder_onnx_path=str(first_stage_template),
        key_list_path=str(t2s_keys),
        output_dir=output_dir,
        cache_dir=cache_dir,
    )
    t2s_conv.convert(format=format)
    
    # Convert VITS
    logger.info("🔄 Converting VITS vocoder...")
    vits_conv = VITSConverter(
        torch_pth_path=pth_path,
        vits_onnx_path=str(vits_template),
        key_list_path=str(vits_keys),
        output_dir=output_dir,
        cache_dir=cache_dir,
        model_version=version,
    )
    vits_conv.convert(format=format)
    
    # Convert PromptEncoder for v2ProPlus
    if version == 'v2ProPlus':
        logger.info("🔄 Converting PromptEncoder...")
        from .core.prompt_encoder_converter import PromptEncoderConverter
        
        pe_template = _get_resource_path(version, 'prompt_encoder')
        pe_keys = _get_resource_path(version, 'prompt_encoder_keys')
        
        pe_conv = PromptEncoderConverter(
            torch_pth_path=pth_path,
            onnx_template_path=str(pe_template),
            key_list_path=str(pe_keys),
            output_dir=output_dir,
            cache_dir=cache_dir,
        )
        pe_conv.convert(format=format)
    
    # Save model info
    model_info = {"version": version, "format": format}
    with open(os.path.join(output_dir, "model_info.json"), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Cleanup cache
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    
    logger.info(f"✅ Conversion complete! Output: {output_dir}")
