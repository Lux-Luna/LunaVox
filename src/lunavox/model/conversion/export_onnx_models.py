#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers.float16 import convert_float_to_float16

SCRIPT_DIR = Path(__file__).resolve().parent
HF_EXPORT_DIR = SCRIPT_DIR / "hf_export"
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if str(HF_EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(HF_EXPORT_DIR))

from .onnx_export_wrappers import (  # noqa: E402
    CodecEncoderExportWrapper,
    SpeakerEncoderExportWrapper,
    StatefulDecoderDynamoCombined,
)
from .speaker_encoder_local import load_speaker_encoder_from_base_dir  # noqa: E402
from .hf_export.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (  # noqa: E402
    Qwen3TTSTokenizerV2Model,
)


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def select_tokenizer_dir(base_dir: Path) -> Path:
    speech_tokenizer = base_dir / "speech_tokenizer"
    return speech_tokenizer if speech_tokenizer.exists() else base_dir


def ensure_exists(path: Path, name: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing {name}: {path}")


def convert_fp32_to_fp16(src_fp32: Path, dst_fp16: Path) -> None:
    model = onnx.load(str(src_fp32))
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=True,
        min_positive_val=1e-7,
        max_finite_val=65504,
        op_block_list=["LayerNormalization", "Softmax", "Range"],
    )
    onnx.save(model_fp16, str(dst_fp16))


def export_codec_encoder(base_dir: Path, output_dir: Path) -> Path:
    dst_fp16 = output_dir / "qwen3_tts_codec_encoder.fp16.onnx"
    if dst_fp16.exists():
        eprint(f"[skip] codec encoder exists: {dst_fp16}")
        return dst_fp16

    tokenizer_dir = select_tokenizer_dir(base_dir)
    model = Qwen3TTSTokenizerV2Model.from_pretrained(str(tokenizer_dir))
    model.eval()
    model.config.return_dict = False
    if hasattr(model, "encoder") and hasattr(model.encoder, "config"):
        model.encoder.config.return_dict = False
    model.encoder.apply(lambda m: m.remove_weight_norm() if hasattr(m, "remove_weight_norm") else None)

    wrapper = CodecEncoderExportWrapper(model).eval()
    dummy_input = torch.randn(1, 24000)
    tmp_fp32 = output_dir / "qwen3_tts_codec_encoder.fp32.onnx"
    def export_with_torchscript() -> None:
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(tmp_fp32),
            input_names=["input_values"],
            output_names=["audio_codes"],
            dynamic_axes={
                "input_values": {0: "batch_size", 1: "sequence_length"},
                "audio_codes": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )

    def export_with_dynamo() -> None:
        batch = torch.export.Dim("batch", min=1, max=8)
        seq = torch.export.Dim("sequence_length", min=1, max=240000)
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(tmp_fp32),
            input_names=["input_values"],
            output_names=["audio_codes"],
            dynamic_shapes=({0: batch, 1: seq},),
            opset_version=18,
            do_constant_folding=True,
            dynamo=True,
        )

    try:
        export_with_torchscript()
        eprint("[ok] codec encoder exported via torchscript path")
    except Exception as err:
        eprint(f"[warn] torchscript codec export failed, fallback to dynamo: {err}")
        if tmp_fp32.exists():
            tmp_fp32.unlink()
        export_with_dynamo()
        eprint("[ok] codec encoder exported via dynamo fallback")

    convert_fp32_to_fp16(tmp_fp32, dst_fp16)
    eprint(f"[ok] exported codec encoder: {dst_fp16}")
    return dst_fp16


def export_speaker_encoder(base_dir: Path, output_dir: Path) -> Path:
    dst_fp16 = output_dir / "qwen3_tts_speaker_encoder.fp16.onnx"
    if dst_fp16.exists():
        eprint(f"[skip] speaker encoder exists: {dst_fp16}")
        return dst_fp16

    speaker_encoder_module = load_speaker_encoder_from_base_dir(base_dir)
    wrapper = SpeakerEncoderExportWrapper(speaker_encoder_module).eval()
    dummy_input = torch.randn(1, 100, 128)
    tmp_fp32 = output_dir / "qwen3_tts_speaker_encoder.fp32.onnx"

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(tmp_fp32),
        input_names=["mels"],
        output_names=["spk_emb"],
        dynamic_axes={
            "mels": {0: "batch_size", 1: "sequence_length"},
            "spk_emb": {0: "batch_size"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    convert_fp32_to_fp16(tmp_fp32, dst_fp16)
    eprint(f"[ok] exported speaker encoder: {dst_fp16}")
    return dst_fp16


def export_decoder(base_dir: Path, output_dir: Path) -> Path:
    dst_fp16 = output_dir / "qwen3_tts_decoder.fp16.onnx"
    if dst_fp16.exists():
        eprint(f"[skip] decoder exists: {dst_fp16}")
        return dst_fp16

    tokenizer_dir = select_tokenizer_dir(base_dir)
    model = Qwen3TTSTokenizerV2Model.from_pretrained(str(tokenizer_dir)).to("cpu")
    model.eval()
    model.config.decoder_config._attn_implementation = "eager"
    model.config.decoder_config.head_dim = 64

    wrapper = StatefulDecoderDynamoCombined(model.decoder).to("cpu").eval()
    num_layers = wrapper.num_layers
    cfg = model.decoder.config
    num_heads = cfg.num_key_value_heads if hasattr(cfg, "num_key_value_heads") else cfg.num_attention_heads
    head_dim = cfg.head_dim

    bsz = 1
    n_frames = 3
    n_q = 16
    dummy_audio_codes = torch.zeros(bsz, n_frames, n_q, dtype=torch.long)
    dummy_pre_conv_h = torch.zeros(bsz, 512, 0)
    dummy_latent_buf = torch.zeros(bsz, 1024, 0)
    dummy_conv_h = torch.zeros(bsz, 1024, 0)
    dummy_is_last = torch.tensor([0.0])
    dummy_kv = [torch.zeros(bsz, num_heads, 0, head_dim) for _ in range(num_layers * 2)]

    dummy_inputs = (
        dummy_audio_codes,
        dummy_pre_conv_h,
        dummy_latent_buf,
        dummy_conv_h,
        dummy_is_last,
        *dummy_kv,
    )

    batch = torch.export.Dim("batch", min=1, max=8)
    num_frames = torch.export.Dim("num_frames", min=1, max=1024)
    past_seq = torch.export.Dim("past_seq", min=0, max=72)
    pre_conv_seq = torch.export.Dim("pre_conv_seq", min=0, max=2)
    latent_seq = torch.export.Dim("latent_seq", min=0, max=4)
    conv_seq = torch.export.Dim("conv_seq", min=0, max=4)
    dynamic_shapes = (
        {0: batch, 1: num_frames},
        {0: batch, 2: pre_conv_seq},
        {0: batch, 2: latent_seq},
        {0: batch, 2: conv_seq},
        None,
        tuple([{0: batch, 2: past_seq}] * (num_layers * 2)),
    )

    input_names = ["audio_codes", "pre_conv_history", "latent_buffer", "conv_history", "is_last"]
    output_names = ["final_wav", "valid_samples", "next_pre_conv_history", "next_latent_buffer", "next_conv_history"]
    input_names.extend([f"past_key_{i}" for i in range(num_layers)])
    input_names.extend([f"past_value_{i}" for i in range(num_layers)])
    output_names.extend([f"next_key_{i}" for i in range(num_layers)])
    output_names.extend([f"next_value_{i}" for i in range(num_layers)])

    tmp_fp32 = output_dir / "qwen3_tts_decoder.fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(tmp_fp32),
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            opset_version=18,
            dynamo=True,
        )
    convert_fp32_to_fp16(tmp_fp32, dst_fp16)
    eprint(f"[ok] exported decoder: {dst_fp16}")
    return dst_fp16


def quantize_int8_models(output_dir: Path, targets: Iterable[Path]) -> None:
    for fp32_path in targets:
        if not fp32_path.exists():
            continue
        out_int8 = fp32_path.with_suffix(".int8.onnx")
        quantize_dynamic(
            str(fp32_path),
            str(out_int8),
            op_types_to_quantize=["MatMul", "Attention", "Conv"],
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        eprint(f"[ok] quantized int8: {out_int8}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export Qwen3-TTS ONNX artifacts from local model files")
    ap.add_argument("--base-dir", required=True, help="Base model directory (Qwen3-TTS-12Hz-0.6B-Base)")
    ap.add_argument("--tokenizer-dir", default="", help="Compatibility arg (unused, local export auto-detects)")
    ap.add_argument("--output-dir", required=True, help="Destination directory for exported ONNX files")
    ap.add_argument(
        "--stage",
        choices=["codec_encoder", "speaker_encoder", "decoder", "quantize", "all"],
        default="all",
        help="Run a single stage or all stages",
    )
    ap.add_argument("--enable-quant", action="store_true", help="Enable optional local INT8 quantization")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_exists(base_dir, "base model directory")
    ensure_exists(base_dir / "model.safetensors", "base model weights")
    ensure_exists(base_dir / "config.json", "base model config")

    stage = args.stage
    if stage in {"codec_encoder", "all"}:
        export_codec_encoder(base_dir, output_dir)
    if stage in {"speaker_encoder", "all"}:
        export_speaker_encoder(base_dir, output_dir)
    if stage in {"decoder", "all"}:
        export_decoder(base_dir, output_dir)
    if args.enable_quant and stage in {"quantize", "all"}:
        quantize_int8_models(
            output_dir,
            [
                output_dir / "qwen3_tts_codec_encoder.fp32.onnx",
                output_dir / "qwen3_tts_speaker_encoder.fp32.onnx",
                output_dir / "qwen3_tts_decoder.fp32.onnx",
            ],
        )
    elif stage == "quantize":
        eprint("[skip] quantize stage requested but --enable-quant is not set")

    eprint("[done] local ONNX export completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
