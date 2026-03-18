#!/usr/bin/env python3
"""Export Talker and Predictor from HF safetensors as separate F16 GGUFs.

These GGUFs use standard llama.cpp tensor naming so that llama-quantize.exe
can apply high-quality K-Quants quantization.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def load_tensor_as_numpy(sf_handle, name: str) -> np.ndarray:
    """Load a tensor from safetensors and convert to float32 numpy array.
    Handles bfloat16 tensors that numpy doesn't natively support."""
    tensor = sf_handle.get_tensor(name)
    if isinstance(tensor, torch.Tensor):
        return tensor.float().numpy()
    return tensor.astype(np.float32)

# ---------------------------------------------------------------------------
# Standard llama.cpp tensor name for Qwen2-style models
# ---------------------------------------------------------------------------

TALKER_HF_TO_GGUF = {
    # Embeddings
    "talker.model.text_embedding.weight": "token_embd.weight",
    "talker.model.codec_embedding.weight": "token_embd_codec.weight",
    "talker.codec_head.weight":            "output.weight",
    "talker.model.norm.weight":            "output_norm.weight",
    # Text projection
    "talker.text_projection.linear_fc1.weight": "text_proj.fc1.weight",
    "talker.text_projection.linear_fc1.bias":   "text_proj.fc1.bias",
    "talker.text_projection.linear_fc2.weight": "text_proj.fc2.weight",
    "talker.text_projection.linear_fc2.bias":   "text_proj.fc2.bias",
}

TALKER_LAYER_HF_TO_GGUF = {
    "self_attn.q_proj.weight":              "attn_q.weight",
    "self_attn.k_proj.weight":              "attn_k.weight",
    "self_attn.v_proj.weight":              "attn_v.weight",
    "self_attn.o_proj.weight":              "attn_output.weight",
    "self_attn.q_norm.weight":              "attn_q_norm.weight",
    "self_attn.k_norm.weight":              "attn_k_norm.weight",
    "input_layernorm.weight":               "attn_norm.weight",
    "post_attention_layernorm.weight":       "ffn_norm.weight",
    "mlp.gate_proj.weight":                 "ffn_gate.weight",
    "mlp.up_proj.weight":                   "ffn_up.weight",
    "mlp.down_proj.weight":                 "ffn_down.weight",
}

PREDICTOR_HF_TO_GGUF = {
    "talker.code_predictor.model.norm.weight": "output_norm.weight",
}

PREDICTOR_LAYER_HF_TO_GGUF = TALKER_LAYER_HF_TO_GGUF  # Same attention/MLP structure


def map_talker_name(hf_name: str) -> str | None:
    """Map an HF tensor name to a standard llama.cpp name for the Talker."""
    if hf_name in TALKER_HF_TO_GGUF:
        return TALKER_HF_TO_GGUF[hf_name]

    # Layer tensors: talker.model.layers.{N}.{suffix}
    prefix = "talker.model.layers."
    if hf_name.startswith(prefix):
        rest = hf_name[len(prefix):]
        dot_pos = rest.index(".")
        layer_idx = int(rest[:dot_pos])
        suffix = rest[dot_pos + 1:]
        if suffix in TALKER_LAYER_HF_TO_GGUF:
            return f"blk.{layer_idx}.{TALKER_LAYER_HF_TO_GGUF[suffix]}"

    return None


def map_predictor_name(hf_name: str) -> str | None:
    """Map an HF tensor name to a standard llama.cpp name for the Predictor."""
    if hf_name in PREDICTOR_HF_TO_GGUF:
        return PREDICTOR_HF_TO_GGUF[hf_name]

    # Layer tensors: talker.code_predictor.model.layers.{N}.{suffix}
    prefix = "talker.code_predictor.model.layers."
    if hf_name.startswith(prefix):
        rest = hf_name[len(prefix):]
        dot_pos = rest.index(".")
        layer_idx = int(rest[:dot_pos])
        suffix = rest[dot_pos + 1:]
        if suffix in PREDICTOR_LAYER_HF_TO_GGUF:
            return f"blk.{layer_idx}.{PREDICTOR_LAYER_HF_TO_GGUF[suffix]}"

    # Codec embeddings: talker.code_predictor.model.codec_embedding.{i}.weight
    prefix_embd = "talker.code_predictor.model.codec_embedding."
    if hf_name.startswith(prefix_embd) and hf_name.endswith(".weight"):
        idx = int(hf_name[len(prefix_embd):hf_name.rindex(".")])
        return f"codec_embd.{idx}.weight"

    # LM heads: talker.code_predictor.lm_head.{i}.weight
    prefix_head = "talker.code_predictor.lm_head."
    if hf_name.startswith(prefix_head) and hf_name.endswith(".weight"):
        idx = int(hf_name[len(prefix_head):hf_name.rindex(".")])
        return f"lm_head.{idx}.weight"

    return None


def write_gguf_f16(
    output_path: Path,
    tensors: dict[str, np.ndarray],
    metadata: dict,
    architecture: str = "qwen2",
):
    """Write a minimal GGUF file in F16 with the given tensors and metadata.

    Uses the gguf Python package for correct GGUF v3 format.
    """
    try:
        import gguf
    except ImportError:
        print("ERROR: gguf package not installed. Run: pip install gguf", file=sys.stderr)
        sys.exit(1)

    writer = gguf.GGUFWriter(str(output_path), architecture)

    # Write metadata
    for key, value in metadata.items():
        if isinstance(value, int):
            writer.add_uint32(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        elif isinstance(value, str):
            writer.add_string(key, value)

    # Write tensors
    for name, data in sorted(tensors.items()):
        # Convert to F16 for 2D weight tensors, keep F32 for 1D (norms, biases)
        if data.ndim >= 2:
            tensor_data = data.astype(np.float16)
            writer.add_tensor(name, tensor_data)
        else:
            tensor_data = data.astype(np.float32)
            writer.add_tensor(name, tensor_data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  Written: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


def export_talker(model_dir: Path, output_dir: Path, config: dict):
    """Export Talker component as F16 GGUF with standard llama.cpp names."""
    print("[Step 1/2] Exporting Talker F16 GGUF...")
    safetensors_path = model_dir / "model.safetensors"

    tensors = {}
    with safe_open(str(safetensors_path), framework="pt") as f:
        for hf_name in f.keys():
            gguf_name = map_talker_name(hf_name)
            if gguf_name is not None:
                tensors[gguf_name] = load_tensor_as_numpy(f, hf_name)

    print(f"  Mapped {len(tensors)} Talker tensors")

    # Talker config as Qwen2-compatible metadata
    talker_cfg = config.get("talker", {})
    model_cfg = talker_cfg.get("model", {})
    hidden_size = model_cfg.get("hidden_size", 1024)
    n_layers = model_cfg.get("num_hidden_layers", 28)
    n_heads = model_cfg.get("num_attention_heads", 16)
    n_kv_heads = model_cfg.get("num_key_value_heads", 8)
    intermediate_size = model_cfg.get("intermediate_size", 3072)
    head_dim = hidden_size // n_heads
    vocab_size = model_cfg.get("vocab_size", 3072)
    rms_norm_eps = model_cfg.get("rms_norm_eps", 1e-6)
    rope_theta = model_cfg.get("rope_theta", 1000000.0)

    metadata = {
        # Required for llama-quantize to recognize the architecture
        "general.name": "qwen3-tts-talker",
        f"{hidden_size}.embedding_length": hidden_size,
        f"{n_layers}.block_count": n_layers,
        f"{n_heads}.attention.head_count": n_heads,
        f"{n_kv_heads}.attention.head_count_kv": n_kv_heads,
        f"{intermediate_size}.feed_forward_length": intermediate_size,
        f"{head_dim}.attention.key_length": head_dim,
        f"{head_dim}.attention.value_length": head_dim,
        f"{rms_norm_eps}.attention.layer_norm_rms_epsilon": rms_norm_eps,
        f"{rope_theta}.rope.freq_base": rope_theta,
    }

    # Create proper metadata with correct keys
    meta = {
        "general.name": "qwen3-tts-talker",
        "qwen2.embedding_length": hidden_size,
        "qwen2.block_count": n_layers,
        "qwen2.attention.head_count": n_heads,
        "qwen2.attention.head_count_kv": n_kv_heads,
        "qwen2.feed_forward_length": intermediate_size,
        "qwen2.attention.key_length": head_dim,
        "qwen2.attention.value_length": head_dim,
        "qwen2.attention.layer_norm_rms_epsilon": rms_norm_eps,
        "qwen2.rope.freq_base": rope_theta,
        "qwen2.context_length": 4096,
        "qwen2.vocab_size": vocab_size,
    }

    output_path = output_dir / "talker-f16.gguf"
    write_gguf_f16(output_path, tensors, meta, architecture="qwen2")
    return output_path


def export_predictor(model_dir: Path, output_dir: Path, config: dict):
    """Export Code Predictor component as F16 GGUF with standard llama.cpp names."""
    print("[Step 2/2] Exporting Predictor F16 GGUF...")
    safetensors_path = model_dir / "model.safetensors"

    tensors = {}
    with safe_open(str(safetensors_path), framework="pt") as f:
        for hf_name in f.keys():
            gguf_name = map_predictor_name(hf_name)
            if gguf_name is not None:
                tensors[gguf_name] = load_tensor_as_numpy(f, hf_name)

    print(f"  Mapped {len(tensors)} Predictor tensors")

    # Predictor config
    talker_cfg = config.get("talker", {})
    code_pred_cfg = talker_cfg.get("code_predictor", {})
    model_cfg = code_pred_cfg.get("model", {})
    hidden_size = model_cfg.get("hidden_size", 1024)
    n_layers = model_cfg.get("num_hidden_layers", 5)
    n_heads = model_cfg.get("num_attention_heads", 16)
    n_kv_heads = model_cfg.get("num_key_value_heads", 8)
    intermediate_size = model_cfg.get("intermediate_size", 3072)
    head_dim = hidden_size // n_heads
    vocab_size = model_cfg.get("vocab_size", 2048)
    rms_norm_eps = model_cfg.get("rms_norm_eps", 1e-6)
    rope_theta = model_cfg.get("rope_theta", 1000000.0)

    meta = {
        "general.name": "qwen3-tts-predictor",
        "qwen2.embedding_length": hidden_size,
        "qwen2.block_count": n_layers,
        "qwen2.attention.head_count": n_heads,
        "qwen2.attention.head_count_kv": n_kv_heads,
        "qwen2.feed_forward_length": intermediate_size,
        "qwen2.attention.key_length": head_dim,
        "qwen2.attention.value_length": head_dim,
        "qwen2.attention.layer_norm_rms_epsilon": rms_norm_eps,
        "qwen2.rope.freq_base": rope_theta,
        "qwen2.context_length": 16,
        "qwen2.vocab_size": vocab_size,
    }

    output_path = output_dir / "predictor-f16.gguf"
    write_gguf_f16(output_path, tensors, meta, architecture="qwen2")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export Talker & Predictor as F16 GGUFs")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to HF model directory (e.g. models/Qwen3-TTS-12Hz-0.6B-Base)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for intermediate GGUF files")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: config.json not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    safetensors_path = model_dir / "model.safetensors"
    if not safetensors_path.exists():
        print(f"ERROR: model.safetensors not found at {safetensors_path}", file=sys.stderr)
        sys.exit(1)

    export_talker(model_dir, output_dir, config)
    export_predictor(model_dir, output_dir, config)

    print("\n[done] F16 GGUFs exported successfully.")
    print(f"  Talker:    {output_dir / 'talker-f16.gguf'}")
    print(f"  Predictor: {output_dir / 'predictor-f16.gguf'}")


if __name__ == "__main__":
    main()
