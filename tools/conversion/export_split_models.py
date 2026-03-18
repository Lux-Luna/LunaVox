#!/usr/bin/env python3
"""
将 HuggingFace 的 Qwen3-TTS safetensors 模型导出为 5 个独立的 GGUF 文件与 1 个 embeddings 文件夹。
输出路径为 models/base_small/
所有输出保持 FP16 (权重)。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

try:
    import gguf
except ImportError:
    print("ERROR: gguf package not installed. Run: pip install gguf", file=sys.stderr)
    sys.exit(1)


def load_tensor_as_numpy(sf_handle, name: str) -> np.ndarray:
    """Load a tensor from safetensors and convert to numpy array."""
    tensor = sf_handle.get_tensor(name)
    if isinstance(tensor, torch.Tensor):
        return tensor.float().numpy()
    return tensor.astype(np.float32)


# ===========================================================================
# Tensor name mapping: HF -> gguf standard names
# ===========================================================================

TALKER_HF_TO_GGUF = {
    "talker.codec_head.weight": "output.weight",
    "talker.model.norm.weight": "output_norm.weight",
    "talker.text_projection.linear_fc1.weight": "text_proj.fc1.weight",
    "talker.text_projection.linear_fc1.bias": "text_proj.fc1.bias",
    "talker.text_projection.linear_fc2.weight": "text_proj.fc2.weight",
    "talker.text_projection.linear_fc2.bias": "text_proj.fc2.bias",
}

TALKER_LAYER_HF_TO_GGUF = {
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}

PREDICTOR_HF_TO_GGUF = {
    "talker.code_predictor.model.norm.weight": "output_norm.weight",
}

PREDICTOR_LAYER_HF_TO_GGUF = TALKER_LAYER_HF_TO_GGUF

# Mapping functions
def map_talker_name(hf_name: str) -> str:
    if hf_name in TALKER_HF_TO_GGUF:
        return TALKER_HF_TO_GGUF[hf_name]
    prefix = "talker.model.layers."
    if hf_name.startswith(prefix):
        rest = hf_name[len(prefix):]
        dot_pos = rest.index(".")
        layer_idx = int(rest[:dot_pos])
        suffix = rest[dot_pos + 1:]
        if suffix in TALKER_LAYER_HF_TO_GGUF:
            return f"blk.{layer_idx}.{TALKER_LAYER_HF_TO_GGUF[suffix]}"
    return None

def map_predictor_name(hf_name: str) -> str:
    if hf_name in PREDICTOR_HF_TO_GGUF:
        return PREDICTOR_HF_TO_GGUF[hf_name]
    prefix = "talker.code_predictor.model.layers."
    if hf_name.startswith(prefix):
        rest = hf_name[len(prefix):]
        dot_pos = rest.index(".")
        layer_idx = int(rest[:dot_pos])
        suffix = rest[dot_pos + 1:]
        if suffix in PREDICTOR_LAYER_HF_TO_GGUF:
            return f"blk.{layer_idx}.{PREDICTOR_LAYER_HF_TO_GGUF[suffix]}"
    
    # Keep output heads
    prefix_head = "talker.code_predictor.lm_head."
    if hf_name.startswith(prefix_head) and hf_name.endswith(".weight"):
        idx = int(hf_name[len(prefix_head):hf_name.rindex(".")])
        return f"lm_head.{idx}.weight"
    return None

def write_gguf_f16(output_path: Path, tensors: dict, metadata: dict, architecture: str, tokens=None, merges=None):
    """Write a minimal F16 GGUF file."""
    writer = gguf.GGUFWriter(str(output_path), architecture)
    
    for key, value in metadata.items():
        if isinstance(value, int):
            writer.add_uint32(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        elif isinstance(value, str):
            writer.add_string(key, value)

    if tokens is not None:
        writer.add_string("tokenizer.ggml.model", "gpt2")
        writer.add_token_list(tokens)
    if merges is not None:
        writer.add_token_merges(merges)
            
    for name, data in sorted(tensors.items()):
        if name == "token_embd.weight":
            writer.add_tensor(name, data.astype(np.float32))
        elif data.ndim >= 2:
            writer.add_tensor(name, data.astype(np.float16))
        else:
            writer.add_tensor(name, data.astype(np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  -> Written: {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

def parse_tokenizer(model_dir: Path):
    vocab_path = model_dir / "vocab.json"
    merges_path = model_dir / "merges.txt"
    if not vocab_path.exists():
        return None, None
    with open(vocab_path, encoding="utf-8") as f:
        vocab_dict = json.load(f)
    tokens = [""] * len(vocab_dict)
    for token, idx in vocab_dict.items():
        if idx < len(tokens):
            tokens[idx] = token
    merges = []
    if merges_path.exists():
        with open(merges_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#version"):
                    merges.append(line)
    return tokens, merges

def export_embeddings(f, output_dir: Path):
    """提取各种 embeddings 为 npy，放到 embeddings 文件夹下"""
    emb_dir = output_dir / "embeddings"
    emb_dir.mkdir(exist_ok=True)
    print("\n[1/6] Exporting Embeddings to *.npy ...")
    
    # 1. Text Embedding Projected
    # y = fc2(silu(fc1(x)))
    raw_text = load_tensor_as_numpy(f, "talker.model.text_embedding.weight")
    w1 = load_tensor_as_numpy(f, "talker.text_projection.linear_fc1.weight")
    b1 = load_tensor_as_numpy(f, "talker.text_projection.linear_fc1.bias")
    w2 = load_tensor_as_numpy(f, "talker.text_projection.linear_fc2.weight")
    b2 = load_tensor_as_numpy(f, "talker.text_projection.linear_fc2.bias")
    
    import torch.nn.functional as F
    with torch.no_grad():
        t_raw = torch.from_numpy(raw_text)
        t_w1 = torch.from_numpy(w1)
        t_b1 = torch.from_numpy(b1)
        t_w2 = torch.from_numpy(w2)
        t_b2 = torch.from_numpy(b2)
        h = F.linear(t_raw, t_w1, t_b1)
        h = F.silu(h)
        proj_text = F.linear(h, t_w2, t_b2)
        
    np.save(emb_dir / "text_embedding_projected.npy", proj_text.numpy().astype(np.float16))
    
    # 2. Codec 0
    c0 = load_tensor_as_numpy(f, "talker.model.codec_embedding.weight")
    np.save(emb_dir / "codec_embedding_0.npy", c0)
    
    # 3. Code Predictor embeddings
    for i in range(16):
        k = f"talker.code_predictor.model.codec_embedding.{i}.weight"
        if k in f.keys():
            ci = load_tensor_as_numpy(f, k)
            np.save(emb_dir / f"codec_embedding_{i+1}.npy", ci)

def export_talker(f, output_dir: Path, config: dict):
    print("\n[2/6] Exporting Talker (qwen2) ...")
    tensors = {}
    for hf_name in f.keys():
        g_name = map_talker_name(hf_name)
        if g_name is not None:
            tensors[g_name] = load_tensor_as_numpy(f, hf_name)
            
    # Re-added the actual embeddings because llama.cpp strictly verifies the dimension against vocab_size.
    # Qwen3-TTS-GGUF Optimization: Set vocab_size to 3072 (codec length) and map token_embd.weight to codec_embedding.
    codec_emb_0 = load_tensor_as_numpy(f, "talker.model.codec_embedding.weight")
    tensors["token_embd.weight"] = codec_emb_0.astype(np.float32)

    talker_cfg = config.get("talker_config", config.get("talker", {}))
    # Fill Qwen2 standard metadata
    n_layers = talker_cfg.get("num_hidden_layers", 28)
    n_heads = talker_cfg.get("num_attention_heads", 16)
    n_kv_heads = talker_cfg.get("num_key_value_heads", 8)
    head_dim = talker_cfg.get("head_dim", 128)
    hidden_size = n_heads * head_dim
    intermediate_size = talker_cfg.get("intermediate_size", 3072)
    # STRIP text embedding via fake pure-audio vocab
    vocab_size = 3072
    
    meta = {
        "general.name": "qwen3-tts-talker",
        "qwen2.embedding_length": hidden_size,
        "qwen2.block_count": n_layers,
        "qwen2.attention.head_count": n_heads,
        "qwen2.attention.head_count_kv": n_kv_heads,
        "qwen2.feed_forward_length": intermediate_size,
        "qwen2.attention.key_length": head_dim,
        "qwen2.attention.value_length": head_dim,
        "qwen2.attention.layer_norm_rms_epsilon": float(talker_cfg.get("rms_norm_eps", 1e-6)),
        "qwen2.rope.freq_base": float(talker_cfg.get("rope_theta", 1000000.0)),
        "qwen2.context_length": 4096,
        "qwen2.vocab_size": vocab_size,
    }
    
    write_gguf_f16(output_dir / "talker.gguf", tensors, meta, "qwen2")

def export_predictor(f, output_dir: Path, config: dict):
    print("\n[3/6] Exporting Code Predictor (qwen2) ...")
    tensors = {}
    for hf_name in f.keys():
        g_name = map_predictor_name(hf_name)
        if g_name is not None:
            tensors[g_name] = load_tensor_as_numpy(f, hf_name)
            
    
    tensors["token_embd.weight"] = np.zeros((2048, 1024), dtype=np.float32)

    talker_cfg = config.get("talker_config", config.get("talker", {}))
    pred_cfg = talker_cfg.get("code_predictor_config", talker_cfg.get("code_predictor", {}))
    
    n_layers = pred_cfg.get("num_hidden_layers", 5)
    n_heads = pred_cfg.get("num_attention_heads", 16)
    n_kv_heads = pred_cfg.get("num_key_value_heads", 8)
    hidden_size = pred_cfg.get("hidden_size", 1024)
    intermediate_size = pred_cfg.get("intermediate_size", 3072)
    head_dim = hidden_size // n_heads
    vocab_size = pred_cfg.get("vocab_size", 2048)
    
    meta = {
        "general.name": "qwen3-tts-predictor",
        "qwen2.embedding_length": hidden_size,
        "qwen2.block_count": n_layers,
        "qwen2.attention.head_count": n_heads,
        "qwen2.attention.head_count_kv": n_kv_heads,
        "qwen2.feed_forward_length": intermediate_size,
        "qwen2.attention.key_length": head_dim,
        "qwen2.attention.value_length": head_dim,
        "qwen2.attention.layer_norm_rms_epsilon": float(pred_cfg.get("rms_norm_eps", 1e-6)),
        "qwen2.rope.freq_base": float(pred_cfg.get("rope_theta", 1000000.0)),
        "qwen2.context_length": 16,
        "qwen2.vocab_size": vocab_size,
    }
    
    write_gguf_f16(output_dir / "predictor.gguf", tensors, meta, "qwen2")

def export_audio_components(f_core, f_tok, output_dir: Path):
    print("\n[4-6/6] Exporting Audio Components (Custom GGML graph) ...")
    
    # Speaker Encoder
    spk_tensors = {}
    for k in f_core.keys():
        if k.startswith("speaker_encoder."):
            spk_tensors[f"spk_enc.{k[len('speaker_encoder.'):]}"] = load_tensor_as_numpy(f_core, k)
    if spk_tensors:
        write_gguf_f16(output_dir / "speaker_encoder.gguf", spk_tensors, {"general.name": "speaker-encoder"}, "speaker-encoder")

    # Codec Component wrapper
    dec_tensors = {}
    enc_tensors = {}
    for k in f_tok.keys():
        if k.startswith("decoder."):
            dec_tensors[f"tok_dec.{k[len('decoder.'):]}"] = load_tensor_as_numpy(f_tok, k)
        elif k.startswith("encoder."):
            enc_tensors[f"codec_enc.{k[len('encoder.'):]}"] = load_tensor_as_numpy(f_tok, k)
            
    if dec_tensors:
        write_gguf_f16(output_dir / "codec_decoder.gguf", dec_tensors, {"general.name": "codec-decoder"}, "codec-decoder")
    if enc_tensors:
        write_gguf_f16(output_dir / "codec_encoder.gguf", enc_tensors, {"general.name": "codec-encoder"}, "codec-encoder")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="HF model directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read config
    config_path = model_dir / "config.json"
    with open(config_path) as cf:
        config = json.load(cf)
        
    # Read tokens
    tokens, merges = parse_tokenizer(model_dir)
    
    safetensors_path = model_dir / "model.safetensors"
    speech_tok_path = model_dir / "speech_tokenizer" / "model.safetensors"
    
    if not speech_tok_path.exists():
        speech_tok_path = safetensors_path
        
    with safe_open(str(safetensors_path), framework="pt") as f_core, safe_open(str(speech_tok_path), framework="pt") as f_tok:
        export_embeddings(f_core, output_dir)
        export_talker(f_core, output_dir, config)
        export_predictor(f_core, output_dir, config)
        export_audio_components(f_core, f_tok, output_dir)
        
    import shutil
    shutil.copy(config_path, output_dir / "config.json")
    for fname in ["vocab.json", "merges.txt", "tokenizer_config.json"]:
        if (model_dir / fname).exists():
            shutil.copy(model_dir / fname, output_dir / fname)
    print(f"\nAll done! Converted models saved to {output_dir.absolute()}")

if __name__ == "__main__":
    main()
