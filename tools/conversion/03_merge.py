#!/usr/bin/env python3
"""Merge quantized Talker + Predictor GGUFs with F16 Decoder into final lunavox GGUF.

This is the critical step that:
1. Reads quantized tensor data from Talker/Predictor GGUFs (preserving quant types)
2. Reads Decoder weights from HF safetensors (as F16)
3. Reads text tokenizer data (vocab + merges) from HF
4. Remaps all tensor names to lunavox C++ conventions
5. Writes the final qwen3-tts-0.6B-base.gguf with all metadata
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def load_tensor_as_numpy(sf_handle, name: str) -> np.ndarray:
    """Load a tensor from safetensors and convert to float32 numpy array."""
    tensor = sf_handle.get_tensor(name)
    if isinstance(tensor, torch.Tensor):
        return tensor.float().numpy()
    return tensor.astype(np.float32)



# ===========================================================================
# Tensor name mapping: standard llama.cpp → lunavox C++ convention
# ===========================================================================

# Talker: standard names back to lunavox names
TALKER_STD_TO_LUNAVOX = {
    "token_embd.weight":      "talker.text_embd.weight",
    "token_embd_codec.weight":"talker.codec_embd.weight",
    "output.weight":          "talker.codec_head.weight",
    "output_norm.weight":     "talker.output_norm.weight",
    "text_proj.fc1.weight":   "talker.text_proj.fc1.weight",
    "text_proj.fc1.bias":     "talker.text_proj.fc1.bias",
    "text_proj.fc2.weight":   "talker.text_proj.fc2.weight",
    "text_proj.fc2.bias":     "talker.text_proj.fc2.bias",
}

TALKER_LAYER_STD_TO_LUNAVOX = {
    "attn_q.weight":       "attn_q.weight",
    "attn_k.weight":       "attn_k.weight",
    "attn_v.weight":       "attn_v.weight",
    "attn_output.weight":  "attn_output.weight",
    "attn_q_norm.weight":  "attn_q_norm.weight",
    "attn_k_norm.weight":  "attn_k_norm.weight",
    "attn_norm.weight":    "attn_norm.weight",
    "ffn_norm.weight":     "ffn_norm.weight",
    "ffn_gate.weight":     "ffn_gate.weight",
    "ffn_up.weight":       "ffn_up.weight",
    "ffn_down.weight":     "ffn_down.weight",
}

# Predictor: standard names back to lunavox names
PREDICTOR_STD_TO_LUNAVOX = {
    "output_norm.weight": "code_pred.output_norm.weight",
}


def map_talker_std_to_lunavox(std_name: str) -> str | None:
    if std_name in TALKER_STD_TO_LUNAVOX:
        return TALKER_STD_TO_LUNAVOX[std_name]
    m = re.match(r"blk\.(\d+)\.(.+)", std_name)
    if m:
        layer_idx = m.group(1)
        suffix = m.group(2)
        if suffix in TALKER_LAYER_STD_TO_LUNAVOX:
            return f"talker.blk.{layer_idx}.{TALKER_LAYER_STD_TO_LUNAVOX[suffix]}"
    return None


def map_predictor_std_to_lunavox(std_name: str) -> str | None:
    if std_name in PREDICTOR_STD_TO_LUNAVOX:
        return PREDICTOR_STD_TO_LUNAVOX[std_name]
    m = re.match(r"blk\.(\d+)\.(.+)", std_name)
    if m:
        layer_idx = m.group(1)
        suffix = m.group(2)
        if suffix in TALKER_LAYER_STD_TO_LUNAVOX:
            return f"code_pred.blk.{layer_idx}.{TALKER_LAYER_STD_TO_LUNAVOX[suffix]}"
    # Codec embeddings and lm_heads pass through with code_pred prefix
    if std_name.startswith("codec_embd.") or std_name.startswith("lm_head."):
        return f"code_pred.{std_name}"
    return None


# ===========================================================================
# Decoder HF → lunavox tensor name mapping
# ===========================================================================

def map_decoder_hf_to_lunavox(hf_name: str) -> str | None:
    """Map decoder HF tensor name to lunavox tok_dec.* convention."""
    if not hf_name.startswith("decoder."):
        return None

    rest = hf_name[len("decoder."):]

    # --- Quantizer / VQ ---
    if rest.startswith("quantizer.rvq_first.input_proj.weight"):
        return "tok_dec.vq_first.input_proj.weight"
    if rest.startswith("quantizer.rvq_first.output_proj.weight"):
        return "tok_dec.vq_first.output_proj.weight"
    if rest.startswith("quantizer.rvq_rest.input_proj.weight"):
        return "tok_dec.vq_rest.input_proj.weight"
    if rest.startswith("quantizer.rvq_rest.output_proj.weight"):
        return "tok_dec.vq_rest.output_proj.weight"

    # rvq_first codebook: decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum
    m = re.match(r"quantizer\.rvq_first\.vq\.layers\.0\._codebook\.(.+)", rest)
    if m:
        field = m.group(1)
        if field == "embedding_sum":
            return "tok_dec.vq_first.0.codebook"
        if field == "cluster_usage":
            return "tok_dec.vq_first.0.usage"
        return None

    # rvq_rest codebook: decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.*
    m = re.match(r"quantizer\.rvq_rest\.vq\.layers\.(\d+)\._codebook\.(.+)", rest)
    if m:
        i = int(m.group(1))
        field = m.group(2)
        if field == "embedding_sum":
            return f"tok_dec.vq_rest.{i}.codebook"
        if field == "cluster_usage":
            return f"tok_dec.vq_rest.{i}.usage"
        return None

    # --- Pre-conv ---
    if rest == "pre_conv.conv.weight":
        return "tok_dec.pre_conv.weight"
    if rest == "pre_conv.conv.bias":
        return "tok_dec.pre_conv.bias"

    # --- Pre-transformer ---
    if rest == "pre_transformer.input_proj.weight":
        return "tok_dec.pre_tfm.input_proj.weight"
    if rest == "pre_transformer.input_proj.bias":
        return "tok_dec.pre_tfm.input_proj.bias"
    if rest == "pre_transformer.norm.weight":
        return "tok_dec.pre_tfm.norm.weight"
    if rest == "pre_transformer.output_proj.weight":
        return "tok_dec.pre_tfm.output_proj.weight"
    if rest == "pre_transformer.output_proj.bias":
        return "tok_dec.pre_tfm.output_proj.bias"

    # Pre-transformer layers
    m = re.match(r"pre_transformer\.layers\.(\d+)\.(.+)", rest)
    if m:
        blk = int(m.group(1))
        suffix = m.group(2)
        mapping = {
            "input_layernorm.weight":    "attn_norm.weight",
            "self_attn.q_proj.weight":   "attn_q.weight",
            "self_attn.k_proj.weight":   "attn_k.weight",
            "self_attn.v_proj.weight":   "attn_v.weight",
            "self_attn.o_proj.weight":   "attn_output.weight",
            "self_attn_layer_scale.scale": "attn_scale",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "mlp.gate_proj.weight":      "ffn_gate.weight",
            "mlp.up_proj.weight":        "ffn_up.weight",
            "mlp.down_proj.weight":      "ffn_down.weight",
            "mlp_layer_scale.scale":     "ffn_scale",
        }
        if suffix in mapping:
            return f"tok_dec.pre_tfm.blk.{blk}.{mapping[suffix]}"
        return None

    # --- Decoder blocks ---
    # decoder.decoder.0.conv.*  → tok_dec.dec.0.conv.*
    if rest == "decoder.0.conv.weight":
        return "tok_dec.dec.0.conv.weight"
    if rest == "decoder.0.conv.bias":
        return "tok_dec.dec.0.conv.bias"

    # decoder.decoder.{N}.block.0.alpha/beta → tok_dec.dec.{N}.snake.alpha/beta  (N=1..4)
    m = re.match(r"decoder\.(\d+)\.block\.0\.(alpha|beta)", rest)
    if m:
        blk = int(m.group(1))
        ab = m.group(2)
        return f"tok_dec.dec.{blk}.snake.{ab}"

    # decoder.decoder.{N}.block.1.conv.* → tok_dec.dec.{N}.conv_t.*  (transposed conv)
    m = re.match(r"decoder\.(\d+)\.block\.1\.conv\.(weight|bias)", rest)
    if m:
        blk = int(m.group(1))
        wb = m.group(2)
        return f"tok_dec.dec.{blk}.conv_t.{wb}"

    # decoder.decoder.{N}.block.{R}.act{A}.alpha/beta → tok_dec.dec.{N}.res.{R}.act{A}.alpha/beta
    m = re.match(r"decoder\.(\d+)\.block\.(\d+)\.(act[12])\.(alpha|beta)", rest)
    if m:
        blk = int(m.group(1))
        res = int(m.group(2))
        act = m.group(3)
        ab = m.group(4)
        return f"tok_dec.dec.{blk}.res.{res}.{act}.{ab}"

    # decoder.decoder.{N}.block.{R}.conv{C}.conv.* → tok_dec.dec.{N}.res.{R}.conv{C}.*
    m = re.match(r"decoder\.(\d+)\.block\.(\d+)\.(conv[12])\.conv\.(weight|bias)", rest)
    if m:
        blk = int(m.group(1))
        res = int(m.group(2))
        conv = m.group(3)
        wb = m.group(4)
        return f"tok_dec.dec.{blk}.res.{res}.{conv}.{wb}"

    # decoder.decoder.5.snake.alpha/beta → tok_dec.dec.5.snake.alpha/beta
    m = re.match(r"decoder\.5\.snake\.(alpha|beta)", rest)
    if m:
        return f"tok_dec.dec.5.snake.{m.group(1)}"

    # decoder.decoder.6.conv.* → tok_dec.dec.6.conv.*
    if rest == "decoder.6.conv.weight":
        return "tok_dec.dec.6.conv.weight"
    if rest == "decoder.6.conv.bias":
        return "tok_dec.dec.6.conv.bias"

    # --- Upsample blocks ---
    # decoder.upsample.{U}.0.conv.* → tok_dec.upsample.{U}.conv.*
    m = re.match(r"upsample\.(\d+)\.0\.conv\.(weight|bias)", rest)
    if m:
        u = int(m.group(1))
        wb = m.group(2)
        return f"tok_dec.upsample.{u}.conv.{wb}"

    # decoder.upsample.{U}.1.* → tok_dec.upsample.{U}.*
    m = re.match(r"upsample\.(\d+)\.1\.(.*)", rest)
    if m:
        u = int(m.group(1))
        sub = m.group(2)
        sub_map = {
            "dwconv.conv.weight": "dwconv.weight",
            "dwconv.conv.bias":   "dwconv.bias",
            "norm.weight":        "norm.weight",
            "norm.bias":          "norm.bias",
            "pwconv1.weight":     "pwconv1.weight",
            "pwconv1.bias":       "pwconv1.bias",
            "pwconv2.weight":     "pwconv2.weight",
            "pwconv2.bias":       "pwconv2.bias",
            "gamma":              "gamma",
        }
        if sub in sub_map:
            return f"tok_dec.upsample.{u}.{sub_map[sub]}"

    return None


# ===========================================================================
# Speaker Encoder HF → lunavox tensor name mapping
# ===========================================================================

def map_encoder_hf_to_lunavox(hf_name: str) -> str | None:
    """Map encoder HF tensor name to lunavox spk_enc.* convention."""
    if not hf_name.startswith("encoder."):
        return None
    rest = hf_name[len("encoder."):]
    return f"spk_enc.{rest}"


# ===========================================================================
# Read quantized tensors from a GGUF file
# ===========================================================================

def read_gguf_tensors(gguf_path: Path) -> list[tuple[str, np.ndarray, int]]:
    """Read all tensors from a GGUF file, preserving raw quantized data.

    Returns list of (tensor_name, raw_data_bytes, ggml_type).
    """
    from gguf import GGUFReader
    reader = GGUFReader(str(gguf_path))
    tensors = []
    for tensor in reader.tensors:
        name = tensor.name
        data = tensor.data
        tensor_type = tensor.tensor_type
        shape = tensor.shape
        tensors.append((name, data, tensor_type, shape))
    return tensors


# ===========================================================================
# Main merge logic
# ===========================================================================

def load_tokenizer_data(model_dir: Path) -> tuple[list[str], list[str]]:
    """Load vocab and merges from HF tokenizer files."""
    vocab_path = model_dir / "vocab.json"
    merges_path = model_dir / "merges.txt"

    with open(vocab_path, encoding="utf-8") as f:
        vocab_dict = json.load(f)

    # Sort by token ID to get ordered list
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


def main():
    parser = argparse.ArgumentParser(description="Merge quantized components into final lunavox GGUF")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="HF model directory")
    parser.add_argument("--tmp-dir", type=str, required=True,
                        help="Directory with quantized GGUFs")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Final output directory for GGUF files")
    parser.add_argument("--talker-quant", type=str, default="q5_k_m",
                        help="Talker quantization type tag (for filename matching)")
    parser.add_argument("--predictor-quant", type=str, default="q8_0",
                        help="Predictor quantization type tag (for filename matching)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tmp_dir = Path(args.tmp_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import gguf
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}", file=sys.stderr)
        sys.exit(1)

    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # ---------- Read quantized Talker ----------
    talker_gguf_path = tmp_dir / f"talker-{args.talker_quant}.gguf"
    if not talker_gguf_path.exists():
        print(f"ERROR: Quantized Talker not found: {talker_gguf_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[1/5] Reading quantized Talker from {talker_gguf_path}...")
    talker_tensors = read_gguf_tensors(talker_gguf_path)
    print(f"  Read {len(talker_tensors)} tensors")

    # ---------- Read quantized Predictor ----------
    predictor_gguf_path = tmp_dir / f"predictor-{args.predictor_quant}.gguf"
    if not predictor_gguf_path.exists():
        print(f"ERROR: Quantized Predictor not found: {predictor_gguf_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[2/5] Reading quantized Predictor from {predictor_gguf_path}...")
    predictor_tensors = read_gguf_tensors(predictor_gguf_path)
    print(f"  Read {len(predictor_tensors)} tensors")

    # ---------- Read Decoder/Encoder from speech_tokenizer ----------
    print(f"[3/5] Reading Decoder/Encoder weights from speech_tokenizer...")
    speech_tok_path = model_dir / "speech_tokenizer" / "model.safetensors"
    if not speech_tok_path.exists():
        # Fallback: try main model file
        speech_tok_path = model_dir / "model.safetensors"
        print(f"  (fallback to main model.safetensors)")
    decoder_tensors = {}
    encoder_tensors = {}

    with safe_open(str(speech_tok_path), framework="pt") as f:
        for hf_name in f.keys():
            lunavox_name = map_decoder_hf_to_lunavox(hf_name)
            if lunavox_name is not None:
                decoder_tensors[lunavox_name] = load_tensor_as_numpy(f, hf_name)
            enc_name = map_encoder_hf_to_lunavox(hf_name)
            if enc_name is not None:
                encoder_tensors[enc_name] = load_tensor_as_numpy(f, hf_name)

    print(f"  Mapped {len(decoder_tensors)} Decoder tensors")
    print(f"  Mapped {len(encoder_tensors)} Encoder tensors")

    # ---------- Read tokenizer ----------
    print(f"[4/5] Reading tokenizer data...")
    tokens, merges = load_tokenizer_data(model_dir)
    print(f"  Vocab size: {len(tokens)}, Merges: {len(merges)}")

    # ---------- Write main GGUF ----------
    print(f"[5/5] Writing main GGUF...")
    main_output = output_dir / "qwen3-tts-0.6B-base.gguf"

    # Architecture for the writer (custom for lunavox)
    writer = gguf.GGUFWriter(str(main_output), "qwen3-tts")

    # --- Write metadata ---
    # Config paths: config.json uses "talker_config" and "code_predictor_config"
    talker_cfg = config.get("talker_config", {})
    code_pred_cfg = talker_cfg.get("code_predictor_config", {})

    # Text config
    text_vocab_size = talker_cfg.get("text_vocab_size", 151936)
    text_embd_dim = talker_cfg.get("text_hidden_size", 2048)
    writer.add_uint32("qwen3-tts.text.vocab_size", text_vocab_size)
    writer.add_uint32("qwen3-tts.text.embedding_dim", text_embd_dim)

    # Talker config
    hidden_size = talker_cfg.get("hidden_size", 1024)
    n_layers = talker_cfg.get("num_hidden_layers", 28)
    n_heads = talker_cfg.get("num_attention_heads", 16)
    n_kv_heads = talker_cfg.get("num_key_value_heads", 8)
    intermediate_size = talker_cfg.get("intermediate_size", 3072)
    head_dim = talker_cfg.get("head_dim", 128)  # Explicit from config, NOT hidden_size/num_heads!
    rms_norm_eps = talker_cfg.get("rms_norm_eps", 1e-6)
    rope_theta = talker_cfg.get("rope_theta", 1000000.0)
    codec_vocab_size = talker_cfg.get("vocab_size", 3072)
    n_codebooks = talker_cfg.get("num_code_groups", 16)

    # M-RoPE section
    rope_scaling = talker_cfg.get("rope_scaling", {})
    mrope_sections = rope_scaling.get("mrope_section", [24, 20, 20])

    writer.add_uint32("qwen3-tts.talker.embedding_length", hidden_size)
    writer.add_uint32("qwen3-tts.talker.block_count", n_layers)
    writer.add_uint32("qwen3-tts.talker.attention.head_count", n_heads)
    writer.add_uint32("qwen3-tts.talker.attention.head_count_kv", n_kv_heads)
    writer.add_uint32("qwen3-tts.talker.feed_forward_length", intermediate_size)
    writer.add_uint32("qwen3-tts.talker.attention.key_length", head_dim)
    writer.add_uint32("qwen3-tts.talker.attention.value_length", head_dim)
    writer.add_float32("qwen3-tts.talker.attention.layer_norm_rms_epsilon", rms_norm_eps)
    writer.add_float32("qwen3-tts.talker.rope.freq_base", rope_theta)
    writer.add_uint32("qwen3-tts.talker.codec_vocab_size", codec_vocab_size)
    writer.add_uint32("qwen3-tts.talker.num_codebooks", n_codebooks)
    writer.add_uint32("qwen3-tts.rope.mrope_section", mrope_sections[1])  # matches old model's value

    # Code predictor config
    cp_n_layers = code_pred_cfg.get("num_hidden_layers", 5)
    cp_vocab_size = code_pred_cfg.get("vocab_size", 2048)
    writer.add_uint32("qwen3-tts.code_pred.layer_count", cp_n_layers)
    writer.add_uint32("qwen3-tts.code_pred.vocab_size", cp_vocab_size)

    # Speaker encoder config
    spk_cfg = config.get("speaker_encoder_config", {})
    writer.add_uint32("qwen3-tts.speaker_encoder.embedding_length",
                      spk_cfg.get("enc_dim", 1024))
    writer.add_uint32("qwen3-tts.speaker_encoder.sample_rate",
                      spk_cfg.get("sample_rate", 24000))

    # Special token IDs (match config.json key names)
    codec_bos = talker_cfg.get("codec_bos_id", 2149)
    codec_eos = talker_cfg.get("codec_eos_token_id", 2150)
    codec_pad = talker_cfg.get("codec_pad_id", 2148)
    tts_bos = config.get("tts_bos_token_id", 151672)
    tts_eos = config.get("tts_eos_token_id", 151673)
    tts_pad = config.get("tts_pad_token_id", 151671)
    writer.add_uint32("qwen3-tts.codec.bos_id", codec_bos)
    writer.add_uint32("qwen3-tts.codec.eos_id", codec_eos)
    writer.add_uint32("qwen3-tts.codec.pad_id", codec_pad)
    writer.add_uint32("qwen3-tts.tts_bos_token_id", tts_bos)
    writer.add_uint32("qwen3-tts.tts_eos_token_id", tts_eos)
    writer.add_uint32("qwen3-tts.tts_pad_token_id", tts_pad)

    # Tokenizer audio config
    writer.add_uint32("qwen3-tts.tokenizer.sample_rate", 24000)
    writer.add_uint32("qwen3-tts.tokenizer.num_codebooks", n_codebooks)
    writer.add_uint32("qwen3-tts.tokenizer.codebook_size", 2048)

    # Think tokens
    writer.add_uint32("qwen3-tts.codec.think_id", talker_cfg.get("codec_think_id", 2154))
    writer.add_uint32("qwen3-tts.codec.nothink_id", talker_cfg.get("codec_nothink_id", 2155))
    writer.add_uint32("qwen3-tts.codec.think_bos_id", talker_cfg.get("codec_think_bos_id", 2156))
    writer.add_uint32("qwen3-tts.codec.think_eos_id", talker_cfg.get("codec_think_eos_id", 2157))

    # Language
    codec_lang = talker_cfg.get("codec_language_id", {})
    writer.add_uint32("qwen3-tts.language.english_id", codec_lang.get("english", 2050))

    # --- Write tokenizer ---
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_token_list(tokens)
    writer.add_token_merges(merges)

    # Find special token IDs in vocab
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        with open(tokenizer_config_path) as f:
            tok_config = json.load(f)
        # BOS/EOS for text tokenizer
        added_tokens = tok_config.get("added_tokens_decoder", {})
        for tid, info in added_tokens.items():
            content = info.get("content", "")
            if content == "<|im_start|>":
                writer.add_uint32("tokenizer.ggml.bos_token_id", int(tid))
            elif content == "<|im_end|>":
                writer.add_uint32("tokenizer.ggml.eos_token_id", int(tid))

    # --- Write Talker tensors (quantized) ---
    talker_count = 0
    for std_name, data, tensor_type, shape in talker_tensors:
        lunavox_name = map_talker_std_to_lunavox(std_name)
        if lunavox_name is None:
            print(f"  WARNING: Unmapped Talker tensor: {std_name}")
            continue
        raw_dtype = gguf.GGMLQuantizationType(tensor_type)
        writer.add_tensor(lunavox_name, data, raw_dtype=raw_dtype)
        talker_count += 1

    print(f"  Added {talker_count} Talker tensors")

    # --- Write Predictor tensors (quantized) ---
    pred_count = 0
    for std_name, data, tensor_type, shape in predictor_tensors:
        lunavox_name = map_predictor_std_to_lunavox(std_name)
        if lunavox_name is None:
            print(f"  WARNING: Unmapped Predictor tensor: {std_name}")
            continue
        raw_dtype = gguf.GGMLQuantizationType(tensor_type)
        writer.add_tensor(lunavox_name, data, raw_dtype=raw_dtype)
        pred_count += 1

    print(f"  Added {pred_count} Predictor tensors")

    # --- Write Decoder tensors (F16) ---
    dec_count = 0
    for name, data in sorted(decoder_tensors.items()):
        if data.ndim >= 2:
            writer.add_tensor(name, data.astype(np.float16))
        else:
            writer.add_tensor(name, data.astype(np.float32))
        dec_count += 1

    print(f"  Added {dec_count} Decoder tensors")

    # Finalize
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    main_size = main_output.stat().st_size / 1024 / 1024
    total_tensors = talker_count + pred_count + dec_count
    print(f"\n  Main model: {main_output} ({main_size:.1f} MB, {total_tensors} tensors)")

    # ---------- Write auxiliary GGUF (Speaker Encoder) ----------
    aux_output = output_dir / "qwen3-tts-aux-f16.gguf"
    print(f"\nWriting auxiliary GGUF (Speaker Encoder)...")
    aux_writer = gguf.GGUFWriter(str(aux_output), "qwen3-tts-aux")

    # Minimal metadata for aux model
    aux_writer.add_uint32("qwen3-tts.tokenizer.sample_rate", 24000)
    aux_writer.add_uint32("qwen3-tts.tokenizer.num_codebooks", n_codebooks)
    aux_writer.add_uint32("qwen3-tts.tokenizer.codebook_size", 2048)

    enc_count = 0
    for name, data in sorted(encoder_tensors.items()):
        if data.ndim >= 2:
            aux_writer.add_tensor(name, data.astype(np.float16))
        else:
            aux_writer.add_tensor(name, data.astype(np.float32))
        enc_count += 1

    aux_writer.write_header_to_file()
    aux_writer.write_kv_data_to_file()
    aux_writer.write_tensors_to_file()
    aux_writer.close()

    aux_size = aux_output.stat().st_size / 1024 / 1024
    print(f"  Aux model: {aux_output} ({aux_size:.1f} MB, {enc_count} tensors)")
    print(f"\n[done] All GGUF files written successfully.")


if __name__ == "__main__":
    main()
