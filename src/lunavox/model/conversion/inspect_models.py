#!/usr/bin/env python3
"""
Inspect Qwen3-TTS models: tensor names, shapes, dtypes, and architecture.
Outputs comprehensive information for GGUF conversion planning.
"""

import json
import os
from pathlib import Path

from safetensors import safe_open


def print_separator(title: str, char: str = "=", width: int = 80):
    """Print a formatted section separator."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}\n")


def inspect_safetensors(model_path: Path, model_name: str):
    """Load and inspect a safetensors model file."""
    print_separator(f"TENSORS: {model_name}", "=")

    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return {}

    tensor_info = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        tensor_names = sorted(f.keys())
        print(f"Total tensors: {len(tensor_names)}\n")
        print(f"{'Tensor Name':<70} {'Shape':<25} {'Dtype':<15}")
        print("-" * 110)

        for name in tensor_names:
            tensor = f.get_tensor(name)
            shape = tuple(tensor.shape)
            dtype = str(tensor.dtype)
            print(f"{name:<70} {str(shape):<25} {dtype:<15}")
            tensor_info[name] = {"shape": shape, "dtype": dtype}

    return tensor_info


def inspect_config(config_path: Path, config_name: str):
    """Load and display config.json contents."""
    print_separator(f"CONFIG: {config_name}", "=")

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return {}

    with open(config_path) as f:
        config = json.load(f)

    print(json.dumps(config, indent=2))
    return config


def analyze_architecture(config: dict, name: str):
    """Extract and display key architecture parameters."""
    print_separator(f"ARCHITECTURE ANALYSIS: {name}", "-")

    # Top-level info
    if "architectures" in config:
        print(f"Architecture: {config['architectures']}")
    if "model_type" in config:
        print(f"Model Type: {config['model_type']}")

    # TTS Base model specific
    if "talker_config" in config:
        tc = config["talker_config"]
        print("\n--- Talker (Main TTS Transformer) ---")
        print(f"  Hidden Size: {tc.get('hidden_size')}")
        print(f"  Intermediate Size: {tc.get('intermediate_size')}")
        print(f"  Num Hidden Layers: {tc.get('num_hidden_layers')}")
        print(f"  Num Attention Heads: {tc.get('num_attention_heads')}")
        print(f"  Num KV Heads: {tc.get('num_key_value_heads')}")
        print(f"  Head Dim: {tc.get('head_dim')}")
        print(f"  Vocab Size: {tc.get('vocab_size')}")
        print(f"  Text Vocab Size: {tc.get('text_vocab_size')}")
        print(f"  Text Hidden Size: {tc.get('text_hidden_size')}")
        print(f"  Num Code Groups (Codebooks): {tc.get('num_code_groups')}")
        print(f"  RMS Norm Eps: {tc.get('rms_norm_eps')}")
        print(f"  RoPE Theta: {tc.get('rope_theta')}")
        print(f"  Max Position Embeddings: {tc.get('max_position_embeddings')}")
        print(f"  Hidden Act: {tc.get('hidden_act')}")

        if "rope_scaling" in tc:
            rs = tc["rope_scaling"]
            print(f"  RoPE Scaling: {rs}")

        # Code predictor
        if "code_predictor_config" in tc:
            cp = tc["code_predictor_config"]
            print("\n--- Code Predictor (Delay Pattern Transformer) ---")
            print(f"  Hidden Size: {cp.get('hidden_size')}")
            print(f"  Intermediate Size: {cp.get('intermediate_size')}")
            print(f"  Num Hidden Layers: {cp.get('num_hidden_layers')}")
            print(f"  Num Attention Heads: {cp.get('num_attention_heads')}")
            print(f"  Num KV Heads: {cp.get('num_key_value_heads')}")
            print(f"  Head Dim: {cp.get('head_dim')}")
            print(f"  Vocab Size: {cp.get('vocab_size')}")
            print(f"  Num Code Groups: {cp.get('num_code_groups')}")
            print(f"  Layer Types: {cp.get('layer_types')}")

    # Speaker encoder
    if "speaker_encoder_config" in config:
        se = config["speaker_encoder_config"]
        print("\n--- Speaker Encoder ---")
        print(f"  Enc Dim: {se.get('enc_dim')}")
        print(f"  Sample Rate: {se.get('sample_rate')}")

    # Tokenizer specific
    if "encoder_config" in config:
        ec = config["encoder_config"]
        print("\n--- Tokenizer Encoder (Audio -> Codes) ---")
        print(f"  Frame Rate: {ec.get('_frame_rate')}")
        print(f"  Hidden Size: {ec.get('hidden_size')}")
        print(f"  Intermediate Size: {ec.get('intermediate_size')}")
        print(f"  Num Hidden Layers: {ec.get('num_hidden_layers')}")
        print(f"  Num Attention Heads: {ec.get('num_attention_heads')}")
        print(f"  Num KV Heads: {ec.get('num_key_value_heads')}")
        print(f"  Head Dim: {ec.get('head_dim')}")
        print(f"  Codebook Size: {ec.get('codebook_size')}")
        print(f"  Codebook Dim: {ec.get('codebook_dim')}")
        print(f"  Num Quantizers: {ec.get('num_quantizers')}")
        print(f"  Num Semantic Quantizers: {ec.get('num_semantic_quantizers')}")
        print(f"  Sampling Rate: {ec.get('sampling_rate')}")
        print(f"  Upsampling Ratios: {ec.get('upsampling_ratios')}")
        print(f"  Num Filters: {ec.get('num_filters')}")
        print(f"  Kernel Size: {ec.get('kernel_size')}")
        print(f"  Use Causal Conv: {ec.get('use_causal_conv')}")

    if "decoder_config" in config:
        dc = config["decoder_config"]
        print("\n--- Tokenizer Decoder (Codes -> Audio, Vocoder) ---")
        print(f"  Latent Dim: {dc.get('latent_dim')}")
        print(f"  Decoder Dim: {dc.get('decoder_dim')}")
        print(f"  Hidden Size: {dc.get('hidden_size')}")
        print(f"  Intermediate Size: {dc.get('intermediate_size')}")
        print(f"  Num Hidden Layers: {dc.get('num_hidden_layers')}")
        print(f"  Num Attention Heads: {dc.get('num_attention_heads')}")
        print(f"  Num KV Heads: {dc.get('num_key_value_heads')}")
        print(f"  Head Dim: {dc.get('head_dim')}")
        print(f"  Codebook Size: {dc.get('codebook_size')}")
        print(f"  Codebook Dim: {dc.get('codebook_dim')}")
        print(f"  Semantic Codebook Size: {dc.get('semantic_codebook_size')}")
        print(f"  Num Quantizers: {dc.get('num_quantizers')}")
        print(f"  Num Semantic Quantizers: {dc.get('num_semantic_quantizers')}")
        print(f"  Upsample Rates: {dc.get('upsample_rates')}")
        print(f"  Upsampling Ratios: {dc.get('upsampling_ratios')}")
        print(f"  Sliding Window: {dc.get('sliding_window')}")
        print(f"  VQ Hidden Dim: {dc.get('vector_quantization_hidden_dimension')}")

    # Top-level tokenizer params
    if "encoder_valid_num_quantizers" in config:
        print("\n--- Tokenizer Global ---")
        print(f"  Valid Num Quantizers: {config.get('encoder_valid_num_quantizers')}")
        print(f"  Input Sample Rate: {config.get('input_sample_rate')}")
        print(f"  Output Sample Rate: {config.get('output_sample_rate')}")
        print(f"  Encode Downsample Rate: {config.get('encode_downsample_rate')}")
        print(f"  Decode Upsample Rate: {config.get('decode_upsample_rate')}")


def categorize_tensors(tensor_info: dict, model_name: str):
    """Categorize tensors by component for easier mapping."""
    print_separator(f"TENSOR CATEGORIES: {model_name}", "-")

    categories = {}
    for name in tensor_info:
        # Determine category from tensor name prefix
        parts = name.split(".")
        if len(parts) >= 2:
            category = ".".join(parts[:2])
        else:
            category = parts[0]

        if category not in categories:
            categories[category] = []
        categories[category].append(name)

    for category, tensors in sorted(categories.items()):
        print(f"\n{category} ({len(tensors)} tensors):")
        for t in sorted(tensors)[:5]:  # Show first 5
            print(f"  - {t}")
        if len(tensors) > 5:
            print(f"  ... and {len(tensors) - 5} more")

    return categories


def main():
    project_root = Path(
        os.environ.get("LUNAVOX_PROJECT_ROOT", str(Path(__file__).resolve().parents[4]))
    ).resolve()
    base_dir = project_root / "models"

    # Model paths
    tts_base_dir = base_dir / "Qwen3-TTS-12Hz-0.6B-Base"
    tokenizer_dir = base_dir / "Qwen3-TTS-Tokenizer-12Hz"

    print_separator("QWEN3-TTS MODEL INSPECTION REPORT", "#", 80)
    print("This report documents all tensors, shapes, and architecture details")
    print("for GGUF conversion planning.\n")

    # ========== TTS BASE MODEL ==========
    print_separator("TTS BASE MODEL (Qwen3-TTS-12Hz-0.6B-Base)", "#")

    # Main model
    tts_config = inspect_config(tts_base_dir / "config.json", "TTS Base")
    analyze_architecture(tts_config, "TTS Base")
    tts_tensors = inspect_safetensors(tts_base_dir / "model.safetensors", "TTS Base Main")
    categorize_tensors(tts_tensors, "TTS Base Main")

    # Speech tokenizer embedded in TTS base
    speech_tok_config = inspect_config(
        tts_base_dir / "speech_tokenizer" / "config.json", "TTS Base Speech Tokenizer"
    )
    analyze_architecture(speech_tok_config, "TTS Base Speech Tokenizer")
    speech_tok_tensors = inspect_safetensors(
        tts_base_dir / "speech_tokenizer" / "model.safetensors", "TTS Base Speech Tokenizer"
    )
    categorize_tensors(speech_tok_tensors, "TTS Base Speech Tokenizer")

    # ========== STANDALONE TOKENIZER ==========
    print_separator("STANDALONE TOKENIZER (Qwen3-TTS-Tokenizer-12Hz)", "#")

    tokenizer_config = inspect_config(tokenizer_dir / "config.json", "Standalone Tokenizer")
    analyze_architecture(tokenizer_config, "Standalone Tokenizer")
    tokenizer_tensors = inspect_safetensors(
        tokenizer_dir / "model.safetensors", "Standalone Tokenizer"
    )
    categorize_tensors(tokenizer_tensors, "Standalone Tokenizer")

    # ========== SUMMARY ==========
    print_separator("SUMMARY", "#")
    print(f"TTS Base Main Model: {len(tts_tensors)} tensors")
    print(f"TTS Base Speech Tokenizer: {len(speech_tok_tensors)} tensors")
    print(f"Standalone Tokenizer: {len(tokenizer_tensors)} tensors")

    print("\n--- Key Architecture Parameters ---")
    if "talker_config" in tts_config:
        tc = tts_config["talker_config"]
        print(
            f"TTS Talker: {tc.get('num_hidden_layers')} layers, "
            f"{tc.get('hidden_size')} hidden, "
            f"{tc.get('num_attention_heads')} heads, "
            f"{tc.get('num_code_groups')} codebooks"
        )
        if "code_predictor_config" in tc:
            cp = tc["code_predictor_config"]
            print(
                f"Code Predictor: {cp.get('num_hidden_layers')} layers, "
                f"{cp.get('hidden_size')} hidden, "
                f"{cp.get('num_code_groups')} codebooks"
            )

    if "encoder_config" in tokenizer_config:
        ec = tokenizer_config["encoder_config"]
        print(
            f"Tokenizer Encoder: {ec.get('num_hidden_layers')} layers, "
            f"{ec.get('hidden_size')} hidden, "
            f"{ec.get('num_quantizers')} quantizers"
        )

    if "decoder_config" in tokenizer_config:
        dc = tokenizer_config["decoder_config"]
        print(
            f"Tokenizer Decoder: {dc.get('num_hidden_layers')} layers, "
            f"{dc.get('decoder_dim')} decoder dim, "
            f"{dc.get('num_quantizers')} quantizers"
        )


if __name__ == "__main__":
    main()
