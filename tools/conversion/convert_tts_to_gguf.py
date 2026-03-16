#!/usr/bin/env python3
"""
Convert HuggingFace Qwen3-TTS-12Hz-0.6B-Base model to GGUF format.

Usage:
    python tools/conversion/convert_tts_to_gguf.py \
        --input models/Qwen3-TTS-12Hz-0.6B-Base \
        --output models/qwen3-tts-0.6B-base.gguf \
        --talker-type q5_k \
        --predictor-type q8_0 \
        --speaker-type f16
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

# Add gguf-py to path (if available)
GGUF_PY_PATH = Path(__file__).resolve().parents[2] / "gguf-py"
try:
    if GGUF_PY_PATH.exists():
        sys.path.insert(0, str(GGUF_PY_PATH))
except (PermissionError, OSError):
    pass

import gguf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Qwen3TTSConverter:
    """Converter for Qwen3-TTS-12Hz-0.6B-Base model to GGUF format."""

    SUPPORTED_TYPES = {"f32", "f16", "q8_0", "q4_k", "q5_k"}

    QUANT_TYPE_MAP = {
        "f32": gguf.GGMLQuantizationType.F32,
        "f16": gguf.GGMLQuantizationType.F16,
        "q8_0": gguf.GGMLQuantizationType.Q8_0,
        "q4_k": gguf.GGMLQuantizationType.Q4_K,
        "q5_k": gguf.GGMLQuantizationType.Q5_K,
    }

    FILE_TYPE_MAP = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q4_k": gguf.LlamaFileType.MOSTLY_Q4_K_M,
        "q5_k": gguf.LlamaFileType.MOSTLY_Q5_K_M,
    }

    # Keep in sync with enum ggml_type in ggml/include/ggml.h.
    GGML_TYPE_MAP = {
        "f32": 0,
        "f16": 1,
        "q8_0": 8,
        "q4_k": 12,
        "q5_k": 13,
    }

    # Direct tensor name mapping from HuggingFace to GGML conventions
    TENSOR_MAP = {
        # Talker - Main embeddings and heads
        "talker.model.codec_embedding.weight": "talker.codec_embd.weight",
        "talker.model.text_embedding.weight": "talker.text_embd.weight",
        "talker.codec_head.weight": "talker.codec_head.weight",
        "talker.model.norm.weight": "talker.output_norm.weight",
        # Talker - Text projection
        "talker.text_projection.linear_fc1.weight": "talker.text_proj.fc1.weight",
        "talker.text_projection.linear_fc1.bias": "talker.text_proj.fc1.bias",
        "talker.text_projection.linear_fc2.weight": "talker.text_proj.fc2.weight",
        "talker.text_projection.linear_fc2.bias": "talker.text_proj.fc2.bias",
        # Code Predictor - Output norm
        "talker.code_predictor.model.norm.weight": "code_pred.output_norm.weight",
        # Speaker Encoder - Initial conv
        "speaker_encoder.blocks.0.conv.weight": "spk_enc.conv0.weight",
        "speaker_encoder.blocks.0.conv.bias": "spk_enc.conv0.bias",
        # Speaker Encoder - ASP (Attentive Statistics Pooling)
        "speaker_encoder.asp.conv.weight": "spk_enc.asp.conv.weight",
        "speaker_encoder.asp.conv.bias": "spk_enc.asp.conv.bias",
        "speaker_encoder.asp.tdnn.conv.weight": "spk_enc.asp.tdnn.weight",
        "speaker_encoder.asp.tdnn.conv.bias": "spk_enc.asp.tdnn.bias",
        # Speaker Encoder - MFA (Multi-layer Feature Aggregation)
        "speaker_encoder.mfa.conv.weight": "spk_enc.mfa.weight",
        "speaker_encoder.mfa.conv.bias": "spk_enc.mfa.bias",
        # Speaker Encoder - Final FC
        "speaker_encoder.fc.weight": "spk_enc.fc.weight",
        "speaker_encoder.fc.bias": "spk_enc.fc.bias",
    }

    # Regex patterns for layer-specific tensors
    TALKER_LAYER_PATTERNS = [
        # Talker transformer layers (28 layers)
        (r"talker\.model\.layers\.(\d+)\.input_layernorm\.weight", "talker.blk.{}.attn_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight", "talker.blk.{}.attn_q.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight", "talker.blk.{}.attn_k.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight", "talker.blk.{}.attn_v.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight", "talker.blk.{}.attn_output.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.q_norm\.weight", "talker.blk.{}.attn_q_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.k_norm\.weight", "talker.blk.{}.attn_k_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.post_attention_layernorm\.weight", "talker.blk.{}.ffn_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight", "talker.blk.{}.ffn_gate.weight"),
        (r"talker\.model\.layers\.(\d+)\.mlp\.up_proj\.weight", "talker.blk.{}.ffn_up.weight"),
        (r"talker\.model\.layers\.(\d+)\.mlp\.down_proj\.weight", "talker.blk.{}.ffn_down.weight"),
    ]

    CODE_PREDICTOR_LAYER_PATTERNS = [
        # Code Predictor transformer layers (5 layers)
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.input_layernorm\.weight", "code_pred.blk.{}.attn_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight", "code_pred.blk.{}.attn_q.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight", "code_pred.blk.{}.attn_k.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight", "code_pred.blk.{}.attn_v.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight", "code_pred.blk.{}.attn_output.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.q_norm\.weight", "code_pred.blk.{}.attn_q_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.k_norm\.weight", "code_pred.blk.{}.attn_k_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.post_attention_layernorm\.weight", "code_pred.blk.{}.ffn_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight", "code_pred.blk.{}.ffn_gate.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.mlp\.up_proj\.weight", "code_pred.blk.{}.ffn_up.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.mlp\.down_proj\.weight", "code_pred.blk.{}.ffn_down.weight"),
    ]

    CODE_PREDICTOR_CODEBOOK_PATTERNS = [
        # Code Predictor codebook embeddings (15 codebooks, indices 0-14)
        (r"talker\.code_predictor\.model\.codec_embedding\.(\d+)\.weight", "code_pred.codec_embd.{}.weight"),
        # Code Predictor LM heads (15 heads)
        (r"talker\.code_predictor\.lm_head\.(\d+)\.weight", "code_pred.lm_head.{}.weight"),
    ]

    SPEAKER_ENCODER_PATTERNS = [
        # Speaker Encoder Res2Net blocks (blocks 1-3)
        (r"speaker_encoder\.blocks\.(\d+)\.res2net_block\.blocks\.(\d+)\.conv\.weight", "spk_enc.blk.{}.res2net.{}.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.res2net_block\.blocks\.(\d+)\.conv\.bias", "spk_enc.blk.{}.res2net.{}.bias"),
        # Speaker Encoder SE blocks
        (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv1\.weight", "spk_enc.blk.{}.se.conv1.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv1\.bias", "spk_enc.blk.{}.se.conv1.bias"),
        (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv2\.weight", "spk_enc.blk.{}.se.conv2.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv2\.bias", "spk_enc.blk.{}.se.conv2.bias"),
        # Speaker Encoder TDNN layers
        (r"speaker_encoder\.blocks\.(\d+)\.tdnn1\.conv\.weight", "spk_enc.blk.{}.tdnn1.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.tdnn1\.conv\.bias", "spk_enc.blk.{}.tdnn1.bias"),
        (r"speaker_encoder\.blocks\.(\d+)\.tdnn2\.conv\.weight", "spk_enc.blk.{}.tdnn2.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.tdnn2\.conv\.bias", "spk_enc.blk.{}.tdnn2.bias"),
    ]

    def __init__(
        self,
        input_dir: Path,
        output_path: Path,
        output_type: str = "f16",
        talker_type: str | None = None,
        predictor_type: str | None = None,
        speaker_type: str | None = None,
    ):
        self.input_dir = input_dir
        self.output_path = output_path
        self.output_type = output_type.lower()
        self.quant_policy = {
            "talker": (talker_type or self.output_type).lower(),
            "code_pred": (predictor_type or self.output_type).lower(),
            "spk_enc": (speaker_type or self.output_type).lower(),
        }
        self._ggml_quant_lib: ctypes.CDLL | None = None
        self._ggml_quant_lib_path: Path | None = None
        self._validate_quant_policy()
        if "q5_k" in self.quant_policy.values():
            self._init_ggml_quant_lib()

        # Load config
        self.config = self._load_config()

        # Extract model parameters
        self._extract_params()

    def _load_config(self) -> dict[str, Any]:
        """Load model configuration from config.json."""
        config_path = self.input_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_params(self) -> None:
        """Extract model parameters from config."""
        talker_config = self.config.get("talker_config", {})
        code_predictor_config = talker_config.get("code_predictor_config", {})
        speaker_encoder_config = self.config.get("speaker_encoder_config", {})

        # Talker parameters
        self.hidden_size = talker_config.get("hidden_size", 1024)
        self.intermediate_size = talker_config.get("intermediate_size", 3072)
        self.num_hidden_layers = talker_config.get("num_hidden_layers", 28)
        self.num_attention_heads = talker_config.get("num_attention_heads", 16)
        self.num_kv_heads = talker_config.get("num_key_value_heads", 8)
        self.head_dim = talker_config.get("head_dim", 128)
        self.vocab_size = talker_config.get("vocab_size", 3072)  # codec vocab
        self.text_vocab_size = talker_config.get("text_vocab_size", 151936)
        self.text_hidden_size = talker_config.get("text_hidden_size", 2048)
        self.num_code_groups = talker_config.get("num_code_groups", 16)
        self.rms_norm_eps = talker_config.get("rms_norm_eps", 1e-6)
        self.rope_theta = talker_config.get("rope_theta", 1000000)

        # M-RoPE configuration
        rope_scaling = talker_config.get("rope_scaling", {})
        self.mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])

        # Code Predictor parameters
        self.code_predictor_num_layers = code_predictor_config.get("num_hidden_layers", 5)
        self.code_predictor_vocab_size = code_predictor_config.get("vocab_size", 2048)

        # Speaker Encoder parameters
        self.speaker_enc_dim = speaker_encoder_config.get("enc_dim", 1024)
        self.speaker_sample_rate = speaker_encoder_config.get("sample_rate", 24000)

        # Special codec token IDs
        self.codec_pad_id = talker_config.get("codec_pad_id", 2148)
        self.codec_bos_id = talker_config.get("codec_bos_id", 2149)
        self.codec_eos_id = talker_config.get("codec_eos_token_id", 2150)

        # Model name
        self.model_name = "Qwen3-TTS-12Hz-0.6B"

    def _validate_quant_policy(self) -> None:
        """Validate quantization policy and type support."""
        for module_name, quant_type in self.quant_policy.items():
            if quant_type not in self.SUPPORTED_TYPES:
                supported = ", ".join(sorted(self.SUPPORTED_TYPES))
                raise ValueError(
                    f"Unsupported quantization type '{quant_type}' for module '{module_name}'. "
                    f"Supported types: {supported}"
                )

            # Ensure requested quantization type exists in current gguf package.
            if quant_type not in self.QUANT_TYPE_MAP:
                raise ValueError(f"Quantization type '{quant_type}' is not available in current gguf package")

    def _init_ggml_quant_lib(self) -> None:
        """Load ggml-base library for quantization types missing in gguf-py (e.g., Q5_K)."""
        if self._ggml_quant_lib is not None:
            return

        repo_root = Path(__file__).resolve().parents[2]
        env_path = os.environ.get("QWEN3_TTS_GGML_BASE_LIB", "").strip()
        candidates: list[Path] = []

        if env_path:
            candidates.append(Path(env_path))

        if sys.platform == "win32":
            candidates.append(repo_root / "ggml" / "build" / "bin" / "ggml-base.dll")
        elif sys.platform == "darwin":
            candidates.append(repo_root / "ggml" / "build" / "src" / "libggml-base.dylib")
            candidates.append(repo_root / "ggml" / "build" / "bin" / "libggml-base.dylib")
        else:
            candidates.append(repo_root / "ggml" / "build" / "src" / "libggml-base.so")
            candidates.append(repo_root / "ggml" / "build" / "bin" / "libggml-base.so")

        last_err: Exception | None = None
        for path in candidates:
            if not path.exists():
                continue
            try:
                lib = ctypes.CDLL(str(path))
                lib.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
                lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
                lib.ggml_blck_size.argtypes = [ctypes.c_int]
                lib.ggml_blck_size.restype = ctypes.c_longlong
                lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_longlong]
                lib.ggml_row_size.restype = ctypes.c_size_t
                lib.ggml_quantize_chunk.argtypes = [
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_void_p,
                    ctypes.c_longlong,
                    ctypes.c_longlong,
                    ctypes.c_longlong,
                    ctypes.POINTER(ctypes.c_float),
                ]
                lib.ggml_quantize_chunk.restype = ctypes.c_size_t
                self._ggml_quant_lib = lib
                self._ggml_quant_lib_path = path
                logger.info("Using ggml quantization backend from %s", path)
                return
            except Exception as err:
                last_err = err

        raise RuntimeError(
            "Q5_K quantization requires ggml-base runtime library, but it could not be loaded. "
            f"Tried: {[str(p) for p in candidates]}. "
            "Build ggml first or set QWEN3_TTS_GGML_BASE_LIB to the library path."
            + (f" Last error: {last_err}" if last_err else "")
        )

    def _quantize_with_ggml(
        self,
        data: np.ndarray,
        target_type: str,
        tensor_name: str,
    ) -> np.ndarray:
        """Quantize a 2D float matrix using ggml runtime quantize API."""
        self._init_ggml_quant_lib()
        assert self._ggml_quant_lib is not None

        if data.ndim != 2:
            raise RuntimeError(f"ggml quantization expects 2D tensors, got {data.ndim}D for '{tensor_name}'")

        ggml_type = self.GGML_TYPE_MAP[target_type]
        if self._ggml_quant_lib.ggml_quantize_requires_imatrix(ggml_type):
            raise RuntimeError(
                f"Quantization type '{target_type}' requires an importance matrix, unsupported for '{tensor_name}'"
            )

        nrows = int(data.shape[0])
        n_per_row = int(data.shape[1])
        block_size = int(self._ggml_quant_lib.ggml_blck_size(ggml_type))
        if block_size <= 0 or n_per_row % block_size != 0:
            raise RuntimeError(
                f"Tensor '{tensor_name}' shape {data.shape} is incompatible with {target_type} "
                f"(n_per_row={n_per_row}, block_size={block_size})"
            )

        row_size = int(self._ggml_quant_lib.ggml_row_size(ggml_type, n_per_row))
        src = np.ascontiguousarray(data, dtype=np.float32)
        dst = np.empty((nrows * row_size,), dtype=np.uint8)

        written = int(
            self._ggml_quant_lib.ggml_quantize_chunk(
                ggml_type,
                src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                dst.ctypes.data_as(ctypes.c_void_p),
                0,
                nrows,
                n_per_row,
                None,
            )
        )
        if written != dst.nbytes:
            raise RuntimeError(
                f"ggml quantization wrote {written} bytes, expected {dst.nbytes} for '{tensor_name}'"
            )

        return dst.reshape((nrows, row_size))

    def _map_tensor_name(self, hf_name: str) -> str | None:
        """Map HuggingFace tensor name to GGML convention."""
        # Check direct mapping first
        if hf_name in self.TENSOR_MAP:
            return self.TENSOR_MAP[hf_name]

        # Check Talker layer patterns
        for pattern, template in self.TALKER_LAYER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                layer_idx = match.group(1)
                return template.format(layer_idx)

        # Check Code Predictor layer patterns
        for pattern, template in self.CODE_PREDICTOR_LAYER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                layer_idx = match.group(1)
                return template.format(layer_idx)

        # Check Code Predictor codebook patterns
        for pattern, template in self.CODE_PREDICTOR_CODEBOOK_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                codebook_idx = match.group(1)
                return template.format(codebook_idx)

        # Check Speaker Encoder patterns
        for pattern, template in self.SPEAKER_ENCODER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    return template.format(groups[0], groups[1])
                else:
                    return template.format(groups[0])

        return None

    def _get_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Iterate over all tensors from safetensors files."""
        safetensor_files = list(self.input_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {self.input_dir}")

        for sf_path in sorted(safetensor_files):
            logger.info(f"Loading tensors from {sf_path.name}")
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)

    def _module_from_tensor_name(self, tensor_name: str) -> str:
        """Get module name for a GGML tensor name."""
        if tensor_name.startswith("talker."):
            return "talker"
        if tensor_name.startswith("code_pred."):
            return "code_pred"
        if tensor_name.startswith("spk_enc."):
            return "spk_enc"
        return "talker"

    def _target_type_for_tensor(self, tensor_name: str) -> str:
        """Get configured target type for a GGML tensor name."""
        module_name = self._module_from_tensor_name(tensor_name)
        return self.quant_policy[module_name]

    def _should_quantize(self, tensor_name: str, n_dims: int) -> bool:
        """Determine if a tensor should be quantized or kept in floating point."""
        # Only quantize 2D weight matrices.
        if n_dims != 2 or ".weight" not in tensor_name:
            return False

        # Keep embeddings in F16
        if any(x in tensor_name for x in ["_embd", "codebook"]):
            return False

        # Keep layer norms in F16
        if "_norm" in tensor_name:
            return False

        # Keep biases in F16
        if ".bias" in tensor_name:
            return False

        # Keep LM heads in F16
        if "lm_head" in tensor_name or "codec_head" in tensor_name:
            return False

        # Quantize weight matrices
        return True

    def _convert_dtype(self, tensor: torch.Tensor, tensor_name: str = "") -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
        """Convert tensor to appropriate dtype for GGUF."""
        # Convert to numpy
        if tensor.dtype == torch.bfloat16:
            data = tensor.float().numpy()
        else:
            data = tensor.numpy()

        n_dims = len(data.shape)

        if n_dims == 3 and "weight" in tensor_name:
            logger.info(f"Conv1d weight {tensor_name}: shape {data.shape} [OC,IC,K] - GGUF will reverse to [K,IC,OC]")

        # 1D tensors (norms, biases) should be F32
        if n_dims <= 1:
            return data.astype(np.float32), gguf.GGMLQuantizationType.F32

        target_type = self._target_type_for_tensor(tensor_name)

        # For 2D+ tensors, use module-specific output type.
        if target_type == "f32":
            return data.astype(np.float32), gguf.GGMLQuantizationType.F32
        if target_type == "f16":
            return data.astype(np.float16), gguf.GGMLQuantizationType.F16

        if not self._should_quantize(tensor_name, n_dims):
            logger.debug(f"Keeping {tensor_name} in F16 (not quantizing)")
            return data.astype(np.float16), gguf.GGMLQuantizationType.F16

        data = data.astype(np.float32)
        quant_type = self.QUANT_TYPE_MAP[target_type]
        if target_type == "q5_k":
            quantized = self._quantize_with_ggml(data, target_type, tensor_name)
            return quantized, quant_type

        try:
            quantized = gguf.quants.quantize(data, quant_type)
            return quantized, quant_type
        except Exception as e:
            raise RuntimeError(
                f"Quantization failed for tensor '{tensor_name}' with target type '{target_type}': {e}"
            ) from e

    def _load_tokenizer(self) -> tuple[list[str], list[int], list[str]]:
        """Load tokenizer vocabulary and merges."""
        vocab_path = self.input_dir / "vocab.json"
        merges_path = self.input_dir / "merges.txt"

        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)

        # Sort by token ID
        sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])

        tokens = []
        toktypes = []

        for token, token_id in sorted_vocab:
            tokens.append(token)
            # Determine token type
            if token.startswith("<|") and token.endswith("|>"):
                toktypes.append(gguf.TokenType.CONTROL)
            else:
                toktypes.append(gguf.TokenType.NORMAL)

        # Pad to text_vocab_size if needed
        while len(tokens) < self.text_vocab_size:
            tokens.append(f"[PAD{len(tokens)}]")
            toktypes.append(gguf.TokenType.UNUSED)

        # Load merges
        merges = []
        if merges_path.exists():
            with open(merges_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        merges.append(line)

        return tokens, toktypes, merges

    def convert(self) -> None:
        """Convert the model to GGUF format."""
        logger.info(f"Converting {self.model_name} to GGUF format")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_path}")
        logger.info(
            "Quant policy: talker=%s, predictor=%s, speaker=%s",
            self.quant_policy["talker"],
            self.quant_policy["code_pred"],
            self.quant_policy["spk_enc"],
        )

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize GGUF writer
        arch = "qwen3-tts"
        writer = gguf.GGUFWriter(path=None, arch=arch)

        # Add metadata
        self._add_metadata(writer)

        # Add tokenizer
        self._add_tokenizer(writer)

        # Process tensors
        tensor_count = 0
        skipped_count = 0

        logger.info("Processing tensors...")
        for hf_name, tensor in tqdm(list(self._get_tensors()), desc="Converting"):
            ggml_name = self._map_tensor_name(hf_name)

            if ggml_name is None:
                logger.warning(f"Skipping unmapped tensor: {hf_name}")
                skipped_count += 1
                continue

            # Convert tensor
            data, dtype = self._convert_dtype(tensor, ggml_name)

            # Add tensor to writer
            writer.add_tensor(ggml_name, data, raw_dtype=dtype)
            tensor_count += 1

            logger.debug(f"  {hf_name} -> {ggml_name} [{dtype.name}] {data.shape}")

        logger.info(f"Converted {tensor_count} tensors, skipped {skipped_count}")

        # Write to file
        logger.info(f"Writing GGUF file to {self.output_path}")
        writer.write_header_to_file(path=self.output_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()

        logger.info("Conversion complete!")

    def _add_metadata(self, writer: gguf.GGUFWriter) -> None:
        """Add model metadata to GGUF writer."""
        arch = "qwen3-tts"
        
        # General metadata
        writer.add_name(self.model_name)
        writer.add_type(gguf.GGUFType.MODEL)

        # File type metadata follows the talker quantization type,
        # because talker is the dominant parameter block in this model.
        ftype = self.FILE_TYPE_MAP[self.quant_policy["talker"]]
        writer.add_file_type(ftype)

        # Quantization version
        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

        # Talker (main architecture) parameters
        writer.add_block_count(self.num_hidden_layers)
        writer.add_embedding_length(self.hidden_size)
        writer.add_feed_forward_length(self.intermediate_size)
        writer.add_head_count(self.num_attention_heads)
        writer.add_head_count_kv(self.num_kv_heads)
        writer.add_key_length(self.head_dim)
        writer.add_value_length(self.head_dim)
        writer.add_rope_freq_base(self.rope_theta)
        writer.add_layer_norm_rms_eps(self.rms_norm_eps)
        writer.add_vocab_size(self.vocab_size)

        # TTS-specific parameters
        writer.add_uint32(f"{arch}.text_vocab_size", self.text_vocab_size)
        writer.add_uint32(f"{arch}.text_hidden_size", self.text_hidden_size)
        writer.add_uint32(f"{arch}.num_code_groups", self.num_code_groups)

        # M-RoPE configuration
        writer.add_array(f"{arch}.rope.mrope_section", self.mrope_section)

        # Code Predictor parameters
        writer.add_uint32(f"{arch}.code_predictor.layer_count", self.code_predictor_num_layers)
        writer.add_uint32(f"{arch}.code_predictor.vocab_size", self.code_predictor_vocab_size)

        # Speaker Encoder parameters
        writer.add_uint32(f"{arch}.speaker_encoder.embedding_length", self.speaker_enc_dim)
        writer.add_uint32(f"{arch}.speaker_encoder.sample_rate", self.speaker_sample_rate)

        # Special codec token IDs
        writer.add_uint32(f"{arch}.codec.pad_id", self.codec_pad_id)
        writer.add_uint32(f"{arch}.codec.bos_id", self.codec_bos_id)
        writer.add_uint32(f"{arch}.codec.eos_id", self.codec_eos_id)
        writer.add_string(f"{arch}.quant.talker", self.quant_policy["talker"])
        writer.add_string(f"{arch}.quant.code_predictor", self.quant_policy["code_pred"])
        writer.add_string(f"{arch}.quant.speaker_encoder", self.quant_policy["spk_enc"])

        logger.info("Added model metadata")

    def _add_tokenizer(self, writer: gguf.GGUFWriter) -> None:
        """Add tokenizer to GGUF writer."""
        tokens, toktypes, merges = self._load_tokenizer()

        # Tokenizer model type
        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("qwen2")

        # Token list
        writer.add_token_list(tokens)
        writer.add_token_types(toktypes)

        # Merges
        if merges:
            writer.add_token_merges(merges)

        # Special tokens from tokenizer_config.json
        tokenizer_config_path = self.input_dir / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                tokenizer_config = json.load(f)

            # EOS token
            eos_token = tokenizer_config.get("eos_token")
            if isinstance(eos_token, dict):
                eos_token = eos_token.get("content")
            if eos_token:
                vocab_path = self.input_dir / "vocab.json"
                with open(vocab_path, "r", encoding="utf-8") as f:
                    vocab = json.load(f)
                if eos_token in vocab:
                    writer.add_eos_token_id(vocab[eos_token])

            # PAD token
            pad_token = tokenizer_config.get("pad_token")
            if isinstance(pad_token, dict):
                pad_token = pad_token.get("content")
            if pad_token:
                vocab_path = self.input_dir / "vocab.json"
                with open(vocab_path, "r", encoding="utf-8") as f:
                    vocab = json.load(f)
                if pad_token in vocab:
                    writer.add_pad_token_id(vocab[pad_token])

            # Chat template
            chat_template = tokenizer_config.get("chat_template")
            if chat_template:
                writer.add_chat_template(chat_template)

        logger.info(f"Added tokenizer with {len(tokens)} tokens and {len(merges)} merges")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-TTS-12Hz-0.6B-Base model to GGUF format"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to HuggingFace model directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output GGUF file path"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["f16", "f32", "q8_0", "q4_k", "q5_k"],
        default="f16",
        help="Global output data type. Can be overridden per module with --talker-type/--predictor-type/--speaker-type."
    )
    parser.add_argument(
        "--talker-type",
        choices=["f16", "f32", "q8_0", "q4_k", "q5_k"],
        default=None,
        help="Talker quantization type (module override)"
    )
    parser.add_argument(
        "--predictor-type",
        choices=["f16", "f32", "q8_0", "q4_k", "q5_k"],
        default=None,
        help="Code predictor quantization type (module override)"
    )
    parser.add_argument(
        "--speaker-type",
        choices=["f16", "f32", "q8_0", "q4_k", "q5_k"],
        default=None,
        help="Speaker encoder quantization type (module override)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    converter = Qwen3TTSConverter(
        input_dir=args.input,
        output_path=args.output,
        output_type=args.type,
        talker_type=args.talker_type,
        predictor_type=args.predictor_type,
        speaker_type=args.speaker_type,
    )
    converter.convert()


if __name__ == "__main__":
    main()
