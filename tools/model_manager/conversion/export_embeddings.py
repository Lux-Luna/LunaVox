#!/usr/bin/env python3
"""
Export Qwen3-TTS embedding assets:

- embeddings/text_embedding_projected.npy
- embeddings/codec_embedding_0.npy ... codec_embedding_15.npy
- optional embeddings/proj_weight.npy / proj_bias.npy (for 1.7B-style projection path)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def _load_all_tensors(model_dir: Path) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for sf in sorted(model_dir.glob("*.safetensors")):
        with safe_open(sf, framework="pt", device="cpu") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
    if not out:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")
    return out


def export_embeddings(input_dir: Path, output_dir: Path) -> None:
    tensors = _load_all_tensors(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_key = "talker.model.text_embedding.weight"
    fc1_w_key = "talker.text_projection.linear_fc1.weight"
    fc1_b_key = "talker.text_projection.linear_fc1.bias"
    fc2_w_key = "talker.text_projection.linear_fc2.weight"
    fc2_b_key = "talker.text_projection.linear_fc2.bias"

    missing = [k for k in [text_key, fc1_w_key, fc1_b_key, fc2_w_key, fc2_b_key] if k not in tensors]
    if missing:
        raise KeyError(f"Missing required talker projection tensors: {missing}")

    text_table = tensors[text_key].float()  # [vocab, text_hidden]
    fc1_w = tensors[fc1_w_key].float()      # [text_hidden, text_hidden] (PyTorch linear weight)
    fc1_b = tensors[fc1_b_key].float()      # [text_hidden]
    fc2_w = tensors[fc2_w_key].float()      # [hidden, text_hidden]
    fc2_b = tensors[fc2_b_key].float()      # [hidden]

    with torch.no_grad():
        x = torch.nn.functional.linear(text_table, fc1_w, fc1_b)
        x = torch.nn.functional.silu(x)
        x = torch.nn.functional.linear(x, fc2_w, fc2_b)
    np.save(output_dir / "text_embedding_projected.npy", x.cpu().numpy().astype(np.float32))

    codec0_key = "talker.model.codec_embedding.weight"
    if codec0_key not in tensors:
        raise KeyError(f"Missing required tensor: {codec0_key}")
    np.save(output_dir / "codec_embedding_0.npy", tensors[codec0_key].float().cpu().numpy().astype(np.float32))

    # Predictor codebook embeddings: 15 groups -> codec_embedding_1..15
    for i in range(15):
        k = f"talker.code_predictor.model.codec_embedding.{i}.weight"
        if k not in tensors:
            raise KeyError(f"Missing predictor embedding tensor: {k}")
        np.save(output_dir / f"codec_embedding_{i + 1}.npy", tensors[k].float().cpu().numpy().astype(np.float32))

    # Optional projection export for compatibility with 1.7B style pipelines.
    proj_w_key = "talker.code_predictor.input_proj.weight"
    proj_b_key = "talker.code_predictor.input_proj.bias"
    if proj_w_key in tensors and proj_b_key in tensors:
        np.save(output_dir / "proj_weight.npy", tensors[proj_w_key].float().cpu().numpy().astype(np.float32))
        np.save(output_dir / "proj_bias.npy", tensors[proj_b_key].float().cpu().numpy().astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS embedding assets")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to HF base model directory")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output embeddings directory")
    args = parser.parse_args()
    export_embeddings(args.input, args.output)


if __name__ == "__main__":
    main()

