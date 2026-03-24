#!/usr/bin/env python3
"""
Extract talker/predictor HF checkpoints and convert to llama.cpp-compatible GGUF.

Outputs:
  - qwen3_tts_talker.q5_k.gguf
  - qwen3_tts_predictor.q8_0.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(
    os.environ.get("LUNAVOX_PROJECT_ROOT", str(Path(__file__).resolve().parents[4]))
).resolve()
HF_EXPORT_DIR = Path(__file__).resolve().parent / "hf_export"
LLAMA_QUANT = REPO_ROOT / "lib" / "llama" / (
    "llama-quantize.exe" if os.name == "nt" else "llama-quantize"
)


def _ensure_hf_export_present() -> None:
    if not (HF_EXPORT_DIR / "convert_hf_to_gguf.py").exists():
        raise FileNotFoundError(
            f"Missing converter dependency: {HF_EXPORT_DIR / 'convert_hf_to_gguf.py'}"
        )


def _write_minimal_tokenizer(dst_hf_dir: Path) -> None:
    vocab = {"a": 0, "b": 1, "ab": 2}
    for i in range(3, 10):
        vocab[f"<t_{i}>"] = i

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "vocab": vocab, "merges": ["a b"]},
    }
    with open(dst_hf_dir / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False)
    with open(dst_hf_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump({"tokenizer_class": "Qwen2Tokenizer"}, f, ensure_ascii=False)
    with open(dst_hf_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(dst_hf_dir / "merges.txt", "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        f.write("a b\n")


def _extract_talker_hf(base_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = load_file(str(base_dir / "model.safetensors"))
    with open(base_dir / "config.json", "r", encoding="utf-8") as f:
        full_cfg = json.load(f)

    talker_cfg = dict(full_cfg.get("talker_config", {}))
    out_weights: dict[str, torch.Tensor] = {}

    for key, tensor in weights.items():
        if key.startswith("talker.model."):
            sub = key[len("talker.model.") :]
            if sub == "codec_embedding.weight":
                out_weights["model.embed_tokens.weight"] = tensor
                continue
            if sub == "text_embedding.weight":
                continue
            out_weights[f"model.{sub}"] = tensor
        elif key == "talker.codec_head.weight":
            out_weights["lm_head.weight"] = tensor

    if "model.embed_tokens.weight" not in out_weights or "lm_head.weight" not in out_weights:
        raise RuntimeError("Failed to extract required talker tensors (embed/lm_head)")

    save_file(out_weights, str(out_dir / "model.safetensors"))

    out_cfg = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": int(talker_cfg.get("vocab_size", 3072)),
        "hidden_size": int(talker_cfg.get("hidden_size", 1024)),
        "head_dim": int(talker_cfg.get("head_dim", 128)),
        "intermediate_size": int(talker_cfg.get("intermediate_size", 3072)),
        "num_hidden_layers": int(talker_cfg.get("num_hidden_layers", 28)),
        "num_attention_heads": int(talker_cfg.get("num_attention_heads", 16)),
        "num_key_value_heads": int(talker_cfg.get("num_key_value_heads", 8)),
        "rms_norm_eps": float(talker_cfg.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(talker_cfg.get("rope_theta", 1000000.0)),
        "hidden_act": "silu",
        "max_position_embeddings": int(talker_cfg.get("max_position_embeddings", 32768)),
        "tie_word_embeddings": False,
        "use_cache": True,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, indent=2, ensure_ascii=False)

    _write_minimal_tokenizer(out_dir)


def _extract_predictor_hf(base_dir: Path, out_dir: Path, embeddings_dir: Path) -> None:
    dtype = "float16"
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = load_file(str(base_dir / "model.safetensors"))
    with open(base_dir / "config.json", "r", encoding="utf-8") as f:
        full_cfg = json.load(f)
    pred_cfg = dict(full_cfg.get("talker_config", {}).get("code_predictor_config", {}))

    proj_w_key = "talker.code_predictor.small_to_mtp_projection.weight"
    proj_b_key = "talker.code_predictor.small_to_mtp_projection.bias"
    has_proj = proj_w_key in weights and proj_b_key in weights
    proj_w = weights[proj_w_key] if has_proj else None
    proj_b = weights[proj_b_key] if has_proj else None

    emb_chunks: list[torch.Tensor] = []
    idx = 0
    while True:
        key = f"talker.code_predictor.model.codec_embedding.{idx}.weight"
        if key not in weights:
            break
        raw = weights[key]
        if has_proj:
            raw = torch.nn.functional.linear(raw, proj_w, proj_b)
        emb_chunks.append(raw)
        idx += 1
    if not emb_chunks:
        raise RuntimeError("No predictor codec embeddings found")
    cat_emb = torch.cat(emb_chunks, dim=0).contiguous()

    head_chunks: list[torch.Tensor] = []
    j = 0
    while True:
        key = f"talker.code_predictor.lm_head.{j}.weight"
        if key not in weights:
            break
        head_chunks.append(weights[key])
        j += 1
    if not head_chunks:
        raise RuntimeError("No predictor lm_head blocks found")
    cat_head = torch.cat(head_chunks, dim=0).contiguous()

    out_weights: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": cat_emb,
        "lm_head.weight": cat_head,
        "model.norm.weight": weights["talker.code_predictor.model.norm.weight"],
    }

    layer_count = 0
    while True:
        prefix = f"talker.code_predictor.model.layers.{layer_count}."
        found = False
        for key, tensor in weights.items():
            if key.startswith(prefix):
                found = True
                out_weights[f"model.layers.{layer_count}." + key[len(prefix):]] = tensor
        if not found:
            break
        layer_count += 1
    if layer_count == 0:
        raise RuntimeError("No predictor transformer layers found")

    save_file(out_weights, str(out_dir / "model.safetensors"))

    hidden = int(cat_emb.shape[1])
    heads = int(pred_cfg.get("num_attention_heads", 16))
    out_cfg = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": int(cat_emb.shape[0]),
        "hidden_size": hidden,
        "head_dim": int(pred_cfg.get("head_dim", 128)),
        "intermediate_size": int(pred_cfg.get("intermediate_size", hidden * 3)),
        "num_hidden_layers": int(layer_count),
        "num_attention_heads": heads,
        "num_key_value_heads": int(pred_cfg.get("num_key_value_heads", max(1, heads // 2))),
        "rms_norm_eps": float(pred_cfg.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(pred_cfg.get("rope_theta", 1000000.0)),
        "hidden_act": "silu",
        "max_position_embeddings": int(pred_cfg.get("max_position_embeddings", 32768)),
        "tie_word_embeddings": False,
        "use_cache": True,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, indent=2, ensure_ascii=False)

    _write_minimal_tokenizer(out_dir)

    if has_proj:
        import numpy as np

        embeddings_dir.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_dir / "proj_weight.npy", proj_w.float().cpu().numpy().astype(dtype))
        np.save(embeddings_dir / "proj_bias.npy", proj_b.float().cpu().numpy().astype(dtype))


def _convert_hf_to_gguf(hf_dir: Path, out_gguf: Path) -> None:
    _ensure_hf_export_present()
    sys.path.insert(0, str(HF_EXPORT_DIR))
    try:
        import convert_hf_to_gguf  # type: ignore
        from convert_hf_to_gguf import ModelBase, TextModel  # type: ignore
    finally:
        sys.path.pop(0)

    def patched_get_vocab_base_pre(self, tokenizer) -> str:  # type: ignore[no-untyped-def]
        return "qwen2"

    def patched_load_hparams(dir_model: Path, is_mistral_format: bool):  # type: ignore[no-untyped-def]
        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "llm_config" in cfg:
            cfg["text_config"] = cfg["llm_config"]
        if "lm_config" in cfg:
            cfg["text_config"] = cfg["lm_config"]
        return cfg

    TextModel.get_vocab_base_pre = patched_get_vocab_base_pre
    ModelBase.load_hparams = staticmethod(patched_load_hparams)

    argv_old = list(sys.argv)
    sys.argv = [
        "convert_hf_to_gguf.py",
        str(hf_dir),
        "--outfile",
        str(out_gguf),
        "--outtype",
        "f16",
    ]
    try:
        convert_hf_to_gguf.main()
    except SystemExit as se:
        if int(getattr(se, "code", 0) or 0) != 0:
            raise
    finally:
        sys.argv = argv_old


def _quantize(src_f16: Path, dst_quant: Path, quant_type: str) -> None:
    if dst_quant.exists():
        dst_quant.unlink()
    if not LLAMA_QUANT.exists():
        raise FileNotFoundError(f"Missing quantize binary: {LLAMA_QUANT}")
    cmd = [str(LLAMA_QUANT), str(src_f16), str(dst_quant), quant_type]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert talker/predictor to llama.cpp GGUF (Q5_K + Q8_0)")
    ap.add_argument("--input", required=True, help="HF base model directory (contains model.safetensors/config.json)")
    ap.add_argument("--out-talker", required=True, help="Output talker q5_k GGUF path")
    ap.add_argument("--out-predictor", required=True, help="Output predictor q8_0 GGUF path")
    ap.add_argument(
        "--embeddings-dir",
        default="",
        help="Embeddings directory (optional, used to write proj_weight/proj_bias when projection exists)",
    )
    ap.add_argument("--work-dir", default="", help="Temporary working directory")
    ap.add_argument("--keep-work-dir", action="store_true", help="Keep temporary extraction artifacts")
    args = ap.parse_args()

    input_dir = Path(args.input).resolve()
    out_talker = Path(args.out_talker).resolve()
    out_predictor = Path(args.out_predictor).resolve()
    embeddings_dir = (
        Path(args.embeddings_dir).resolve()
        if args.embeddings_dir
        else out_talker.parent / "embeddings"
    )
    work_dir = (
        Path(args.work_dir).resolve()
        if args.work_dir
        else out_talker.parent / "_tmp_hf_extract"
    )

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    talker_hf = work_dir / "talker_hf"
    predictor_hf = work_dir / "predictor_hf"
    talker_f16 = work_dir / "qwen3_tts_talker.f16.gguf"
    predictor_f16 = work_dir / "qwen3_tts_predictor.f16.gguf"

    out_talker.parent.mkdir(parents=True, exist_ok=True)
    out_predictor.parent.mkdir(parents=True, exist_ok=True)

    _extract_talker_hf(input_dir, talker_hf)
    _extract_predictor_hf(input_dir, predictor_hf, embeddings_dir)

    _convert_hf_to_gguf(talker_hf, talker_f16)
    _convert_hf_to_gguf(predictor_hf, predictor_f16)

    _quantize(talker_f16, out_talker, "q5_k")
    _quantize(predictor_f16, out_predictor, "q8_0")

    if not args.keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"[ok] talker: {out_talker}")
    print(f"[ok] predictor: {out_predictor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
