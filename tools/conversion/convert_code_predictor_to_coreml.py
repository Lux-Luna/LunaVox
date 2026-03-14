#!/usr/bin/env python3
"""
Export Qwen3-TTS code predictor to CoreML for macOS ANE inference.

This exporter builds a single CoreML model that runs the 5-layer code predictor
on a fixed 16-token context window. Runtime provides:

- `seq_embd`: [1, 16, 1024]   input embeddings (padded with zeros)
- `attn_mask`: [1, 1, 16, 16]  additive causal+padding mask (0 or large negative)
- `selector`: [1, 16, 1]       one-hot row selecting the "last" token hidden state

The model outputs:

- `logits_all`: [1, 15, 2048]  per-codebook logits from the selected hidden state

Usage:
  python scripts/convert_code_predictor_to_coreml.py \
      --input models/Qwen3-TTS-12Hz-0.6B-Base \
      --output models/coreml/code_predictor.mlpackage
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_f32(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.bfloat16:
        return t.float().contiguous()
    if t.dtype == torch.float16:
        return t.float().contiguous()
    return t.contiguous()


def load_safetensor_map(model_dir: Path) -> Dict[str, torch.Tensor]:
    st_path = model_dir / "model.safetensors"
    if not st_path.exists():
        raise FileNotFoundError(f"Missing {st_path}")

    tensors: Dict[str, torch.Tensor] = {}
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = to_f32(f.get_tensor(key))
    return tensors


class RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x_f = x.float()
        var = (x_f * x_f).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        return x_norm * self.weight.view(1, 1, -1)


class CodePredBlock(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        weights: Dict[str, torch.Tensor],
        hidden_size: int,
        n_head: int,
        n_kv_head: int,
        head_dim: int,
        eps: float,
        rope_theta: float,
        max_seq: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(float(head_dim))

        pfx = f"talker.code_predictor.model.layers.{layer_idx}"

        self.attn_norm = RMSNorm(weights[f"{pfx}.input_layernorm.weight"], eps)
        self.ffn_norm = RMSNorm(weights[f"{pfx}.post_attention_layernorm.weight"], eps)

        self.w_q = nn.Parameter(weights[f"{pfx}.self_attn.q_proj.weight"].clone(), requires_grad=False)
        self.w_k = nn.Parameter(weights[f"{pfx}.self_attn.k_proj.weight"].clone(), requires_grad=False)
        self.w_v = nn.Parameter(weights[f"{pfx}.self_attn.v_proj.weight"].clone(), requires_grad=False)
        self.w_o = nn.Parameter(weights[f"{pfx}.self_attn.o_proj.weight"].clone(), requires_grad=False)

        self.q_norm_w = nn.Parameter(weights[f"{pfx}.self_attn.q_norm.weight"].clone(), requires_grad=False)
        self.k_norm_w = nn.Parameter(weights[f"{pfx}.self_attn.k_norm.weight"].clone(), requires_grad=False)

        self.w_gate = nn.Parameter(weights[f"{pfx}.mlp.gate_proj.weight"].clone(), requires_grad=False)
        self.w_up = nn.Parameter(weights[f"{pfx}.mlp.up_proj.weight"].clone(), requires_grad=False)
        self.w_down = nn.Parameter(weights[f"{pfx}.mlp.down_proj.weight"].clone(), requires_grad=False)

        half = head_dim // 2
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32) / float(half)))
        pos = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", pos, inv_freq)
        self.register_buffer("rope_cos", torch.cos(freqs), persistent=False)
        self.register_buffer("rope_sin", torch.sin(freqs), persistent=False)

    def _rmsnorm_last(self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x_f = x.float()
        var = (x_f * x_f).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + eps)
        return x_norm * weight.view(1, 1, 1, -1)

    def _apply_rope_neox(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H, D] — all dims fixed at trace time
        half = self.head_dim // 2
        x1 = x[..., :half]
        x2 = x[..., half:]

        cos = self.rope_cos.unsqueeze(0).unsqueeze(2)  # [1, max_seq, 1, half]
        sin = self.rope_sin.unsqueeze(0).unsqueeze(2)

        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.cat([out1, out2], dim=-1)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], attn_mask: [B, 1, T, T]
        # All dimensions are fixed (B=1, T=max_seq) — avoid dynamic shape queries
        # which produce traced tensor scalars incompatible with coremltools int cast.
        residual = x
        x_norm = self.attn_norm(x)

        q = F.linear(x_norm, self.w_q)
        k = F.linear(x_norm, self.w_k)
        v = F.linear(x_norm, self.w_v)

        q = q.reshape(1, -1, self.n_head, self.head_dim)
        k = k.reshape(1, -1, self.n_kv_head, self.head_dim)
        v = v.reshape(1, -1, self.n_kv_head, self.head_dim)

        q = self._rmsnorm_last(q, self.q_norm_w)
        k = self._rmsnorm_last(k, self.k_norm_w)

        q = self._apply_rope_neox(q)
        k = self._apply_rope_neox(k)

        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + attn_mask
        probs = torch.softmax(scores, dim=-1)

        ctx = torch.matmul(probs, v)
        ctx = ctx.permute(0, 2, 1, 3).reshape(1, -1, self.n_head * self.head_dim)

        x = F.linear(ctx, self.w_o)
        x = x + residual

        residual = x
        x_norm = self.ffn_norm(x)
        gate = F.linear(x_norm, self.w_gate)
        up = F.linear(x_norm, self.w_up)
        x = F.silu(gate) * up
        x = F.linear(x, self.w_down)
        x = x + residual
        return x


class CodePredictorCoreMLModel(nn.Module):
    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        hidden_size: int,
        n_layers: int,
        n_head: int,
        n_kv_head: int,
        head_dim: int,
        eps: float,
        rope_theta: float,
        n_codebooks_minus1: int,
        vocab_size: int,
        max_seq: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_codebooks_minus1 = n_codebooks_minus1

        self.blocks = nn.ModuleList(
            [
                CodePredBlock(
                    layer_idx=i,
                    weights=weights,
                    hidden_size=hidden_size,
                    n_head=n_head,
                    n_kv_head=n_kv_head,
                    head_dim=head_dim,
                    eps=eps,
                    rope_theta=rope_theta,
                    max_seq=max_seq,
                )
                for i in range(n_layers)
            ]
        )

        self.output_norm = RMSNorm(weights["talker.code_predictor.model.norm.weight"], eps)
        self.heads = nn.ParameterList(
            [
                nn.Parameter(
                    weights[f"talker.code_predictor.lm_head.{i}.weight"].clone(),
                    requires_grad=False,
                )
                for i in range(n_codebooks_minus1)
            ]
        )

        # Validate expected dimensions early.
        for i, w in enumerate(self.heads):
            if w.shape != (vocab_size, hidden_size):
                raise ValueError(f"lm_head.{i} shape {tuple(w.shape)} != ({vocab_size}, {hidden_size})")

    def forward(self, seq_embd: torch.Tensor, attn_mask: torch.Tensor, selector: torch.Tensor) -> torch.Tensor:
        # seq_embd: [B, 16, D]
        # attn_mask: [B, 1, 16, 16]
        # selector: [B, 16, 1] one-hot on last valid token row
        x = seq_embd
        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = self.output_norm(x)
        last_hidden = (x * selector).sum(dim=1)

        logits = [F.linear(last_hidden, w) for w in self.heads]
        return torch.stack(logits, dim=1)


def summarize_compute_plan(model_path: Path) -> Dict[str, int]:
    from coremltools.models.compute_plan import MLComputePlan
    from coremltools.models.utils import compile_model

    compiled_path = compile_model(str(model_path))
    plan = MLComputePlan.load_from_path(compiled_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

    ops_total = 0
    ops_with_usage = 0
    ops_ne_supported = 0
    ops_no_ne: List[str] = []

    fn = plan.model_structure.program.functions["main"]
    for op in fn.block.operations:
        ops_total += 1
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            continue

        ops_with_usage += 1
        supported = [type(d).__name__ for d in usage.supported_compute_devices]
        has_ne = any("NeuralEngine" in s for s in supported)
        if has_ne:
            ops_ne_supported += 1
        else:
            ops_no_ne.append(op.operator_name)

    if ops_no_ne:
        uniq = sorted(set(ops_no_ne))
        raise RuntimeError(
            "CoreML compute plan reports ops without ANE support: " + ", ".join(uniq)
        )

    return {
        "ops_total": ops_total,
        "ops_with_usage": ops_with_usage,
        "ops_ne_supported": ops_ne_supported,
    }


def run_numeric_sanity(
    torch_model: nn.Module,
    coreml_model: ct.models.MLModel,
    hidden_size: int,
    max_seq: int,
    n_codebooks_minus1: int,
    vocab_size: int,
) -> None:
    rng = np.random.default_rng(123)

    seq_embd = rng.standard_normal((1, max_seq, hidden_size), dtype=np.float32) * 0.1
    attn_mask = np.full((1, 1, max_seq, max_seq), -1e9, dtype=np.float32)
    selector = np.zeros((1, max_seq, 1), dtype=np.float32)

    valid_len = 7
    for i in range(valid_len):
        for j in range(valid_len):
            if j <= i:
                attn_mask[0, 0, i, j] = 0.0
    selector[0, valid_len - 1, 0] = 1.0

    with torch.no_grad():
        t_out = torch_model(
            torch.from_numpy(seq_embd),
            torch.from_numpy(attn_mask),
            torch.from_numpy(selector),
        ).cpu().numpy()

    c_out = coreml_model.predict(
        {
            "seq_embd": seq_embd,
            "attn_mask": attn_mask,
            "selector": selector,
        }
    )["logits_all"]

    # CoreML may return float16 arrays depending on backend.
    c_out = np.asarray(c_out, dtype=np.float32)
    if c_out.shape != (1, n_codebooks_minus1, vocab_size):
        raise RuntimeError(
            f"Unexpected CoreML output shape {c_out.shape}, expected {(1, n_codebooks_minus1, vocab_size)}"
        )

    diff = np.max(np.abs(t_out - c_out))
    denom = np.linalg.norm(t_out.reshape(-1)) * np.linalg.norm(c_out.reshape(-1))
    cos = float(np.dot(t_out.reshape(-1), c_out.reshape(-1)) / (denom + 1e-12))

    if not np.isfinite(diff) or not np.isfinite(cos):
        raise RuntimeError(f"Non-finite sanity metrics: max_abs_diff={diff}, cosine={cos}")

    print(f"[sanity] max_abs_diff={diff:.6f} cosine={cos:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS code predictor to CoreML")
    parser.add_argument("--input", type=Path, required=True, help="HF model directory (contains model.safetensors + config.json)")
    parser.add_argument("--output", type=Path, required=True, help="Output .mlpackage path")
    parser.add_argument("--max-seq", type=int, default=16, help="Fixed sequence length for CoreML model")
    parser.add_argument("--skip-sanity", action="store_true", help="Skip torch vs CoreML sanity check")
    parser.add_argument("--skip-ane-check", action="store_true", help="Skip compute-plan ANE support validation")
    args = parser.parse_args()

    cfg = load_json(args.input / "config.json")
    talker = cfg["talker_config"]
    cp_cfg = talker["code_predictor_config"]

    hidden_size = int(cp_cfg.get("hidden_size", talker.get("hidden_size", 1024)))
    n_layers = int(cp_cfg.get("num_hidden_layers", 5))
    n_head = int(cp_cfg.get("num_attention_heads", 16))
    n_kv_head = int(cp_cfg.get("num_key_value_heads", 8))
    head_dim = int(cp_cfg.get("head_dim", 128))
    eps = float(cp_cfg.get("rms_norm_eps", talker.get("rms_norm_eps", 1e-6)))
    rope_theta = float(cp_cfg.get("rope_theta", talker.get("rope_theta", 1000000.0)))
    vocab_size = int(cp_cfg.get("vocab_size", 2048))
    n_codebooks_minus1 = int(talker.get("num_code_groups", 16)) - 1

    print("[config] hidden_size=", hidden_size)
    print("[config] n_layers=", n_layers)
    print("[config] n_head=", n_head, "n_kv_head=", n_kv_head, "head_dim=", head_dim)
    print("[config] vocab_size=", vocab_size, "codebooks_minus1=", n_codebooks_minus1)
    print("[config] max_seq=", args.max_seq)

    weights = load_safetensor_map(args.input)

    model = CodePredictorCoreMLModel(
        weights=weights,
        hidden_size=hidden_size,
        n_layers=n_layers,
        n_head=n_head,
        n_kv_head=n_kv_head,
        head_dim=head_dim,
        eps=eps,
        rope_theta=rope_theta,
        n_codebooks_minus1=n_codebooks_minus1,
        vocab_size=vocab_size,
        max_seq=args.max_seq,
    ).eval()

    with torch.no_grad():
        ex_seq = torch.zeros((1, args.max_seq, hidden_size), dtype=torch.float32)
        ex_mask = torch.zeros((1, 1, args.max_seq, args.max_seq), dtype=torch.float32)
        ex_sel = torch.zeros((1, args.max_seq, 1), dtype=torch.float32)
        ex_sel[0, 0, 0] = 1.0
        traced = torch.jit.trace(model, (ex_seq, ex_mask, ex_sel))

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        inputs=[
            ct.TensorType(name="seq_embd", shape=ex_seq.shape, dtype=np.float32),
            ct.TensorType(name="attn_mask", shape=ex_mask.shape, dtype=np.float32),
            ct.TensorType(name="selector", shape=ex_sel.shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logits_all", dtype=np.float32)],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.output))
    print(f"[save] {args.output}")

    if not args.skip_sanity:
        # Re-open from saved package to validate runtime behavior with model loading path.
        mlmodel_runtime = ct.models.MLModel(
            str(args.output),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        run_numeric_sanity(
            torch_model=model,
            coreml_model=mlmodel_runtime,
            hidden_size=hidden_size,
            max_seq=args.max_seq,
            n_codebooks_minus1=n_codebooks_minus1,
            vocab_size=vocab_size,
        )

    if not args.skip_ane_check:
        summary = summarize_compute_plan(args.output)
        print(
            "[ane-check] ops_total={ops_total} ops_with_usage={ops_with_usage} "
            "ops_ne_supported={ops_ne_supported}".format(**summary)
        )


if __name__ == "__main__":
    main()
