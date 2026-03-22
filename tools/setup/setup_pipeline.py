#!/usr/bin/env python3
"""
One-shot model setup for LunaVox.

Converts a pre-downloaded HuggingFace model into runtime artifacts:

- models/<variant>/qwen3_tts_talker.q5_k.gguf
- models/<variant>/qwen3_tts_predictor.q8_0.gguf
- models/<variant>/qwen3_tts_codec_encoder.fp16.onnx
- models/<variant>/qwen3_tts_speaker_encoder.fp16.onnx
- models/<variant>/qwen3_tts_decoder.fp16.onnx
- models/<variant>/embeddings/*
- models/<variant>/tokenizer.json

Model source paths are resolved from model_config.py (no HF download).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"

# Import model config
sys.path.insert(0, str(REPO_ROOT))
from model_config import Models, ModelConfig
sys.path.pop(0)


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def run_cmd(
    cmd: list[str],
    cwd: Path,
    env: Optional[dict[str, str]] = None,
    timeout_sec: Optional[int] = None,
    log_file: Optional[Path] = None,
) -> None:
    eprint(f"[run] {' '.join(cmd)}")
    start = time.time()
    latest_log = REPO_ROOT / "logs" / "latest.log"
    header = (
        f"\n{'='*80}\n"
        f"TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"CMD: {' '.join(cmd)}\n"
        f"CWD: {cwd}\n"
        f"{'='*80}\n"
    )
    
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec if timeout_sec and timeout_sec > 0 else None,
        )
        output_text = proc.stdout or ""
        latest_log.parent.mkdir(parents=True, exist_ok=True)
        elapsed = time.time() - start
        log_entry = (
            f"{header}"
            f"STATUS: ok\n"
            f"ELAPSED: {elapsed:.3f}s\n\n"
            f"{output_text}\n"
        )
        with open(latest_log, "a", encoding="utf-8") as f:
            f.write(log_entry)
                
    except subprocess.TimeoutExpired as err:
        output_text = ""
        if err.stdout:
            output_text += err.stdout if isinstance(err.stdout, str) else err.stdout.decode("utf-8", errors="ignore")
        if err.stderr:
            output_text += err.stderr if isinstance(err.stderr, str) else err.stderr.decode("utf-8", errors="ignore")
        
        elapsed = time.time() - start
        log_entry = (
            f"{header}"
            f"STATUS: timeout\n"
            f"ELAPSED: {elapsed:.3f}s\n\n"
            f"{output_text}\n"
        )
        latest_log.parent.mkdir(parents=True, exist_ok=True)
        with open(latest_log, "a", encoding="utf-8") as f:
            f.write(log_entry)
            
        msg = f"Command timed out after {timeout_sec}s: {' '.join(cmd)}\nSee log: {latest_log}"
        raise RuntimeError(msg) from err
    except subprocess.CalledProcessError as err:
        output_text = err.stdout or ""
        elapsed = time.time() - start
        log_entry = (
            f"{header}"
            f"STATUS: failed\n"
            f"RETURNCODE: {err.returncode}\n"
            f"ELAPSED: {elapsed:.3f}s\n\n"
            f"{output_text}\n"
        )
        latest_log.parent.mkdir(parents=True, exist_ok=True)
        with open(latest_log, "a", encoding="utf-8") as f:
            f.write(log_entry)
        raise


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def require_modules(modules: Iterable[tuple[str, str]]) -> None:
    missing = [f"{name} ({pip_name})" for name, pip_name in modules if not has_module(name)]
    if missing:
        raise RuntimeError(
            "Missing required Python modules: "
            + ", ".join(missing)
            + "\nInstall them, for example:\n"
            + f"  {sys.executable} -m pip install "
            + " ".join(pip_name for _, pip_name in modules)
        )


def ensure_source_exists(cfg: ModelConfig) -> None:
    """Verify the HF source model exists on disk."""
    required = [
        cfg.source / "config.json",
        cfg.source / "model.safetensors",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Model source files not found for '{cfg.name}':\n"
            + "\n".join(f"  - {p}" for p in missing)
            + f"\n\nExpected location: {cfg.source}"
            + "\nPlease ensure the model is downloaded to the HuggingFace cache."
        )
    eprint(f"[ok] source model '{cfg.name}' found: {cfg.source}")


def ensure_talker_predictor(
    python_exe: str,
    base_dir: Path,
    out_talker: Path,
    out_predictor: Path,
    out_embeddings_dir: Path,
) -> None:
    if out_talker.exists() and out_predictor.exists():
        eprint(f"[ok] exists: {out_talker}")
        eprint(f"[ok] exists: {out_predictor}")
        return

    run_cmd(
        [
            python_exe,
            str(TOOLS_DIR / "conversion" / "convert_talker_predictor_llama.py"),
            "--input",
            str(base_dir),
            "--out-talker",
            str(out_talker),
            "--out-predictor",
            str(out_predictor),
            "--embeddings-dir",
            str(out_embeddings_dir),
        ],
        cwd=REPO_ROOT,
    )


def ensure_embeddings(
    python_exe: str,
    base_dir: Path,
    out_embeddings_dir: Path,
) -> None:
    text_emb = out_embeddings_dir / "text_embedding_projected.npy"
    codec_emb0 = out_embeddings_dir / "codec_embedding_0.npy"
    if text_emb.exists() and codec_emb0.exists():
        eprint(f"[ok] exists: {out_embeddings_dir}")
        return
    run_cmd(
        [
            python_exe,
            str(TOOLS_DIR / "conversion" / "export_embeddings.py"),
            "--input",
            str(base_dir),
            "--output",
            str(out_embeddings_dir),
        ],
        cwd=REPO_ROOT,
    )


def ensure_tokenizer_json(base_dir: Path, out_tokenizer_json: Path) -> None:
    src = base_dir / "tokenizer.json"
    out_tokenizer_json.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, out_tokenizer_json)
        return
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True, fix_mistral_regex=True)
    if not hasattr(tok, "backend_tokenizer"):
        raise RuntimeError("Tokenizer backend is unavailable; failed to generate tokenizer.json")
    tok.backend_tokenizer.save(str(out_tokenizer_json))


def run_onnx_stage(
    python_exe: str,
    stage: str,
    base_dir: Path,
    models_dir: Path,
    timeout_sec: int,
    logs_dir: Path,
    enable_quant: bool,
) -> None:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [
        python_exe,
        str(TOOLS_DIR / "conversion" / "export_onnx_models.py"),
        "--base-dir",
        str(base_dir),
        "--output-dir",
        str(models_dir),
        "--stage",
        stage,
    ]
    if enable_quant:
        cmd.append("--enable-quant")
    run_cmd(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        timeout_sec=timeout_sec,
    )


def ensure_onnx_artifacts(
    python_exe: str,
    base_dir: Path,
    models_dir: Path,
    out_codec_encoder: Path,
    out_speaker_encoder: Path,
    out_decoder: Path,
    timeout_sec: int,
    logs_dir: Path,
    enable_quant: bool,
) -> None:
    stage_to_output = {
        "codec_encoder": out_codec_encoder,
        "speaker_encoder": out_speaker_encoder,
        "decoder": out_decoder,
    }

    logs_dir.mkdir(parents=True, exist_ok=True)
    for stage, artifact in stage_to_output.items():
        if artifact.exists():
            eprint(f"[skip] ONNX stage '{stage}' already done: {artifact}")
            continue
        run_onnx_stage(
            python_exe=python_exe,
            stage=stage,
            base_dir=base_dir,
            models_dir=models_dir,
            timeout_sec=timeout_sec,
            logs_dir=logs_dir,
            enable_quant=enable_quant,
        )

    if enable_quant:
        run_onnx_stage(
            python_exe=python_exe,
            stage="quantize",
            base_dir=base_dir,
            models_dir=models_dir,
            timeout_sec=timeout_sec,
            logs_dir=logs_dir,
            enable_quant=enable_quant,
        )

    missing = [p for p in [out_codec_encoder, out_speaker_encoder, out_decoder] if not p.exists()]
    if missing:
        raise RuntimeError("ONNX export did not produce required files:\n" + "\n".join(str(p) for p in missing))

    run_cmd(
        [
            python_exe,
            str(TOOLS_DIR / "conversion" / "validate_onnx_models.py"),
            "--models-dir",
            str(models_dir),
        ],
        cwd=REPO_ROOT,
        timeout_sec=timeout_sec,
    )

    # Auto-cleanup: remove intermediate fp32 ONNX files after successful validation
    fp32_patterns = [
        "qwen3_tts_codec_encoder.fp32.onnx",
        "qwen3_tts_codec_encoder.fp32.onnx.data",
        "qwen3_tts_speaker_encoder.fp32.onnx",
        "qwen3_tts_speaker_encoder.fp32.onnx.data",
        "qwen3_tts_decoder.fp32.onnx",
        "qwen3_tts_decoder.fp32.onnx.data",
    ]
    cleaned = []
    for name in fp32_patterns:
        fp32_file = models_dir / name
        if fp32_file.exists():
            fp32_file.unlink()
            cleaned.append(name)
    if cleaned:
        eprint(f"[cleanup] Removed {len(cleaned)} intermediate fp32 file(s): {', '.join(cleaned)}")


def parse_args() -> argparse.Namespace:
    valid_models = ", ".join(m.name for m in Models.ALL)
    p = argparse.ArgumentParser(description="Convert pre-downloaded models into LunaVox runtime artifacts")
    p.add_argument(
        "--model",
        default="base_small",
        help=f"Model variant to convert. Available: {valid_models} (default: base_small)",
    )
    p.add_argument("--models-dir", default="", help="Override target models directory (default: from model_config)")
    p.add_argument("--skip-convert", action="store_true", help="Skip artifact conversion/export")
    p.add_argument("--force", action="store_true", help="Re-generate outputs")
    p.add_argument("--timeout-sec", type=int, default=170, help="Per-stage timeout in seconds for ONNX export")
    p.add_argument(
        "--skip-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip optional local ONNX int8 quantization stage (default: true)",
    )
    p.add_argument("--enable-quant", action="store_true", help="Alias of --no-skip-quant")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    enable_quant = bool(args.enable_quant or (not args.skip_quant))

    # Resolve model config
    cfg = Models.by_name(args.model)
    eprint(f"[config] Selected model: {cfg.name}")
    eprint(f"[config] Source: {cfg.source}")

    require_modules(
        [
            ("torch", "torch"),
            ("numpy", "numpy"),
            ("tqdm", "tqdm"),
            ("safetensors", "safetensors"),
            ("gguf", "gguf"),
            ("transformers", "transformers"),
        ]
    )
    if not args.skip_convert:
        require_modules(
            [
                ("onnx", "onnx"),
                ("onnxruntime", "onnxruntime"),
            ]
        )
        if enable_quant:
            require_modules([("onnxruntime.quantization", "onnxruntime-tools")])

    # Resolve output directory
    models_dir = Path(args.models_dir).resolve() if args.models_dir else cfg.dest.resolve()
    base_dir = cfg.source.resolve()
    eprint(f"[config] Output: {models_dir}")

    # Verify source exists
    ensure_source_exists(cfg)

    # Determine ONNX source: custom/design variants share ONNX with their base model
    onnx_source_cfg = cfg
    if cfg.name in ("custom", "design"):
        onnx_source_cfg = Models.base
    elif cfg.name == "custom_small":
        onnx_source_cfg = Models.base_small

    out_talker = models_dir / "qwen3_tts_talker.q5_k.gguf"
    out_predictor = models_dir / "qwen3_tts_predictor.q8_0.gguf"
    out_codec_encoder = models_dir / "qwen3_tts_codec_encoder.fp16.onnx"
    out_speaker_encoder = models_dir / "qwen3_tts_speaker_encoder.fp16.onnx"
    out_decoder = models_dir / "qwen3_tts_decoder.fp16.onnx"
    out_embeddings_dir = models_dir / "embeddings"
    out_tokenizer_json = models_dir / "tokenizer.json"
    logs_dir = REPO_ROOT / "logs" / "convert_onnx"

    models_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_convert:
        if args.force:
            for p in [out_talker, out_predictor, out_codec_encoder, out_speaker_encoder, out_decoder, out_tokenizer_json]:
                if p.exists():
                    p.unlink()
            if out_embeddings_dir.exists():
                shutil.rmtree(out_embeddings_dir)

        ensure_talker_predictor(sys.executable, base_dir, out_talker, out_predictor, out_embeddings_dir)
        ensure_embeddings(sys.executable, base_dir, out_embeddings_dir)
        ensure_tokenizer_json(base_dir, out_tokenizer_json)

        if onnx_source_cfg.name != cfg.name:
            # Copy ONNX artifacts from base variant instead of re-exporting
            onnx_base_dir = onnx_source_cfg.dest.resolve()
            onnx_files = [
                "qwen3_tts_codec_encoder.fp16.onnx",
                "qwen3_tts_speaker_encoder.fp16.onnx",
                "qwen3_tts_decoder.fp16.onnx",
            ]
            for fname in onnx_files:
                src = onnx_base_dir / fname
                dst = models_dir / fname
                if dst.exists():
                    eprint(f"[skip] ONNX exists: {dst}")
                    continue
                if not src.exists():
                    raise RuntimeError(
                        f"ONNX source not found: {src}\n"
                        f"Please convert the base model first:\n"
                        f"  python manage.py convert --model {onnx_source_cfg.name}"
                    )
                shutil.copy2(src, dst)
                eprint(f"[copy] {src} -> {dst}")
        else:
            ensure_onnx_artifacts(
                python_exe=sys.executable,
                base_dir=base_dir,
                models_dir=models_dir,
                out_codec_encoder=out_codec_encoder,
                out_speaker_encoder=out_speaker_encoder,
                out_decoder=out_decoder,
                timeout_sec=max(1, int(args.timeout_sec)),
                logs_dir=logs_dir,
                enable_quant=enable_quant,
            )

    eprint("\n[done] Model setup complete.")
    eprint(f"  Model: {cfg.name}")
    eprint(f"  - {out_talker}")
    eprint(f"  - {out_predictor}")
    eprint(f"  - {out_codec_encoder}")
    eprint(f"  - {out_speaker_encoder}")
    eprint(f"  - {out_decoder}")
    eprint(f"  - {out_embeddings_dir}")
    eprint(f"  - {out_tokenizer_json}")
    eprint("\nRun:")
    eprint(f"  ./build/qwen3-tts-cli -m {models_dir.relative_to(REPO_ROOT)} -t \"Hello\" -o out.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
