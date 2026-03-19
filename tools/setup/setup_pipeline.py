#!/usr/bin/env python3
"""
One-shot model setup for LunaVox.

This script downloads required Hugging Face model assets and generates all model
artifacts needed by the current C++ pipeline:

- models/qwen3_tts_talker.q5_k.gguf
- models/qwen3_tts_predictor.q8_0.gguf
- models/qwen3_tts_speaker_encoder.gguf
- models/qwen3_tts_codec_encoder.gguf
- models/qwen3_tts_codec_decoder.gguf
- models/embeddings/*
- models/tokenizer.json

Default output directory:
- models/base_small/
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"

BASE_REPO_IDS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-0.6B-Base",
]
TOKENIZER_REPO_IDS = [
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
]


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def run_cmd(cmd: list[str], cwd: Path, env: Optional[dict[str, str]] = None) -> None:
    eprint(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


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


def snapshot_download_repo(
    repo_ids: list[str],
    local_dir: Path,
    token: Optional[str],
    allow_patterns: Optional[list[str]],
) -> None:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for repo_id in repo_ids:
        try:
            eprint(f"[download] {repo_id} -> {local_dir}")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                token=token,
                allow_patterns=allow_patterns,
                resume_download=True,
            )
            return
        except Exception as err:
            last_err = err
            eprint(f"[warn] failed to download {repo_id}: {err}")

    if last_err is not None:
        raise last_err
    raise RuntimeError("No model repositories configured")


def ensure_base_assets(base_dir: Path, token: Optional[str], force_download: bool) -> None:
    required = [
        base_dir / "config.json",
        base_dir / "model.safetensors",
        base_dir / "vocab.json",
        base_dir / "merges.txt",
        base_dir / "tokenizer_config.json",
    ]

    if force_download and base_dir.exists():
        eprint(f"[clean] removing {base_dir}")
        shutil.rmtree(base_dir)

    if all(p.exists() for p in required):
        eprint(f"[ok] base assets already present in {base_dir}")
        return

    allow_patterns = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
        "speech_tokenizer/*",
    ]
    snapshot_download_repo(BASE_REPO_IDS, base_dir, token, allow_patterns)


def ensure_tokenizer_assets(
    base_dir: Path,
    tokenizer_dir: Path,
    token: Optional[str],
    force_download: bool,
) -> Path:
    in_base = base_dir / "speech_tokenizer" / "model.safetensors"
    if in_base.exists():
        eprint("[ok] using tokenizer assets from base repo (speech_tokenizer/)")
        return base_dir

    if force_download and tokenizer_dir.exists():
        eprint(f"[clean] removing {tokenizer_dir}")
        shutil.rmtree(tokenizer_dir)

    req_tok = [tokenizer_dir / "config.json", tokenizer_dir / "model.safetensors"]
    if not all(p.exists() for p in req_tok):
        allow_patterns = [
            "config.json",
            "configuration.json",
            "model.safetensors",
            "preprocessor_config.json",
        ]
        snapshot_download_repo(TOKENIZER_REPO_IDS, tokenizer_dir, token, allow_patterns)

    return tokenizer_dir


def convert_gguf(
    python_exe: str,
    base_dir: Path,
    tokenizer_input_dir: Path,
    out_talker: Path,
    out_predictor: Path,
    out_speaker: Path,
    out_codec_encoder: Path,
    out_codec_decoder: Path,
    out_embeddings_dir: Path,
    out_tokenizer_json: Path,
    force_convert: bool,
) -> None:
    require_modules(
        [
            ("gguf", "gguf"),
            ("torch", "torch"),
            ("safetensors", "safetensors"),
            ("numpy", "numpy"),
            ("tqdm", "tqdm"),
        ]
    )

    outputs = [
        out_talker,
        out_predictor,
        out_speaker,
        out_codec_encoder,
        out_codec_decoder,
    ]
    if force_convert:
        for p in outputs:
            if p.exists():
                p.unlink()
        if out_embeddings_dir.exists():
            shutil.rmtree(out_embeddings_dir)
        if out_tokenizer_json.exists():
            out_tokenizer_json.unlink()

    if out_talker.exists() and out_predictor.exists():
        eprint(f"[ok] exists: {out_talker}")
        eprint(f"[ok] exists: {out_predictor}")

    if not out_talker.exists() or not out_predictor.exists():
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

    if not out_speaker.exists():
        run_cmd(
            [
                python_exe,
                str(TOOLS_DIR / "conversion" / "convert_tts_to_gguf.py"),
                "--input",
                str(base_dir),
                "--output",
                str(out_speaker),
                "--type",
                "f16",
                "--speaker-type",
                "f16",
                "--modules",
                "spk_enc",
            ],
            cwd=REPO_ROOT,
        )
    else:
        eprint(f"[ok] exists: {out_speaker}")

    if not out_codec_encoder.exists():
        run_cmd(
            [
                python_exe,
                str(TOOLS_DIR / "conversion" / "convert_tokenizer_to_gguf.py"),
                "--input",
                str(tokenizer_input_dir),
                "--output",
                str(out_codec_encoder),
                "--type",
                "f16",
                "--modules",
                "tok_enc",
            ],
            cwd=REPO_ROOT,
        )
    else:
        eprint(f"[ok] exists: {out_codec_encoder}")

    if not out_codec_decoder.exists():
        run_cmd(
            [
                python_exe,
                str(TOOLS_DIR / "conversion" / "convert_tokenizer_to_gguf.py"),
                "--input",
                str(tokenizer_input_dir),
                "--output",
                str(out_codec_decoder),
                "--type",
                "f16",
                "--modules",
                "tok_dec",
            ],
            cwd=REPO_ROOT,
        )
    else:
        eprint(f"[ok] exists: {out_codec_decoder}")

    text_emb = out_embeddings_dir / "text_embedding_projected.npy"
    codec_emb0 = out_embeddings_dir / "codec_embedding_0.npy"
    if not text_emb.exists() or not codec_emb0.exists():
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
    else:
        eprint(f"[ok] exists: {out_embeddings_dir}")

    src_tokenizer_json = base_dir / "tokenizer.json"
    if src_tokenizer_json.exists():
        out_tokenizer_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_tokenizer_json, out_tokenizer_json)
    else:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
            if hasattr(tok, "backend_tokenizer"):
                out_tokenizer_json.parent.mkdir(parents=True, exist_ok=True)
                tok.backend_tokenizer.save(str(out_tokenizer_json))
            else:
                eprint("[warn] tokenizer backend not available; tokenizer.json not generated")
        except Exception as err:
            eprint(f"[warn] failed to generate tokenizer.json: {err}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and prepare all runtime models for lunavox")
    p.add_argument(
        "--models-dir",
        default=str(REPO_ROOT / "models" / "base_small"),
        help="Target models directory",
    )
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""), help="Hugging Face token (or set HF_TOKEN)")
    p.add_argument("--skip-download", action="store_true", help="Skip model downloads")
    p.add_argument("--skip-gguf", action="store_true", help="Skip GGUF conversion")
    p.add_argument("--force", action="store_true", help="Re-download/re-generate outputs")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    models_dir = Path(args.models_dir).resolve()
    base_dir = models_dir / "Qwen3-TTS-12Hz-0.6B-Base"
    tokenizer_dir = models_dir / "Qwen3-TTS-Tokenizer-12Hz"
    out_talker = models_dir / "qwen3_tts_talker.q5_k.gguf"
    out_predictor = models_dir / "qwen3_tts_predictor.q8_0.gguf"
    out_speaker = models_dir / "qwen3_tts_speaker_encoder.gguf"
    out_codec_encoder = models_dir / "qwen3_tts_codec_encoder.gguf"
    out_codec_decoder = models_dir / "qwen3_tts_codec_decoder.gguf"
    out_embeddings_dir = models_dir / "embeddings"
    out_tokenizer_json = models_dir / "tokenizer.json"

    hf_token = args.hf_token.strip() or None
    models_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        require_modules([("huggingface_hub", "huggingface_hub")])
        ensure_base_assets(base_dir, hf_token, args.force)
        tokenizer_input_dir = ensure_tokenizer_assets(base_dir, tokenizer_dir, hf_token, args.force)
    else:
        tokenizer_input_dir = base_dir if (base_dir / "speech_tokenizer" / "model.safetensors").exists() else tokenizer_dir

    if not args.skip_gguf:
        convert_gguf(
            sys.executable,
            base_dir,
            tokenizer_input_dir,
            out_talker,
            out_predictor,
            out_speaker,
            out_codec_encoder,
            out_codec_decoder,
            out_embeddings_dir,
            out_tokenizer_json,
            args.force,
        )

    eprint("\n[done] Model setup complete.")
    eprint(f"  - {out_talker}")
    eprint(f"  - {out_predictor}")
    eprint(f"  - {out_speaker}")
    eprint(f"  - {out_codec_encoder}")
    eprint(f"  - {out_codec_decoder}")
    eprint(f"  - {out_embeddings_dir}")
    eprint(f"  - {out_tokenizer_json}")
    eprint("\nRun:")
    eprint("  ./build-cpu/qwen3-tts-cli -m models/base_small -t \"Hello\" -o out.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
