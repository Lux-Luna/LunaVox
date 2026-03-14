#!/usr/bin/env python3
"""
One-shot model setup for qwen3-tts.cpp.

This script downloads required Hugging Face model assets and generates all model
artifacts needed by the final C++ pipeline:

- models/qwen3-tts-0.6b-f16.gguf
- models/qwen3-tts-tokenizer-f16.gguf
- models/coreml/code_predictor.mlpackage (optional, macOS)

Example:
  python scripts/setup_pipeline_models.py

Minimal usage for CI/offline conversion:
  python scripts/setup_pipeline_models.py --skip-download
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
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
    # Prefer the speech_tokenizer assets from base repo when present.
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
    out_tts: Path,
    out_tok: Path,
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

    if force_convert and out_tts.exists():
        out_tts.unlink()
    if force_convert and out_tok.exists():
        out_tok.unlink()

    if not out_tts.exists():
        run_cmd(
            [
                python_exe,
                str(TOOLS_DIR / "conversion" / "convert_tts_to_gguf.py"),
                "--input",
                str(base_dir),
                "--output",
                str(out_tts),
                "--type",
                "f16",
            ],
            cwd=REPO_ROOT,
        )
    else:
        eprint(f"[ok] exists: {out_tts}")

    if not out_tok.exists():
        run_cmd(
            [
                python_exe,
                str(TOOLS_DIR / "conversion" / "convert_tokenizer_to_gguf.py"),
                "--input",
                str(tokenizer_input_dir),
                "--output",
                str(out_tok),
                "--type",
                "f16",
            ],
            cwd=REPO_ROOT,
        )
    else:
        eprint(f"[ok] exists: {out_tok}")


def export_coreml(
    python_exe: str,
    base_dir: Path,
    out_coreml: Path,
    force_convert: bool,
) -> None:
    require_modules(
        [
            ("coremltools", "coremltools"),
            ("torch", "torch"),
            ("safetensors", "safetensors"),
            ("numpy", "numpy"),
        ]
    )

    if force_convert and out_coreml.exists():
        if out_coreml.is_dir():
            shutil.rmtree(out_coreml)
        else:
            out_coreml.unlink()

    if out_coreml.exists():
        eprint(f"[ok] exists: {out_coreml}")
        return

    out_coreml.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            python_exe,
            str(TOOLS_DIR / "conversion" / "convert_code_predictor_to_coreml.py"),
            "--input",
            str(base_dir),
            "--output",
            str(out_coreml),
        ],
        cwd=REPO_ROOT,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and prepare all runtime models for qwen3-tts.cpp")
    p.add_argument("--models-dir", default=str(REPO_ROOT / "models"), help="Target models directory")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""), help="Hugging Face token (or set HF_TOKEN)")
    p.add_argument("--skip-download", action="store_true", help="Skip model downloads")
    p.add_argument("--skip-gguf", action="store_true", help="Skip GGUF conversion")
    p.add_argument(
        "--coreml",
        choices=["auto", "on", "off"],
        default="auto",
        help="CoreML export mode: auto=macOS only, on=force, off=disable",
    )
    p.add_argument("--force", action="store_true", help="Re-download/re-generate outputs")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    models_dir = Path(args.models_dir).resolve()
    base_dir = models_dir / "Qwen3-TTS-12Hz-0.6B-Base"
    tokenizer_dir = models_dir / "Qwen3-TTS-Tokenizer-12Hz"
    out_tts = models_dir / "qwen3-tts-0.6b-f16.gguf"
    out_tok = models_dir / "qwen3-tts-tokenizer-f16.gguf"
    out_coreml = models_dir / "coreml" / "code_predictor.mlpackage"

    hf_token = args.hf_token.strip() or None

    models_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        require_modules([("huggingface_hub", "huggingface_hub")])
        ensure_base_assets(base_dir, hf_token, args.force)
        tokenizer_input_dir = ensure_tokenizer_assets(base_dir, tokenizer_dir, hf_token, args.force)
    else:
        tokenizer_input_dir = base_dir if (base_dir / "speech_tokenizer" / "model.safetensors").exists() else tokenizer_dir

    if not args.skip_gguf:
        convert_gguf(sys.executable, base_dir, tokenizer_input_dir, out_tts, out_tok, args.force)

    wants_coreml = args.coreml == "on" or (args.coreml == "auto" and platform.system() == "Darwin")
    if wants_coreml:
        if platform.system() != "Darwin":
            raise RuntimeError("CoreML export requested on non-macOS platform")
        export_coreml(sys.executable, base_dir, out_coreml, args.force)

    eprint("\n[done] Model setup complete.")
    eprint(f"  - {out_tts}")
    eprint(f"  - {out_tok}")
    if wants_coreml:
        eprint(f"  - {out_coreml}")
        eprint("\nRun (CoreML is enabled by default on macOS):")
        eprint("  ./build/qwen3-tts-cli -m models -t \"Hello\" -o out.wav")
        eprint("  # Optional override path:")
        eprint("  QWEN3_TTS_COREML_MODEL=models/coreml/code_predictor.mlpackage ./build/qwen3-tts-cli -m models -t \"Hello\" -o out.wav")
    else:
        eprint("\nRun without CoreML:")
        eprint("  ./build/qwen3-tts-cli -m models -t \"Hello\" -o out.wav")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
