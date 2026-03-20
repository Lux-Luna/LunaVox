#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"[run] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def ensure_modelscope_layout(base_dir: Path, home_dir: Path) -> Path:
    target = home_dir / ".cache" / "modelscope" / "hub" / "models" / "Qwen" / "Qwen3-TTS-12Hz-0.6B-Base"
    if target.exists() and (target / "model.safetensors").exists():
        print(f"[ok] using existing modelscope base dir: {target}", file=sys.stderr)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    print(f"[sync] copy base model into modelscope cache: {base_dir} -> {target}", file=sys.stderr)
    shutil.copytree(base_dir, target)
    return target


def copy_onnx_from_dir(source_dir: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "qwen3_tts_codec_encoder.fp16.onnx": [
            source_dir / "qwen3_tts_codec_encoder.fp16.onnx",
            source_dir / "qwen3_tts_codec_encoder.fp32.onnx",
            source_dir / "qwen3_tts_codec_encoder.onnx",
        ],
        "qwen3_tts_speaker_encoder.fp16.onnx": [
            source_dir / "qwen3_tts_speaker_encoder.fp16.onnx",
            source_dir / "qwen3_tts_speaker_encoder.fp32.onnx",
            source_dir / "qwen3_tts_speaker_encoder.onnx",
        ],
        "qwen3_tts_decoder.fp16.onnx": [
            source_dir / "qwen3_tts_decoder.fp16.onnx",
            source_dir / "qwen3_tts_decoder.fp32.onnx",
            source_dir / "qwen3_tts_decoder.onnx",
        ],
    }

    missing: list[str] = []
    for dst_name, candidates in mapping.items():
        src = next((p for p in candidates if p.exists()), None)
        if src is None:
            missing.append(dst_name)
            continue
        dst = out_dir / dst_name
        shutil.copy2(src, dst)
        print(f"[ok] {src.name} -> {dst}", file=sys.stderr)
    return missing


def download_prebuilt_onnx(prebuilt_repo: str, out_dir: Path, cache_home: Path) -> None:
    from huggingface_hub import hf_hub_download

    hf_home = cache_home / ".hf_home"
    hf_cache = hf_home / "hub"
    hf_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_cache)
    os.environ["XDG_CACHE_HOME"] = str(hf_home)

    prebuilt_map: Dict[str, str] = {
        "qwen3_tts_codec_encoder.fp16.onnx": "onnx/qwen3_tts_codec_encoder.onnx",
        "qwen3_tts_speaker_encoder.fp16.onnx": "onnx/qwen3_tts_speaker_encoder.onnx",
        "qwen3_tts_decoder.fp16.onnx": "onnx/qwen3_tts_decoder.onnx",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for dst_name, repo_file in prebuilt_map.items():
        downloaded = hf_hub_download(
            repo_id=prebuilt_repo,
            filename=repo_file,
            local_dir=str(out_dir),
        )
        dst = out_dir / dst_name
        shutil.copy2(downloaded, dst)
        print(f"[ok] {repo_file} -> {dst}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export Qwen3-TTS ONNX artifacts via reference Qwen3-TTS-GGUF scripts")
    ap.add_argument("--base-dir", required=True, help="Base model directory (Qwen3-TTS-12Hz-0.6B-Base)")
    ap.add_argument("--tokenizer-dir", required=True, help="Tokenizer model directory (unused; kept for compatibility)")
    ap.add_argument("--output-dir", required=True, help="Destination directory for exported ONNX files")
    ap.add_argument("--ref-repo", default="", help="Path to Qwen3-TTS-GGUF repo (default: sibling repo)")
    ap.add_argument("--prebuilt-repo", default="", help="Optional HF repo for prebuilt ONNX fallback")
    ap.add_argument("--prebuilt-only", action="store_true", help="Skip local export, only download prebuilt ONNX")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    ref_repo = Path(args.ref_repo).resolve() if args.ref_repo else (Path(__file__).resolve().parents[3] / "Qwen3-TTS-GGUF")

    if not base_dir.exists():
        raise RuntimeError(f"Base model directory does not exist: {base_dir}")

    cache_home_env = os.environ.get("LUNAVOX_MODELSCOPE_HOME", "").strip()
    if cache_home_env:
        cache_home = Path(cache_home_env).expanduser()
    else:
        cache_home = Path(__file__).resolve().parents[2] / ".cache_home"
    cache_home.mkdir(parents=True, exist_ok=True)

    if args.prebuilt_only:
        if not args.prebuilt_repo:
            raise RuntimeError("--prebuilt-only requires --prebuilt-repo")
        download_prebuilt_onnx(args.prebuilt_repo, out_dir, cache_home)
        print("[done] ONNX download completed", file=sys.stderr)
        return 0

    try:
        if not ref_repo.exists():
            raise RuntimeError(
                f"Reference repo not found: {ref_repo}\n"
                "Set --ref-repo to your local Qwen3-TTS-GGUF clone."
            )

        ensure_modelscope_layout(base_dir, cache_home)

        scripts = [
            "11-Export-Codec-Encoder.py",
            "12-Export-Speaker-Encoder.py",
            "13-Export-Decoder.py",
            "16-Quantize-ONNX-Models.py",
        ]
        env = os.environ.copy()
        # Reference scripts resolve "~/.cache/modelscope/..." from HOME/USERPROFILE.
        env["HOME"] = str(cache_home)
        env["USERPROFILE"] = str(cache_home)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        py_root = Path(sys.executable).resolve().parent
        extra_path_entries = [
            py_root,
            py_root / "Library" / "bin",
            py_root / "Scripts",
        ]
        path_parts = [str(p) for p in extra_path_entries if p.exists()]
        if path_parts:
            env["PATH"] = os.pathsep.join(path_parts + [env.get("PATH", "")])
        for s in scripts:
            p = ref_repo / s
            if not p.exists():
                raise RuntimeError(f"Missing export script in reference repo: {p}")
            run([sys.executable, str(p)], cwd=ref_repo, env=env)

        export_dir = ref_repo / "model-base-small"
        if not export_dir.exists():
            raise RuntimeError(f"Reference export directory not found: {export_dir}")

        missing = copy_onnx_from_dir(export_dir, out_dir)
        if missing:
            raise RuntimeError("Missing exported ONNX artifacts: " + ", ".join(missing))
    except Exception as export_err:
        if not args.prebuilt_repo:
            raise
        print(f"[warn] local ONNX export failed: {export_err}", file=sys.stderr)
        print(f"[warn] falling back to prebuilt repo: {args.prebuilt_repo}", file=sys.stderr)
        download_prebuilt_onnx(args.prebuilt_repo, out_dir, cache_home)

    print("[done] ONNX export completed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
