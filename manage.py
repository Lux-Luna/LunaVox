#!/usr/bin/env python3
"""Unified CLI entry point for LunaVox tooling."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "tools"
SETUP_SCRIPT = TOOLS / "setup" / "setup_pipeline.py"
BUILD_SCRIPT = TOOLS / "build_manager.py"
COMPARE_SCRIPT = TOOLS / "compare_inference.py"

EXPECTED_CONDA_ENV = "lunavox"
DEFAULT_TIMEOUT_SEC = 300
MANAGE_LOG_DIR = ROOT / "logs" / "manage"


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def require_modules(modules: Iterable[tuple[str, str]]) -> None:
    missing = [f"{name} ({pip_name})" for name, pip_name in modules if not has_module(name)]
    if missing:
        raise RuntimeError(
            "Missing required Python modules: "
            + ", ".join(missing)
            + "\nInstall them with:\n"
            + f"  {sys.executable} -m pip install "
            + " ".join(pip_name for _, pip_name in modules)
        )


def check_conda_env(expected: str) -> None:
    active_name = os.environ.get("CONDA_DEFAULT_ENV", "").strip()
    prefix_name = Path(sys.prefix).name
    if active_name == expected or prefix_name == expected:
        return
    raise RuntimeError(
        f"Expected conda environment '{expected}', but current env is "
        f"CONDA_DEFAULT_ENV='{active_name or '<empty>'}', sys.prefix='{prefix_name}'.\n"
        f"Use: conda run -n {expected} python manage.py <command> ..."
    )


def ensure_git_safe_directory(repo_root: Path, auto_fix: bool) -> None:
    repo_norm = repo_root.as_posix()
    try:
        proc = subprocess.run(
            ["git", "config", "--global", "--get-all", "safe.directory"],
            cwd=str(repo_root),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as err:
        raise RuntimeError("git is not available in PATH") from err

    existing = set()
    if proc.returncode == 0 and proc.stdout:
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line:
                existing.add(Path(line).as_posix())

    if repo_norm in existing:
        return

    if not auto_fix:
        raise RuntimeError(
            f"git safe.directory missing for repo: {repo_norm}\n"
            f"Run: git config --global --add safe.directory {repo_norm}"
        )

    add = subprocess.run(
        ["git", "config", "--global", "--add", "safe.directory", repo_norm],
        cwd=str(repo_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if add.returncode != 0:
        raise RuntimeError(
            f"Failed to add git safe.directory for {repo_norm}.\n"
            f"stderr: {add.stderr.strip() or '<empty>'}"
        )


def check_ort_sdk(root: Path) -> None:
    header = root / "include" / "onnxruntime_cxx_api.h"
    lib_dir = root / "lib"
    if not header.exists():
        raise RuntimeError(f"ONNX Runtime header missing: {header}")
    if not lib_dir.exists():
        raise RuntimeError(f"ONNX Runtime lib dir missing: {lib_dir}")


def check_preflight(
    *,
    need_convert_modules: bool,
    need_build_deps: bool,
    enable_quant: bool,
    fix_git_safe: bool,
) -> None:
    check_conda_env(EXPECTED_CONDA_ENV)
    ensure_git_safe_directory(ROOT, auto_fix=fix_git_safe)

    if need_convert_modules:
        modules = [
            ("torch", "torch"),
            ("numpy", "numpy"),
            ("tqdm", "tqdm"),
            ("safetensors", "safetensors"),
            ("gguf", "gguf"),
            ("transformers", "transformers"),
            ("onnx", "onnx"),
            ("onnxruntime", "onnxruntime"),
        ]
        if enable_quant:
            modules.append(("onnxruntime.quantization", "onnxruntime-tools"))
        require_modules(modules)

    if need_build_deps:
        ort_root = ROOT / "lib" / "onnx"
        check_ort_sdk(ort_root)


def resolve_enable_quant(args: argparse.Namespace) -> bool:
    enable_quant = bool(getattr(args, "enable_quant", False))
    skip_quant = bool(getattr(args, "skip_quant", True))
    return enable_quant or (not skip_quant)


def _safe_stage_name(stage: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", stage).strip("_") or "stage"


def run_stage_process(
    cmd: list[str],
    *,
    cwd: Path,
    timeout_sec: int,
    stage: str,
) -> int:
    MANAGE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stage_name = _safe_stage_name(stage)
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_file = MANAGE_LOG_DIR / f"{ts}-{stage_name}.log"
    start = time.time()
    eprint(f"[run:{stage_name}] {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1, int(timeout_sec)),
        )
        output = proc.stdout or ""
        elapsed = time.time() - start
        status = "ok" if proc.returncode == 0 else "failed"
        log_file.write_text(
            f"[{status}] {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"stage: {stage_name}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"cwd: {cwd}\n"
            f"timeout_sec: {timeout_sec}\n"
            f"elapsed_sec: {elapsed:.3f}\n"
            f"returncode: {proc.returncode}\n\n"
            f"{output}",
            encoding="utf-8",
        )
        if proc.returncode != 0:
            eprint(f"[run:{stage_name}] failed, log: {log_file}")
        return int(proc.returncode)
    except subprocess.TimeoutExpired as err:
        output = ""
        if err.stdout:
            output += err.stdout if isinstance(err.stdout, str) else err.stdout.decode("utf-8", errors="ignore")
        if err.stderr:
            output += err.stderr if isinstance(err.stderr, str) else err.stderr.decode("utf-8", errors="ignore")
        elapsed = time.time() - start
        log_file.write_text(
            f"[timeout] {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"stage: {stage_name}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"cwd: {cwd}\n"
            f"timeout_sec: {timeout_sec}\n"
            f"elapsed_sec: {elapsed:.3f}\n\n"
            f"{output}",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Stage '{stage_name}' timed out after {timeout_sec}s. See log: {log_file}"
        ) from err


def run_python_script(script: Path, extra_args: list[str], *, timeout_sec: int, stage: str) -> int:
    if not script.exists():
        raise RuntimeError(f"Script not found: {script}")
    cmd = [sys.executable, str(script)] + extra_args
    return run_stage_process(cmd, cwd=ROOT, timeout_sec=timeout_sec, stage=stage)


def run_build_verify(timeout_sec: int) -> int:
    build_dir = ROOT / "build-cpu"
    if os.name == "nt":
        exe = build_dir / "qwen3-tts-cli.exe"
    else:
        exe = build_dir / "qwen3-tts-cli"
    if not exe.exists():
        raise RuntimeError(f"Built CLI not found for verify: {exe}")

    rc = run_stage_process(
        [str(exe), "--help"],
        cwd=ROOT,
        timeout_sec=timeout_sec,
        stage="build_verify_help",
    )
    if rc == 0:
        print(f"[ok] verify passed: {exe.name} --help")
    return rc


def command_setup(args: argparse.Namespace) -> int:
    enable_quant = resolve_enable_quant(args)
    check_preflight(
        need_convert_modules=not args.skip_convert,
        need_build_deps=False,
        enable_quant=enable_quant,
        fix_git_safe=args.fix_git_safe,
    )
    setup_args = [
        "--models-dir",
        args.models_dir,
        "--timeout-sec",
        str(args.timeout_sec),
    ]
    if args.hf_token:
        setup_args += ["--hf-token", args.hf_token]
    if args.skip_download:
        setup_args.append("--skip-download")
    if args.skip_convert:
        setup_args.append("--skip-convert")
    if args.force:
        setup_args.append("--force")
    if enable_quant:
        setup_args.append("--enable-quant")
    return run_python_script(
        SETUP_SCRIPT,
        setup_args,
        timeout_sec=max(1, int(args.timeout_sec)),
        stage="setup_pipeline_setup",
    )


def command_convert(args: argparse.Namespace) -> int:
    enable_quant = resolve_enable_quant(args)
    check_preflight(
        need_convert_modules=True,
        need_build_deps=False,
        enable_quant=enable_quant,
        fix_git_safe=args.fix_git_safe,
    )
    setup_args = [
        "--models-dir",
        args.models_dir,
        "--timeout-sec",
        str(args.timeout_sec),
        "--skip-download",
    ]
    if args.hf_token:
        setup_args += ["--hf-token", args.hf_token]
    if args.force:
        setup_args.append("--force")
    if enable_quant:
        setup_args.append("--enable-quant")
    return run_python_script(
        SETUP_SCRIPT,
        setup_args,
        timeout_sec=max(1, int(args.timeout_sec)),
        stage="setup_pipeline_convert",
    )


def command_build(args: argparse.Namespace) -> int:
    check_preflight(
        need_convert_modules=False,
        need_build_deps=True,
        enable_quant=False,
        fix_git_safe=args.fix_git_safe,
    )
    build_args = ["--backend", args.backend, "--j", str(args.j), "--timeout-sec", str(args.timeout_sec)]
    if args.clean:
        build_args.append("--clean")
    if args.verify:
        build_args.append("--verify")
    return run_python_script(
        BUILD_SCRIPT,
        build_args,
        timeout_sec=max(1, int(args.timeout_sec)),
        stage="build_manager_build",
    )


def command_preflight(args: argparse.Namespace) -> int:
    enable_quant = resolve_enable_quant(args)
    check_preflight(
        need_convert_modules=args.check_convert,
        need_build_deps=args.check_build,
        enable_quant=enable_quant,
        fix_git_safe=args.fix_git_safe,
    )
    print("[ok] preflight checks passed")
    return 0


def command_bootstrap(args: argparse.Namespace) -> int:
    enable_quant = resolve_enable_quant(args)
    check_preflight(
        need_convert_modules=not args.skip_convert,
        need_build_deps=True,
        enable_quant=enable_quant,
        fix_git_safe=args.fix_git_safe,
    )

    setup_args = [
        "--models-dir",
        args.models_dir,
        "--timeout-sec",
        str(args.timeout_sec),
    ]
    if args.skip_download:
        setup_args.append("--skip-download")
    if args.skip_convert:
        setup_args.append("--skip-convert")
    if args.force:
        setup_args.append("--force")
    if enable_quant:
        setup_args.append("--enable-quant")

    rc = run_python_script(
        SETUP_SCRIPT,
        setup_args,
        timeout_sec=max(1, int(args.timeout_sec)),
        stage="setup_pipeline_bootstrap",
    )
    if rc != 0:
        return rc

    build_args = ["--backend", args.backend, "--j", str(args.j), "--timeout-sec", str(args.timeout_sec)]
    if args.clean:
        build_args.append("--clean")
    rc = run_python_script(
        BUILD_SCRIPT,
        build_args,
        timeout_sec=max(1, int(args.timeout_sec)),
        stage="build_manager_bootstrap",
    )
    if rc != 0:
        return rc

    if args.verify:
        return run_build_verify(timeout_sec=max(1, int(args.timeout_sec)))
    return 0


def command_compare(args: argparse.Namespace) -> int:
    check_preflight(
        need_convert_modules=False,
        need_build_deps=False,
        enable_quant=False,
        fix_git_safe=args.fix_git_safe,
    )
    compare_args = [
        "--mode",
        args.mode,
        "--timeout-sec",
        str(args.timeout_sec),
        "--report-out",
        args.report_out,
        "--lunavox-model-dir",
        args.models_dir,
        "--lunavox-build-dir",
        args.build_dir,
        "--text",
        args.text,
        "--max-steps",
        str(args.max_steps),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--predictor-temperature",
        str(args.predictor_temperature),
        "--predictor-top-k",
        str(args.predictor_top_k),
        "--predictor-top-p",
        str(args.predictor_top_p),
        "--seed",
        str(args.seed),
        "--predictor-seed",
        str(args.predictor_seed),
    ]
    if args.reference_audio:
        compare_args += ["--lunavox-reference-audio", args.reference_audio]
    if args.qwen_reference_audio:
        compare_args += ["--qwen-reference-audio", args.qwen_reference_audio]
    if args.qwen_reference_text:
        compare_args += ["--qwen-reference-text", args.qwen_reference_text]
    if args.qwen_clone_anchor_seconds is not None:
        compare_args += ["--qwen-clone-anchor-seconds", str(args.qwen_clone_anchor_seconds)]
    if args.qwen_repo:
        compare_args += ["--qwen-repo", args.qwen_repo]
    if args.qwen_model_dir:
        compare_args += ["--qwen-model-dir", args.qwen_model_dir]
    if args.strict_qwen_model:
        compare_args.append("--strict-qwen-model")
    else:
        compare_args.append("--no-strict-qwen-model")
    if args.qwen_python:
        compare_args += ["--qwen-python", args.qwen_python]
    if args.keep_artifacts:
        compare_args.append("--keep-artifacts")

    return run_python_script(
        COMPARE_SCRIPT,
        compare_args,
        timeout_sec=max(1, int(args.timeout_sec)),
        stage="compare_inference",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LunaVox management tool")
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="Download/convert all runtime artifacts")
    p_setup.add_argument("--models-dir", default=str(ROOT / "models" / "base_small"))
    p_setup.add_argument("--hf-token", default="")
    p_setup.add_argument("--fix-git-safe", action=argparse.BooleanOptionalAction, default=True)
    p_setup.add_argument("--skip-download", action="store_true")
    p_setup.add_argument("--skip-convert", action="store_true")
    p_setup.add_argument("--force", action="store_true")
    p_setup.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    p_setup.add_argument(
        "--skip-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip optional local ONNX int8 quantization (default: true)",
    )
    p_setup.add_argument("--enable-quant", action="store_true", help="Alias of --no-skip-quant")
    p_setup.set_defaults(func=command_setup)

    p_convert = sub.add_parser("convert", help="Convert artifacts only (implies --skip-download)")
    p_convert.add_argument("--models-dir", default=str(ROOT / "models" / "base_small"))
    p_convert.add_argument("--hf-token", default="")
    p_convert.add_argument("--fix-git-safe", action=argparse.BooleanOptionalAction, default=True)
    p_convert.add_argument("--force", action="store_true")
    p_convert.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    p_convert.add_argument(
        "--skip-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip optional local ONNX int8 quantization (default: true)",
    )
    p_convert.add_argument("--enable-quant", action="store_true", help="Alias of --no-skip-quant")
    p_convert.set_defaults(func=command_convert)

    p_build = sub.add_parser("build", help="Build C++ runtime")
    p_build.add_argument("--backend", choices=["cpu"], default="cpu")
    p_build.add_argument("--clean", action="store_true")
    p_build.add_argument("--j", type=int, default=4)
    p_build.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    p_build.add_argument("--verify", action="store_true")
    p_build.add_argument("--fix-git-safe", action=argparse.BooleanOptionalAction, default=True)
    p_build.set_defaults(func=command_build)

    p_preflight = sub.add_parser("preflight", help="Run environment checks")
    p_preflight.add_argument("--check-convert", action=argparse.BooleanOptionalAction, default=True)
    p_preflight.add_argument("--check-build", action=argparse.BooleanOptionalAction, default=True)
    p_preflight.add_argument(
        "--skip-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip optional local ONNX int8 quantization checks (default: true)",
    )
    p_preflight.add_argument("--enable-quant", action="store_true", help="Alias of --no-skip-quant")
    p_preflight.add_argument("--fix-git-safe", action=argparse.BooleanOptionalAction, default=True)
    p_preflight.set_defaults(func=command_preflight)

    p_bootstrap = sub.add_parser("bootstrap", help="One-command setup + build + verify")
    p_bootstrap.add_argument("--backend", choices=["cpu"], default="cpu")
    p_bootstrap.add_argument("--models-dir", default=str(ROOT / "models" / "base_small"))
    p_bootstrap.add_argument("--skip-download", action="store_true")
    p_bootstrap.add_argument("--skip-convert", action="store_true")
    p_bootstrap.add_argument("--force", action="store_true")
    p_bootstrap.add_argument(
        "--skip-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip optional local ONNX int8 quantization (default: true)",
    )
    p_bootstrap.add_argument("--enable-quant", action="store_true", help="Alias of --no-skip-quant")
    p_bootstrap.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    p_bootstrap.add_argument("--clean", action="store_true")
    p_bootstrap.add_argument("--j", type=int, default=4)
    p_bootstrap.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    p_bootstrap.add_argument("--fix-git-safe", action=argparse.BooleanOptionalAction, default=True)
    p_bootstrap.set_defaults(func=command_bootstrap)

    p_compare = sub.add_parser("compare", help="Compare lunavox vs Qwen3-TTS-GGUF inference chain")
    p_compare.add_argument("--mode", choices=["base", "clone", "both"], default="both")
    p_compare.add_argument("--text", default="LunaVox alignment test.")
    p_compare.add_argument("--reference-audio", default=str(ROOT / "ref" / "ref.wav"))
    p_compare.add_argument("--qwen-reference-audio", default="")
    p_compare.add_argument("--qwen-reference-text", default="")
    p_compare.add_argument(
        "--qwen-clone-anchor-seconds",
        type=float,
        default=0.0,
        help="Qwen-side clone-only reference clipping seconds for root-cause isolation (default: 0.0)",
    )
    p_compare.add_argument("--models-dir", default=str(ROOT / "models" / "base_small"))
    p_compare.add_argument("--build-dir", default=str(ROOT / "build-cpu"))
    p_compare.add_argument("--qwen-repo", default=str(ROOT.parent / "Qwen3-TTS-GGUF"))
    p_compare.add_argument("--qwen-model-dir", default=str(ROOT / "models" / "base_small"))
    p_compare.add_argument(
        "--strict-qwen-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail compare immediately when --qwen-model-dir misses required core artifacts (default: true)",
    )
    p_compare.add_argument("--qwen-python", default="")
    p_compare.add_argument("--report-out", default=str(ROOT / "logs" / "compare" / "latest_compare_report.json"))
    p_compare.add_argument("--max-steps", type=int, default=220)
    p_compare.add_argument("--temperature", type=float, default=0.6)
    p_compare.add_argument("--top-k", type=int, default=50)
    p_compare.add_argument("--top-p", type=float, default=1.0)
    p_compare.add_argument("--predictor-temperature", type=float, default=0.6)
    p_compare.add_argument("--predictor-top-k", type=int, default=50)
    p_compare.add_argument("--predictor-top-p", type=float, default=1.0)
    p_compare.add_argument("--seed", type=int, default=12345)
    p_compare.add_argument("--predictor-seed", type=int, default=12345)
    p_compare.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    p_compare.add_argument("--keep-artifacts", action="store_true")
    p_compare.add_argument("--fix-git-safe", action=argparse.BooleanOptionalAction, default=True)
    p_compare.set_defaults(func=command_compare)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 1
    except RuntimeError as err:
        eprint(f"Error: {err}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
