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
from typing import Iterable, Optional

# Force UTF-8 for Windows console
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

if os.name == "nt":
    try:
        import ctypes
        # Set console code page to UTF-8
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception: pass

from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.theme import Theme

# Custom theme for professional look
THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "stage": "bold magenta",
})

# Use force_terminal=True and legacy_windows=True for better character handling
console = Console(theme=THEME, force_terminal=True, safe_box=True)

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "tools"
SETUP_SCRIPT = TOOLS / "setup" / "setup_pipeline.py"
BUILD_SCRIPT = TOOLS / "build_manager.py"
EXPECTED_CONDA_ENV = "lunavox"
DEFAULT_TIMEOUT_SEC = 300
LOG_DIR = ROOT / "logs"
LATEST_LOG = LOG_DIR / "latest.log"

# Valid model variants for --model argument
VALID_MODELS = ["base", "custom", "design", "base_small", "custom_small"]


class UI:
    """Helper for consistent, premium CLI output."""
    
    @staticmethod
    def banner(title: str):
        console.print(Panel(title, style="stage", expand=False))

    @staticmethod
    def info(msg: str):
        console.print(f"[info]ℹ[/info] {msg}")

    @staticmethod
    def success(msg: str):
        console.print(f"[success]✔[/success] {msg}")

    @staticmethod
    def warn(msg: str):
        console.print(f"[warning]⚠[/warning] {msg}")

    @staticmethod
    def error(msg: str):
        console.print(f"[error]✘[/error] {msg}")

    @staticmethod
    def table(title: str, columns: list[str], rows: list[list[str]]):
        table = Table(title=title, show_header=True, header_style="bold cyan")
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*row)
        console.print(table)


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
    UI.warn(f"Conda environment mismatch. Expected '{expected}', got '{active_name or prefix_name}'")


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
        UI.warn(f"git safe.directory missing for repo: {repo_norm}")
        return

    UI.info(f"Adding {repo_norm} to git safe.directory...")
    add = subprocess.run(
        ["git", "config", "--global", "--add", "safe.directory", repo_norm],
        cwd=str(repo_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if add.returncode != 0:
        raise RuntimeError(f"Failed to add git safe.directory for {repo_norm}.")


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
    UI.info("Running pre-flight checks...")
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
    verbose: bool = False,
) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stage_name = _safe_stage_name(stage)
    start = time.time()
    
    header = (
        f"\n{'='*80}\n"
        f"STAGE: {stage_name}\n"
        f"TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"CMD: {' '.join(cmd)}\n"
        f"CWD: {cwd}\n"
        f"{'='*80}\n"
    )
    
    try:
        if verbose:
            UI.info(f"Executing {stage_name} (verbose mode)...")
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(1, int(timeout_sec)),
                stdout=None,
                stderr=None,
            )
            rc, output = proc.returncode, "(Output shown in console)"
        else:
            with console.status(f"[bold cyan]Executing {stage_name}...", spinner="dots"):
                proc = subprocess.run(
                    cmd,
                    cwd=str(cwd),
                    check=False,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=max(1, int(timeout_sec)),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                rc, output = proc.returncode, proc.stdout or ""
    except subprocess.TimeoutExpired as err:
        elapsed = time.time() - start
        out_bytes = err.stdout or b""
        output = out_bytes.decode("utf-8", "replace") if isinstance(out_bytes, bytes) else str(out_bytes)
        log_entry = f"{header}STATUS: timeout\nELAPSED: {elapsed:.3f}s\n\n{output}\n"
        with open(LATEST_LOG, "a", encoding="utf-8") as f: f.write(log_entry)
        UI.error(f"{stage_name} timed out after {timeout_sec}s")
        return 1

    elapsed = time.time() - start
    status = "ok" if rc == 0 else "failed"
    
    log_entry = (
        f"{header}"
        f"STATUS: {status}\n"
        f"ELAPSED: {elapsed:.3f}s\n"
        f"RETURNCODE: {rc}\n\n"
        f"{output}\n"
    )
    
    with open(LATEST_LOG, "a", encoding="utf-8") as f:
        f.write(log_entry)
        
    if rc == 0:
        UI.success(f"{stage_name} completed in {elapsed:.2f}s")
    else:
        UI.error(f"{stage_name} failed (rc={rc}). See log: {LATEST_LOG}")
    return int(rc)


def run_python_script(script: Path, extra_args: list[str], *, timeout_sec: int, stage: str, verbose: bool = False) -> int:
    if not script.exists():
        raise RuntimeError(f"Script not found: {script}")
    cmd = [sys.executable, str(script)] + extra_args
    return run_stage_process(cmd, cwd=ROOT, timeout_sec=timeout_sec, stage=stage, verbose=verbose)


def run_build_verify(timeout_sec: int) -> int:
    build_dir = ROOT / "build"
    exe_name = "qwen3-tts-cli"
    exe = build_dir / (f"{exe_name}.exe" if os.name == "nt" else exe_name)
        
    if not exe.exists():
        raise RuntimeError(f"Built CLI not found for verify: {exe}")

    return run_stage_process(
        [str(exe), "--help"],
        cwd=ROOT,
        timeout_sec=timeout_sec,
        stage="verify_help",
    )


def make_build_args(args: argparse.Namespace) -> list[str]:
    build_args = ["--j", str(args.j), "--timeout-sec", str(args.timeout_sec)]
    if getattr(args, "clean", False):
        build_args.append("--clean")
    if getattr(args, "toolchain", "auto") != "auto":
        build_args += ["--toolchain", args.toolchain]
    return build_args


def command_setup(args: argparse.Namespace) -> int:
    UI.banner("Stage: Setup Pipeline")
    enable_quant = resolve_enable_quant(args)
    check_preflight(
        need_convert_modules=not getattr(args, "skip_convert", False),
        need_build_deps=False,
        enable_quant=enable_quant,
        fix_git_safe=getattr(args, "fix_git_safe", True),
    )
    setup_args = ["--model", args.model, "--timeout-sec", str(args.timeout_sec)]
    if getattr(args, "models_dir", None):
        setup_args += ["--models-dir", args.models_dir]
    if getattr(args, "skip_convert", False):
        setup_args.append("--skip-convert")
    if getattr(args, "force", False):
        setup_args.append("--force")
    if enable_quant:
        setup_args.append("--enable-quant")
        
    return run_python_script(SETUP_SCRIPT, setup_args, timeout_sec=args.timeout_sec, stage="setup")


def command_build(args: argparse.Namespace) -> int:
    UI.banner("Stage: C++ Build")
    check_preflight(
        need_convert_modules=False,
        need_build_deps=True,
        enable_quant=False,
        fix_git_safe=getattr(args, "fix_git_safe", True),
    )
    build_args = make_build_args(args)
    if getattr(args, "verify", False):
        build_args.append("--verify")
        
    return run_python_script(BUILD_SCRIPT, build_args, timeout_sec=args.timeout_sec, stage="build", verbose=getattr(args, "verbose", False))


def command_bootstrap(args: argparse.Namespace) -> int:
    UI.banner("LunaVox Bootstrap")
    UI.info(f"Starting full setup & build for model: [bold]{args.model}[/bold]")
    
    rc = command_setup(args)
    if rc != 0: return rc
        
    return command_build(args)


def command_download(args: argparse.Namespace) -> int:
    UI.banner("Stage: Download Models")
    from tools.download.models_downloader import download_model

    models = VALID_MODELS if args.all else [args.model]
    for m in models:
        UI.info(f"Downloading model: {m}...")
        try:
            download_model(m)
            UI.success(f"Downloaded {m}")
        except Exception as e:
            UI.error(f"Failed to download {m}: {e}")
            return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LunaVox management tool")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_setup(p):
        p.add_argument("--model", choices=VALID_MODELS, default="base_small", help="Model variant")
        p.add_argument("--models-dir", default="", help="Override output directory")
        p.add_argument("--skip-convert", action="store_true", help="Skip ONNX conversion if artifacts exist")
        p.add_argument("--force", action="store_true", help="Force re-conversion")
        p.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
        p.add_argument("--skip-quant", action=argparse.BooleanOptionalAction, default=True)
        p.add_argument("--enable-quant", action="store_true", help="Alias of --no-skip-quant")
        p.add_argument("--fix-git-safe", action=argparse.BooleanOptionalAction, default=True)

    def add_common_build(p):
        # Get existing argument strings to avoid duplication in bootstrap
        existing = {a.option_strings[0] for a in p._actions if a.option_strings}
        
        p.add_argument("--clean", action="store_true", help="Clean build directory")
        p.add_argument("--j", type=int, default=4, help="Parallel build jobs")
        p.add_argument("--verbose", action="store_true", help="Show real-time build output")
        p.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
        p.add_argument("--toolchain", choices=["auto", "msvc", "mingw", "clang", "gcc"], default="auto")
        
        if "--timeout-sec" not in existing:
            p.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
        if "--fix-git-safe" not in existing:
            p.add_argument("--fix-git-safe", action=argparse.BooleanOptionalAction, default=True)

    p_setup = sub.add_parser("setup", help="Prepare model artifacts")
    add_common_setup(p_setup)
    p_setup.set_defaults(func=command_setup)

    p_build = sub.add_parser("build", help="Build C++ runtime")
    add_common_build(p_build)
    p_build.set_defaults(func=command_build)

    p_bootstrap = sub.add_parser("bootstrap", help="One-command: Setup + Build + Verify")
    add_common_setup(p_bootstrap)
    add_common_build(p_bootstrap)
    p_bootstrap.set_defaults(func=command_bootstrap)

    p_download = sub.add_parser("download", help="Download model source")
    p_download.add_argument("--model", choices=VALID_MODELS, default="base_small")
    p_download.add_argument("--all", action="store_true")
    p_download.add_argument("--fix-git-safe", action=argparse.BooleanOptionalAction, default=True)
    p_download.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    p_download.set_defaults(func=command_download)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LATEST_LOG, "w", encoding="utf-8") as f:
        f.write(f"LunaVox Manager - Session Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    try:
        rc = int(args.func(args))
        if rc == 0:
            console.print("\n[success]✨ All tasks completed successfully![/success]")
        return rc
    except KeyboardInterrupt:
        UI.warn("Interrupted by user.")
        return 1
    except Exception as err:
        console.print(Panel(f"[bold red]Unexpected Error:[/bold red]\n{str(err)}", border_style="red"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
