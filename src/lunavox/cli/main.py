from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from lunavox.build.lib_downloader import LIBS_CONFIG, update_library, download_platform_libs
from lunavox.build.main import run_build
from lunavox.core.deps import DEPENDENCY_GROUPS, DependencyPolicy, ensure_dependency_group, missing_modules
from lunavox.core.project import resolve_project_root
from lunavox.core.ui import console
from lunavox.model import ModelDownloader, ModelConvertPipeline, Models

app = typer.Typer(no_args_is_help=True, help="LunaVox unified CLI")


@dataclass
class RuntimeState:
    project_root: Path
    yes: bool
    no_install: bool
    verbose: bool
    latest_log: Path


def _state(ctx: typer.Context) -> RuntimeState:
    obj = ctx.obj
    if not isinstance(obj, RuntimeState):
        raise RuntimeError("CLI runtime state is missing")
    return obj


def _confirm_or_fail(state: RuntimeState, prompt: str) -> bool:
    if state.yes:
        return True
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive:
        raise RuntimeError("Non-interactive mode detected. Re-run with --yes.")
    return Confirm.ask(prompt, default=True)


def _ensure_convert_deps(state: RuntimeState) -> None:
    ensure_dependency_group(
        "convert",
        DependencyPolicy(yes=state.yes, no_install=state.no_install),
        project_root=state.project_root,
    )


@app.callback()
def main(
    ctx: typer.Context,
    project_root: Optional[Path] = typer.Option(None, "--project-root", help="LunaVox project root"),
    yes: bool = typer.Option(False, "--yes", help="Auto confirm install/download prompts"),
    no_install: bool = typer.Option(False, "--no-install", help="Never auto-install missing Python dependencies"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose command output"),
) -> None:
    if ctx.resilient_parsing:
        return
    root = resolve_project_root(project_root)
    log_file = root / "logs" / "latest.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"LunaVox CLI session start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    ctx.obj = RuntimeState(
        project_root=root,
        yes=yes,
        no_install=no_install,
        verbose=verbose,
        latest_log=log_file,
    )


def _convert_internal(state: RuntimeState, model: str, models_dir: str, force: bool) -> None:
    _ensure_convert_deps(state)
    models = Models(state.project_root)
    cfg = models.by_name(model)

    required = [cfg.source / "config.json", cfg.source / "model.safetensors"]
    missing = [p for p in required if not p.exists()]
    if missing:
        allow = _confirm_or_fail(
            state,
            f"Model '{model}' source files are missing. Download from HuggingFace now?",
        )
        if not allow:
            raise RuntimeError("Model download cancelled by user.")
        cfg.source = ModelDownloader.download(model)

    target_dir = Path(models_dir).resolve() if models_dir else cfg.dest.resolve()
    pipeline = ModelConvertPipeline(state.project_root)
    pipeline.convert(cfg, target_dir, force=force)


def _build_internal(
    state: RuntimeState,
    clean: bool,
    j: int,
    verify: bool,
    toolchain: str,
    portable: bool,
) -> None:
    run_build(
        root=state.project_root,
        clean=clean,
        jobs=j,
        verify=verify,
        toolchain=toolchain,
        portable=portable,
        verbose=state.verbose,
    )


@app.command("convert")
def convert_command(
    ctx: typer.Context,
    model: Optional[str] = typer.Option(None, "--model", help="Model variant (id)"),
    models_dir: str = typer.Option("", "--models-dir", help="Override output models directory"),
    force: bool = typer.Option(False, "--force", help="Force reconversion"),
    all_models: bool = typer.Option(False, "--all", help="Convert all available models"),
) -> None:
    """Convert source model weights into LunaVox runtime artifacts."""
    state = _state(ctx)
    
    options = [
        ("base_small", "Qwen3-TTS-12Hz-0.6B-Base"),
        ("custom_small", "Qwen3-TTS-12Hz-0.6B-CustomVoice"),
        ("base", "Qwen3-TTS-12Hz-1.7B-Base"),
        ("custom", "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        ("design", "Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    ]

    selected_keys = []

    if all_models:
        selected_keys = [opt[0] for opt in options]
    elif model:
        selected_keys = [model]
    else:
        # Interactive selection
        console.print("\n[bold cyan]═══ Model Selection ═══[/]")
        for i, (key, display) in enumerate(options, 1):
            console.print(f"  [bold]{i}) {display}[/]")
        
        console.print(f"  [bold]{len(options) + 1}) ALL MODELS[/]")
        console.print("  [bold]0) Cancel[/]")
        
        choice = typer.prompt("\nSelect a model to convert", default="1")
        if choice == "0":
            console.print("[warn]Cancelled.[/]")
            return
        
        try:
            idx = int(choice) - 1
            if idx == len(options): # All models
                selected_keys = [opt[0] for opt in options]
            elif 0 <= idx < len(options):
                selected_keys = [options[idx][0]]
            else:
                raise ValueError()
        except (ValueError, IndexError):
            raise RuntimeError("Invalid selection.")

    for key in selected_keys:
        console.print(Panel(f"Converting Model: [bold]{key}[/]", style="stage", expand=False))
        _convert_internal(state, model=key, models_dir=models_dir, force=force)
    
    console.print("[success]Conversion process completed.[/")


@app.command("pull-model")
def pull_model_command(
    ctx: typer.Context,
    model: str = typer.Option("base_small", "--model", help="Model variant"),
) -> None:
    """Download pre-converted runtime artifacts (GGUF/ONNX) from HuggingFace."""
    state = _state(ctx)
    ModelDownloader.download_converted_model(model, state.project_root)
    console.print(f"[success]Model '{model}' pulled successfully.[/success]")


@app.command("build")
def build_command(
    ctx: typer.Context,
    clean: bool = typer.Option(False, "--clean", help="Clean build directory first"),
    j: int = typer.Option(4, "--j", min=1, help="Parallel build jobs"),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Run post-build verification"),
    toolchain: str = typer.Option("auto", "--toolchain", help="Force toolchain"),
    portable: bool = typer.Option(False, "--portable", help="Bundle system runtime dependencies"),
) -> None:
    state = _state(ctx)
    _build_internal(state, clean=clean, j=j, verify=verify, toolchain=toolchain, portable=portable)
    console.print("[success]Build completed successfully.[/success]")


@app.command("bootstrap")
def bootstrap_command(
    ctx: typer.Context,
    model: str = typer.Option("base_small", "--model", help="Model variant"),
    models_dir: str = typer.Option("", "--models-dir", help="Override output models directory"),
    force: bool = typer.Option(False, "--force", help="Force reconversion"),
    clean: bool = typer.Option(False, "--clean", help="Clean build directory first"),
    j: int = typer.Option(4, "--j", min=1, help="Parallel build jobs"),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Run post-build verification"),
    toolchain: str = typer.Option("auto", "--toolchain", help="Force toolchain"),
    portable: bool = typer.Option(False, "--portable", help="Bundle system runtime dependencies"),
) -> None:
    state = _state(ctx)
    _convert_internal(state, model=model, models_dir=models_dir, force=force)
    _build_internal(state, clean=clean, j=j, verify=verify, toolchain=toolchain, portable=portable)
    console.print("[success]Bootstrap completed successfully.[/success]")


@app.command("download-libs")
def download_libs_command(
    ctx: typer.Context,
    platform_key: Optional[str] = typer.Option(None, "--platform", "-p", help="Target platform (win_cuda, win_vulkan, etc.)"),
    lib: Optional[str] = typer.Option(None, "--lib", help="Download a specific library only (onnx, llama)"),
    backend: Optional[str] = typer.Option(None, "--backend", help="Specific backend for the library"),
) -> None:
    """Download binary dependencies (ONNX Runtime, Llama.cpp) for your platform."""
    state = _state(ctx)
    config = LIBS_CONFIG
    
    # 1. Handle Single Lib/Backend override
    if lib:
        if lib not in config["libraries"]:
            raise RuntimeError(f"Unknown library '{lib}'. Available: {', '.join(config['libraries'].keys())}")
        if not backend:
            # Interactive backend selection for this lib
            backends = config["libraries"][lib]["backends"]
            options = list(backends.keys())
            console.print(f"\nSelect backend for [bold cyan]{lib}[/]:")
            for i, opt in enumerate(options, 1):
                console.print(f"  {i}) {opt}")
            idx = int(typer.prompt("Enter choice", default="1")) - 1
            backend = options[idx]
        
        console.print(Panel(f"Downloading {lib} ({backend})", style="stage", expand=False))
        update_library(lib, backend, str(state.project_root))
        console.print("[success]Library download completed.[/success]")
        return

    # 2. Handle Platform selection
    if not platform_key:
        # Interactive platform selection
        platforms = config["platforms"]
        options = list(platforms.keys())
        
        console.print("\n[bold cyan]═══ Library Download Selection ═══[/]")
        for i, key in enumerate(options, 1):
            console.print(f"  [bold]{i}) {platforms[key]['display_name']}[/]")
        
        console.print("  [bold]0) Cancel[/]")
        
        choice = typer.prompt("\nSelect your target platform", default="1")
        if choice == "0":
            console.print("[warn]Cancelled.[/]")
            return
        
        try:
            idx = int(choice) - 1
            platform_key = options[idx]
        except (ValueError, IndexError):
            raise RuntimeError("Invalid selection.")

    console.print(f"[stage]Selected platform: {platform_key}[/]")
    download_platform_libs(platform_key, str(state.project_root))
    console.print("[success]All platform libraries updated successfully.[/success]")


@app.command("doctor")
def doctor_command(ctx: typer.Context) -> None:
    state = _state(ctx)

    table = Table(title="LunaVox Doctor", show_header=True, header_style="bold cyan")
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Details")

    root_ok = all(
        [
            (state.project_root / "CMakeLists.txt").exists(),
            (state.project_root / "src").exists(),
            (state.project_root / "lib").exists(),
            (state.project_root / "models").exists(),
        ]
    )
    table.add_row(
        "Project Root",
        "OK" if root_ok else "FAIL",
        str(state.project_root),
    )

    cmake = shutil.which("cmake")
    table.add_row("CMake", "OK" if cmake else "WARN", cmake or "Not found in PATH")

    llama_dir = state.project_root / "lib" / "llama"
    onnx_header = state.project_root / "lib" / "onnx" / "include" / "onnxruntime_cxx_api.h"
    table.add_row("Llama Runtime", "OK" if llama_dir.exists() else "WARN", str(llama_dir))
    table.add_row("ONNX SDK Header", "OK" if onnx_header.exists() else "WARN", str(onnx_header))

    missing_convert = missing_modules(DEPENDENCY_GROUPS["convert"])
    if missing_convert:
        details = ", ".join(f"{m[0]}({m[1]})" for m in missing_convert)
        table.add_row("Convert Dependencies", "WARN", details)
    else:
        table.add_row("Convert Dependencies", "OK", "All required modules available")

    table.add_row("Log File", "INFO", str(state.latest_log))
    console.print(table)

    if missing_convert:
        install_cmd = f'{sys.executable} -m pip install "lunavox[convert]"'
        console.print(
            f"Missing convert deps. Install with: {install_cmd}",
            style="warning",
            markup=False,
        )


def run() -> int:
    try:
        app()
        return 0
    except Exception as err:
        console.print(str(err), style="error", markup=False)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
