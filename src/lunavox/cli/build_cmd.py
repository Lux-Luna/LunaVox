"""``lunavox build`` and ``lunavox build libs`` — the native toolchain group."""

from __future__ import annotations

from typing import Optional

import typer
from rich.panel import Panel

from lunavox.build.lib_downloader import LIBS_CONFIG, download_platform_libs, update_library
from lunavox.build.main import run_build
from lunavox.core.ui import console

from ._common import RuntimeState, state

app = typer.Typer(
    no_args_is_help=False,
    invoke_without_command=True,
    help="Build the C++ engine or pull native runtime libraries.",
)


@app.callback()
def build_root(
    ctx: typer.Context,
    clean: bool = typer.Option(False, "--clean", help="Clean build directory first"),
    j: int = typer.Option(4, "--j", min=1, help="Parallel build jobs"),
    toolchain: str = typer.Option("auto", "--toolchain", help="Force toolchain"),
) -> None:
    """Build the LunaVox C++ inference engine.

    Running ``lunavox build`` with no subcommand builds the engine;
    ``lunavox build libs`` pulls binary dependencies instead.
    """
    if ctx.invoked_subcommand is not None:
        return
    st = state(ctx)
    build(st, clean=clean, j=j, toolchain=toolchain)
    console.print("[success]Build completed successfully.[/success]")


def build(st: RuntimeState, clean: bool, j: int, toolchain: str) -> None:
    """Internal entry used by both ``lunavox build`` and ``bootstrap``."""
    run_build(
        root=st.project_root,
        clean=clean,
        jobs=j,
        toolchain=toolchain,
        verbose=st.verbose,
    )


@app.command("libs")
def libs_cmd(
    ctx: typer.Context,
    platform_key: Optional[str] = typer.Option(
        None, "--platform", "-p", help="Target platform (win_cuda, win_vulkan, etc.)"
    ),
    lib: Optional[str] = typer.Option(
        None, "--lib", help="Download a specific library only (onnx, llama)"
    ),
    backend: Optional[str] = typer.Option(
        None, "--backend", help="Specific backend for the library"
    ),
) -> None:
    """Download binary dependencies (ONNX Runtime, Llama.cpp) for your platform."""
    st = state(ctx)
    libs(st, platform_key, lib, backend)


def libs(
    st: RuntimeState,
    platform_key: Optional[str] = None,
    lib: Optional[str] = None,
    backend: Optional[str] = None,
) -> None:
    """Internal entry used by both ``lunavox build libs`` and ``bootstrap``."""
    config = LIBS_CONFIG

    if lib:
        if lib not in config["libraries"]:
            raise RuntimeError(
                f"Unknown library '{lib}'. Available: {', '.join(config['libraries'].keys())}"
            )
        if not backend:
            backends = config["libraries"][lib]["backends"]
            options = list(backends.keys())
            console.print(f"\nSelect backend for [bold cyan]{lib}[/]:")
            for i, opt in enumerate(options, 1):
                console.print(f"  {i}) {opt}")
            idx = int(typer.prompt("Enter choice", default="1")) - 1
            backend = options[idx]

        console.print(Panel(f"Downloading {lib} ({backend})", style="stage", expand=False))
        update_library(lib, backend, str(st.project_root))
        console.print("[success]Library download completed.[/success]")
        return

    if not platform_key:
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
        except (ValueError, IndexError) as err:
            raise RuntimeError("Invalid selection.") from err

    console.print(f"[stage]Selected platform: {platform_key}[/]")
    download_platform_libs(platform_key, str(st.project_root))
    console.print("[success]All platform libraries updated successfully.[/success]")
