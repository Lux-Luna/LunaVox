from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

from rich.panel import Panel

from lunavox.core.ui import console
from . import get_builder_class, get_resolver_class
from .context import BuildContext


def run_build(
    *,
    root: Path,
    clean: bool,
    jobs: int,
    toolchain: str,
    verbose: bool,
    platform_key: str | None = None,
) -> int:
    env = os.environ.copy()
    env["LUNAVOX_PROJECT_ROOT"] = str(root)
    if toolchain != "auto":
        env["LUNAVOX_TOOLCHAIN"] = toolchain
    if verbose:
        env["LUNAVOX_BUILD_VERBOSE"] = "1"

    py_prefix = Path(sys.prefix).resolve()
    if platform.system() == "Windows" and (py_prefix / "python.exe").exists():
        candidates = [
            py_prefix,
            py_prefix / "Library/bin",
            py_prefix / "Library/mingw-w64/bin",
            py_prefix / "Scripts",
            py_prefix / "bin",
        ]
        valid_dirs = [str(d) for d in candidates if d.exists()]
        if valid_dirs:
            env["PATH"] = ";".join(valid_dirs + [env.get("PATH", "")])

    tmp_dir = root / "_tmp_build"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env["TMP"] = env["TEMP"] = str(tmp_dir)

    ctx = BuildContext(root, env)
    resolver_cls = get_resolver_class()
    resolver = resolver_cls(ctx)
    builder_cls = get_builder_class()
    builder = builder_cls(ctx)

    console.print(
        Panel(
            f"LunaVox C++ Build Manager - [bold cyan]{platform.system()}[/bold cyan]",
            border_style="cyan",
        )
    )

    builder.build(
        resolver,
        clean=clean,
        parallel=jobs,
        platform_key=platform_key,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LunaVox Build Manager")
    parser.add_argument("--clean", action="store_true", help="Clean build directory first")
    parser.add_argument("--j", type=int, default=4, help="Parallel build jobs")
    parser.add_argument(
        "--toolchain",
        choices=["auto", "msvc", "mingw", "clang", "gcc"],
        default="auto",
        help="Force a toolchain",
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get("LUNAVOX_PROJECT_ROOT", ""),
        help="Project root path",
    )
    parser.add_argument("--verbose", action="store_true", help="Show real-time build output")
    parser.add_argument("--platform", help="Target platform/backend key (e.g. win_cuda12)")
    args = parser.parse_args()

    root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parents[3]
    )
    try:
        return run_build(
            root=root,
            clean=bool(args.clean),
            jobs=int(args.j),
            toolchain=str(args.toolchain),
            verbose=bool(args.verbose),
            platform_key=args.platform,
        )
    except Exception as err:
        console.print(f"\n[bold red][ERROR][/bold red] Build failed: {err}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

