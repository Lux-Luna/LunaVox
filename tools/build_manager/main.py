from __future__ import annotations
import os
import sys
import platform
import argparse
from pathlib import Path
from . import get_resolver_class
from .context import BuildContext
from .windows import WindowsBuilder
from .base import Builder

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.status import Status
    console = Console()
except ImportError:
    # Fallback to plain print
    class DummyConsole:
        def print(self, *args, **kwargs): print(*args, **kwargs)
    console = DummyConsole()

def main():
    parser = argparse.ArgumentParser(description="LunaVox Build Manager")
    parser.add_argument("--clean", action="store_true", help="Clean build directory first")
    parser.add_argument("--j", type=int, default=4, help="Parallel build jobs")
    parser.add_argument("--timeout-sec", type=int, default=200, help="Stage timeout")
    parser.add_argument("--verify", action="store_true", help="Verify build with --help test")
    parser.add_argument("--toolchain", choices=["auto", "msvc", "mingw", "clang", "gcc"], default="auto", help="Force a toolchain")
    args = parser.parse_args()

    # Environment setup
    env = os.environ.copy()
    if args.toolchain != "auto":
        env["LUNAVOX_TOOLCHAIN"] = args.toolchain
    py_prefix = Path(sys.prefix).resolve()
    
    if platform.system() == "Windows" and (py_prefix / "python.exe").exists():
        candidates = [py_prefix, py_prefix / "Library/bin", py_prefix / "Library/mingw-w64/bin", py_prefix / "Scripts", py_prefix / "bin"]
        valid_dirs = [str(d) for d in candidates if d.exists()]
        if valid_dirs:
            env["PATH"] = ";".join(valid_dirs + [env.get("PATH", "")])
    
    root = Path(__file__).resolve().parents[2]
    tmp_dir = root / "_tmp_build"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env["TMP"] = env["TEMP"] = str(tmp_dir)

    ctx = BuildContext(root, env)
    
    # Factory selection
    resolver_cls = get_resolver_class()
    resolver = resolver_cls(ctx)
    
    builder_cls = WindowsBuilder if platform.system() == "Windows" else Builder
    builder = builder_cls(ctx, timeout_sec=args.timeout_sec)
    
    try:
        # We wrap in Panel for professional look if rich is available
        if hasattr(console, "print"):
            console.print(Panel(f"LunaVox C++ Build Manager - [bold cyan]{platform.system()}[/bold cyan]", border_style="cyan"))
        
        builder.build(resolver, clean=args.clean, parallel=args.j, verify=args.verify)
        
    except Exception as e:
        if hasattr(console, "print"):
            console.print(f"\n[bold red][ERROR][/bold red] Build failed: {e}")
        else:
            print(f"\n[ERROR] Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
