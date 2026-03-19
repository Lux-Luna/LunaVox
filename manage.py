#!/usr/bin/env python3
"""Unified CLI entry point for repository Python tooling."""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "tools"

COMMANDS = {
    "setup": TOOLS / "setup" / "setup_pipeline.py",
    # Componentized 5-file conversion pipeline (local-only by default, see main()).
    "convert": TOOLS / "setup" / "setup_pipeline.py",
    "convert-tok": TOOLS / "conversion" / "convert_tokenizer_to_gguf.py",
    "export-embeddings": TOOLS / "conversion" / "export_embeddings.py",
    "inspect": TOOLS / "conversion" / "inspect_models.py",
    "build": TOOLS / "build_manager.py",
}

def usage() -> None:
    print("Qwen3-TTS Management Tool")
    print("Usage: python manage.py <command> [args]")
    print("\nAvailable commands:")
    for cmd in sorted(COMMANDS.keys()):
        print(f"  {cmd:<15} - Run {COMMANDS[cmd].name}")

def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        usage()
        return 0

    cmd_name = sys.argv[1]
    if cmd_name not in COMMANDS:
        print(f"Error: Unknown command '{cmd_name}'")
        usage()
        return 1

    script_path = COMMANDS[cmd_name]
    if not script_path.exists():
        print(f"Error: command target not found: {script_path}")
        return 1

    if script_path.suffix != ".py":
        print(f"Error: Unsupported script type {script_path.suffix}")
        return 1

    cmd = [sys.executable, str(script_path)]

    extra_args = list(sys.argv[2:])
    if cmd_name == "convert" and "--skip-download" not in extra_args:
        extra_args.insert(0, "--skip-download")

    cmd.extend(extra_args)

    try:
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    except subprocess.CalledProcessError as e:
        return e.returncode
    except KeyboardInterrupt:
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
