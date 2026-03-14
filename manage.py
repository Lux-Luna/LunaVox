#!/usr/bin/env python3
"""
Qwen3-TTS Unified CLI Entry Point
Usage: python manage.py <command> [args]
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "tools"

COMMANDS = {
    "setup": TOOLS / "setup" / "setup_pipeline.py",
    "convert": TOOLS / "conversion" / "convert_tts_to_gguf.py",
    "convert-tok": TOOLS / "conversion" / "convert_tokenizer_to_gguf.py",
    "convert-coreml": TOOLS / "conversion" / "convert_code_predictor_to_coreml.py",
    "inspect": TOOLS / "conversion" / "inspect_models.py",
    "test-e2e": TOOLS / "test" / "compare_e2e.py",
    "test-all": TOOLS / "test" / "run_all_tests.sh",
    "verify-tok": TOOLS / "test" / "verify_tokenizer.py",
    "verify-enc": TOOLS / "test" / "verify_encoder.py",
    "bench": TOOLS / "bench" / "benchmark.py",
    "debug-dec": TOOLS / "debug" / "debug_decoder.py",
    "debug-enc": TOOLS / "debug" / "debug_speaker_encoder.py",
    "gen-ref": TOOLS / "debug" / "generate_deterministic_reference.py",
}

def usage():
    print("Qwen3-TTS Management Tool")
    print("Usage: python manage.py <command> [args]")
    print("\nAvailable commands:")
    for cmd in sorted(COMMANDS.keys()):
        print(f"  {cmd:<15} - Run {COMMANDS[cmd].name}")

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        usage()
        return

    cmd_name = sys.argv[1]
    if cmd_name not in COMMANDS:
        print(f"Error: Unknown command '{cmd_name}'")
        usage()
        sys.exit(1)

    script_path = COMMANDS[cmd_name]
    cmd = []
    
    if script_path.suffix == ".py":
        cmd = [sys.executable, str(script_path)]
    elif script_path.suffix == ".sh":
        cmd = ["bash", str(script_path)]
    else:
        print(f"Error: Unsupported script type {script_path.suffix}")
        sys.exit(1)

    cmd.extend(sys.argv[2:])
    
    try:
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(1)

if __name__ == "__main__":
    main()
