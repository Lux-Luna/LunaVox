#!/usr/bin/env python3
"""Quantize F16 GGUFs using llama-quantize.exe.

Applies Q5_K_M to the Talker and Q8_0 to the Predictor.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_quantize_exe(repo_root: Path) -> Path:
    """Find llama-quantize.exe in the bin directory."""
    exe = repo_root / "bin" / "llama-quantize.exe"
    if exe.exists():
        return exe
    # Fallback: check PATH
    import shutil
    found = shutil.which("llama-quantize")
    if found:
        return Path(found)
    print(f"ERROR: llama-quantize.exe not found at {exe}", file=sys.stderr)
    sys.exit(1)


def quantize_gguf(exe: Path, input_path: Path, output_path: Path, quant_type: str):
    """Run llama-quantize.exe on a GGUF file."""
    if output_path.exists():
        print(f"  [skip] Already exists: {output_path}")
        return

    cmd = [str(exe), str(input_path), str(output_path), quant_type]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: llama-quantize failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    out_size = output_path.stat().st_size / 1024 / 1024
    print(f"  Done: {output_path} ({out_size:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Quantize F16 GGUFs with llama-quantize.exe")
    parser.add_argument("--tmp-dir", type=str, required=True,
                        help="Directory containing the F16 GGUFs (from step 01)")
    parser.add_argument("--repo-root", type=str, default=None,
                        help="Repository root (to find bin/llama-quantize.exe)")
    parser.add_argument("--talker-quant", type=str, default="Q5_K_M",
                        help="Quantization type for Talker (default: Q5_K_M)")
    parser.add_argument("--predictor-quant", type=str, default="Q8_0",
                        help="Quantization type for Predictor (default: Q8_0)")
    args = parser.parse_args()

    tmp_dir = Path(args.tmp_dir)
    repo_root = Path(args.repo_root) if args.repo_root else tmp_dir.parent.parent

    exe = find_quantize_exe(repo_root)
    print(f"Using llama-quantize: {exe}")

    talker_f16 = tmp_dir / "talker-f16.gguf"
    predictor_f16 = tmp_dir / "predictor-f16.gguf"

    if not talker_f16.exists():
        print(f"ERROR: {talker_f16} not found. Run 01_export_components.py first.", file=sys.stderr)
        sys.exit(1)
    if not predictor_f16.exists():
        print(f"ERROR: {predictor_f16} not found. Run 01_export_components.py first.", file=sys.stderr)
        sys.exit(1)

    # Quantize
    talker_quant_name = args.talker_quant.lower().replace("_", "_")
    predictor_quant_name = args.predictor_quant.lower().replace("_", "_")

    print(f"\n[Step 1/2] Quantizing Talker to {args.talker_quant}...")
    quantize_gguf(exe, talker_f16, tmp_dir / f"talker-{talker_quant_name}.gguf", args.talker_quant)

    print(f"\n[Step 2/2] Quantizing Predictor to {args.predictor_quant}...")
    quantize_gguf(exe, predictor_f16, tmp_dir / f"predictor-{predictor_quant_name}.gguf", args.predictor_quant)

    print("\n[done] Quantization complete.")


if __name__ == "__main__":
    main()
