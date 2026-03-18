#!/usr/bin/env python3
"""One-click pipeline: convert HF model → quantized lunavox GGUFs.

Usage:
    python tools/conversion/pipeline.py --model-dir models/Qwen3-TTS-12Hz-0.6B-Base

Steps:
    1. Export Talker + Predictor as F16 GGUFs (standard names for llama-quantize)
    2. Quantize with llama-quantize.exe (Q5_K_M / Q8_0)
    3. Merge quantized components + F16 Decoder → final lunavox GGUFs
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def run_step(step_name: str, args: list[str]):
    """Run a pipeline step."""
    print(f"\n{'='*60}")
    print(f"  {step_name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable] + args, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"\nERROR: {step_name} failed with return code {result.returncode}")
        sys.exit(1)
    elapsed = time.time() - t0
    print(f"  [{step_name}] completed in {elapsed:.1f}s")


def validate_output(output_dir: Path):
    """Basic validation of output GGUF files."""
    main_gguf = output_dir / "qwen3-tts-0.6B-base.gguf"
    aux_gguf = output_dir / "qwen3-tts-aux-f16.gguf"

    print(f"\n{'='*60}")
    print(f"  Validation")
    print(f"{'='*60}")

    errors = []
    if not main_gguf.exists():
        errors.append(f"Main model not found: {main_gguf}")
    else:
        size_mb = main_gguf.stat().st_size / 1024 / 1024
        print(f"  Main model: {main_gguf.name} ({size_mb:.1f} MB)")
        if size_mb < 100:
            errors.append(f"Main model suspiciously small: {size_mb:.1f} MB")

    if not aux_gguf.exists():
        errors.append(f"Aux model not found: {aux_gguf}")
    else:
        size_mb = aux_gguf.stat().st_size / 1024 / 1024
        print(f"  Aux model:  {aux_gguf.name} ({size_mb:.1f} MB)")

    # Verify GGUF structure with gguf-py
    try:
        from gguf import GGUFReader
        reader = GGUFReader(str(main_gguf))
        n_tensors = len(reader.tensors)
        print(f"  Tensor count: {n_tensors}")

        # Check for expected tensor prefixes
        prefixes = set()
        for tensor in reader.tensors:
            prefix = tensor.name.split(".")[0]
            prefixes.add(prefix)
        print(f"  Tensor prefixes: {sorted(prefixes)}")

        expected_prefixes = {"talker", "code_pred", "tok_dec"}
        missing = expected_prefixes - prefixes
        if missing:
            errors.append(f"Missing tensor prefixes: {missing}")

        # Check expected metadata
        kv_keys = {field.name for field in reader.fields}
        expected_keys = [
            "qwen3-tts.talker.block_count",
            "qwen3-tts.code_pred.layer_count",
            "tokenizer.ggml.tokens",
        ]
        for key in expected_keys:
            if key not in kv_keys:
                errors.append(f"Missing metadata key: {key}")
            else:
                print(f"  Metadata OK: {key}")

    except Exception as e:
        errors.append(f"GGUF validation error: {e}")

    if errors:
        print(f"\n  VALIDATION ERRORS:")
        for err in errors:
            print(f"    - {err}")
        return False

    print(f"\n  ✓ All validations passed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert HF model to quantized lunavox GGUF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/conversion/pipeline.py
    python tools/conversion/pipeline.py --model-dir models/Qwen3-TTS-12Hz-0.6B-Base
    python tools/conversion/pipeline.py --talker-quant Q5_K --predictor-quant Q8_0
""")
    parser.add_argument("--model-dir", type=str,
                        default=str(REPO_ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-Base"),
                        help="HF model directory")
    parser.add_argument("--output-dir", type=str,
                        default=str(REPO_ROOT / "models"),
                        help="Output directory for final GGUFs")
    parser.add_argument("--talker-quant", type=str, default="Q5_K_M",
                        help="Talker quantization type (default: Q5_K_M)")
    parser.add_argument("--predictor-quant", type=str, default="Q8_0",
                        help="Predictor quantization type (default: Q8_0)")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Don't clean up intermediate files")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip step 1 (export F16 GGUFs)")
    parser.add_argument("--skip-quantize", action="store_true",
                        help="Skip step 2 (llama-quantize)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    tmp_dir = output_dir / "tmp"

    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Pipeline Configuration:")
    print(f"  HF Model:   {model_dir}")
    print(f"  Output:      {output_dir}")
    print(f"  Talker:      {args.talker_quant}")
    print(f"  Predictor:   {args.predictor_quant}")

    t_start = time.time()

    # Step 1: Export F16 GGUFs
    if not args.skip_export:
        run_step("Step 1: Export F16 GGUFs", [
            str(SCRIPT_DIR / "01_export_components.py"),
            "--model-dir", str(model_dir),
            "--output-dir", str(tmp_dir),
        ])

    # Step 2: Quantize
    if not args.skip_quantize:
        run_step("Step 2: Quantize with llama-quantize.exe", [
            str(SCRIPT_DIR / "02_quantize.py"),
            "--tmp-dir", str(tmp_dir),
            "--repo-root", str(REPO_ROOT),
            "--talker-quant", args.talker_quant,
            "--predictor-quant", args.predictor_quant,
        ])

    # Step 3: Merge
    talker_quant_tag = args.talker_quant.lower().replace("_", "_")
    predictor_quant_tag = args.predictor_quant.lower().replace("_", "_")
    run_step("Step 3: Merge into lunavox GGUF", [
        str(SCRIPT_DIR / "03_merge.py"),
        "--model-dir", str(model_dir),
        "--tmp-dir", str(tmp_dir),
        "--output-dir", str(output_dir),
        "--talker-quant", talker_quant_tag,
        "--predictor-quant", predictor_quant_tag,
    ])

    # Validate
    ok = validate_output(output_dir)

    # Cleanup
    if not args.keep_tmp and tmp_dir.exists():
        print(f"\nCleaning up temporary files: {tmp_dir}")
        shutil.rmtree(tmp_dir)

    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Pipeline {'completed' if ok else 'FAILED'} in {t_total:.1f}s")
    print(f"{'='*60}")

    if ok:
        print(f"\nOutput files:")
        print(f"  {output_dir / 'qwen3-tts-0.6B-base.gguf'}")
        print(f"  {output_dir / 'qwen3-tts-aux-f16.gguf'}")
        print(f"\nTest with:")
        print(f"  .\\build-cpu\\qwen3-tts-cli.exe -m models -t \"Hello world\" -o test.wav")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
