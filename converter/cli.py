"""
Command-line interface for LunaVox Model Converter.

Usage:
    python -m converter --ckpt path/to/model.ckpt --pth path/to/model.pth --output output_dir [--format fp16]
"""
import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch GPT-SoVITS models to ONNX format for LunaVox TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert to FP16 format (default)
    python -m converter --ckpt s1bert.ckpt --pth s2G.pth --output ./output
    
    # Force specific model version
    python -m converter --ckpt s1bert.ckpt --pth s2G.pth --output ./output --version v2ProPlus
        """
    )
    
    parser.add_argument("--ckpt", required=True, help="Path to T2S model (.ckpt)")
    parser.add_argument("--pth", required=True, help="Path to VITS model (.pth)")
    parser.add_argument("--output", "-o", default="pretrained", help="Output directory (default: pretrained)")
    parser.add_argument("--format", "-f", choices=["fp16"], default="fp16",
                       help="Output format: fp16 (default, hybrid for CPU/GPU)")
    parser.add_argument("--version", "-v", choices=["v2", "v2Pro", "v2ProPlus"],
                       help="Force model version (auto-detected if not specified)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Run conversion
    from . import convert
    
    try:
        convert(
            ckpt_path=args.ckpt,
            pth_path=args.pth,
            output_dir=args.output,
            format=args.format,
            model_version=args.version,
        )
    except Exception as e:
        logging.error(f"❌ Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
