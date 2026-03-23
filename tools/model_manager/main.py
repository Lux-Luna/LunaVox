#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from model_config import Models
from tools.model_manager.pipeline import ModelSetupPipeline
from tools.model_manager.downloader import ModelDownloader

def main():
    p = argparse.ArgumentParser(description="LunaVox Model Manager")
    sub = p.add_subparsers(dest="command")

    # Download command
    p_dl = sub.add_parser("download", help="Download model source from HF Hub")
    p_dl.add_argument("--model", choices=[m.name for m in Models.ALL] + ["all"], default="base_small")
    
    # Setup command
    p_setup = sub.add_parser("setup", help="Convert models into runtime artifacts")
    p_setup.add_argument("--model", choices=[m.name for m in Models.ALL], default="base_small")
    p_setup.add_argument("--models-dir", default="", help="Override target models directory")
    p_setup.add_argument("--skip-convert", action="store_true")
    p_setup.add_argument("--force", action="store_true")
    p_setup.add_argument("--timeout-sec", type=int, default=170)
    p_setup.add_argument("--enable-quant", action="store_true")

    args = p.parse_args()

    if args.command == "download":
        if args.model == "all":
            ModelDownloader.download_all()
        else:
            ModelDownloader.download(args.model)
    elif args.command == "setup":
        cfg = Models.by_name(args.model)
        models_dir = Path(args.models_dir).resolve() if args.models_dir else cfg.dest.resolve()
        pipeline = ModelSetupPipeline(root)
        pipeline.setup(cfg, models_dir, skip_convert=args.skip_convert, force=args.force, timeout_sec=args.timeout_sec, enable_quant=args.enable_quant)
    else:
        p.print_help()

if __name__ == "__main__":
    main()
