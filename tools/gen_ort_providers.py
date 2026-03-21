#!/usr/bin/env python3
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = ROOT / "src" / "generated"

def main():
    providers_env = os.environ.get("QWEN3_TTS_ORT_PROVIDERS", "CPU").upper()
    active_providers = [p.strip() for p in providers_env.split(",") if p.strip()]

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    header_path = GENERATED_DIR / "ort_provider_injector.h"
    cpp_path = GENERATED_DIR / "ort_provider_injector.cpp"

    header_content = """#pragma once
#include <onnxruntime_cxx_api.h>

void apply_decoder_providers(Ort::SessionOptions& opts);
"""

    cpp_content = """#include "ort_provider_injector.h"

void apply_decoder_providers(Ort::SessionOptions& opts) {
"""
    for p in active_providers:
        if p == "CUDA":
            cpp_content += "    // Attempt to use CUDA Provider\n"
            cpp_content += "    OrtCUDAProviderOptions cuda_opts;\n"
            cpp_content += "    opts.AppendExecutionProvider_CUDA(cuda_opts);\n"
        elif p == "DML":
            cpp_content += "    // Attempt to use DML Provider\n"
            cpp_content += "    opts.AppendExecutionProvider_DML(0);\n"
        elif p == "COREML":
            cpp_content += "    // Attempt to use CoreML Provider\n"
            cpp_content += "    opts.AppendExecutionProvider_CoreML(0);\n"
        # CPU is default, normally we don't need to append it.
    
    cpp_content += "}\n"

    header_path.write_text(header_content, encoding="utf-8")
    cpp_path.write_text(cpp_content, encoding="utf-8")

if __name__ == "__main__":
    main()
