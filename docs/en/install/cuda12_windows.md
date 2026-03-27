# LunaVox Windows CUDA 12 Dependency Guide

This document explains dependencies for LunaVox using CUDA 12 and cuDNN 9 on Windows, aiming to provide a clear minimum dependency set.

---

## 1. Verified Environment

Based on test results with the following runtime combination:

- **ONNX Runtime**: `1.24.4`
- **CUDA Runtime family**: `12.4.x`
- **cuDNN**: `9.1.1.17`
- **Platform**: Windows x64
- **GPU**: NVIDIA RTX 3090 (Compute Capability 8.6)

Key package versions in the environment:
- `cuda-toolkit 12.4.1`
- `cudnn 9.1.1.17`
- `libcublas 12.4.5.8`
- `libcufft 11.2.1.3`
- `libcurand 10.3.5.147`

---

## 2. cuDNN 9 Dependency Characteristics

**Important**: cuDNN 9 on Windows is no longer a single file. You cannot just put `cudnn64_9.dll` in the folder.

`cudnn64_9.dll` acts only as an entry point. The full cuDNN 9 DLL family includes:

- `cudnn64_9.dll`
- `cudnn_adv64_9.dll`
- `cudnn_cnn64_9.dll`
- `cudnn_engines_precompiled64_9.dll`
- `cudnn_engines_runtime_compiled64_9.dll`
- `cudnn_graph64_9.dll`
- `cudnn_heuristic64_9.dll`
- `cudnn_ops64_9.dll`

---

## 3. Minimum Dependency Set (DLL List)

To ensure LunaVox components (GGML/llama.cpp and ONNX Runtime) can successfully enable CUDA acceleration:

### 3.1 Core Runtime DLLs (LunaVox specific)

- `onnxruntime.dll` (GPU version)
- `onnxruntime_providers_cuda.dll`
- `onnxruntime_providers_shared.dll`
- `ggml-cuda.dll`

### 3.2 CUDA 12 / cuDNN 9 Support DLLs

- `cudart64_12.dll`
- `cublas64_12.dll`
- `cublasLt64_12.dll`
- `cufft64_11.dll`
- `curand64_10.dll`
- **cuDNN Family (8 DLLs listed in Section 2)**

### 3.3 Windows / MSVC Runtime

LunaVox depends on standard C/C++ runtimes:
- `vcruntime140.dll`
- `vcruntime140_1.dll`
- `msvcp140.dll`

---

## 4. Preparation

### 4.1 Using Conda Environment (Recommended)

Run in a conda environment with `cuda-toolkit 12.x` and `cudnn 9.x` installed. Ensure `Library\bin` is in `PATH`.

### 4.2 Portable Deployment (Outside Conda)

If packaging LunaVox as a standalone software, ensure all mentioned DLLs (~20 files) are present in the same directory as the executable.

**Never copy just `cudnn64_9.dll` and ignore the other `cudnn_*.dll` components.**

---

## 5. Dependency Checklist

If CUDA fails to enable, check these files:

1.  **Core**: `onnxruntime.dll`, `onnxruntime_providers_cuda.dll`, `ggml-cuda.dll`
2.  **CUDA Base**: `cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`
3.  **cuDNN Family**: Ensure all 8 `cudnn_*.dll` files are present.
4.  **Math**: `cufft64_11.dll`, `curand64_10.dll`
5.  **System**: `vcruntime140.dll`, `vcruntime140_1.dll`, `msvcp140.dll`

---

## 6. Troubleshooting

1.  Check if `metadata.json` has `onnx.provider` as `CUDAExecutionProvider`.
2.  Ensure GPU version of ONNX Runtime is loaded.
3.  Check version consistency between `onnxruntime_providers_cuda.dll` and `onnxruntime.dll`.
4.  Run TTS with `--verbose` to observe which provider is actually enabled.
