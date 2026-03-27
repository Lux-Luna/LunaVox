# LunaVox Windows CUDA 13 Dependency Guide

This document explains dependencies for LunaVox using CUDA 13 and cuDNN 9 on Windows, providing a minimum dependency set guide.

---

## 1. Core Runtime Combination (Verified)

Tested successfully on Windows x64:

- **ONNX Runtime**: `1.24.4` (GPU, built for CUDA 13)
- **Llama.cpp**: `b8470` (built for CUDA 13.1)
- **CUDA Toolkit**: `13.2.0` (Recommended)
- **cuDNN**: `9.20.0` (Compatible with CUDA 13.x)

---

## 2. cuDNN 9 Dependency Characteristics

**Important**: On Windows, cuDNN 9 is not a single file. Even with CUDA 13, it requires the full set of `cudnn_*64_9.dll`.

The entry point `cudnn64_9.dll` must be accompanied by:

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

### 3.1 Core Runtime DLLs (LunaVox specific)

Use binaries compiled for CUDA 13:

- `onnxruntime.dll`
- `onnxruntime_providers_cuda.dll` (References `cublasLt64_13.dll`, not `_12.dll`)
- `onnxruntime_providers_shared.dll`
- `ggml-cuda.dll`

### 3.2 CUDA 13 / cuDNN 9 Support DLLs

- `cudart64_13.dll`
- `cublas64_13.dll`
- `cublasLt64_13.dll`
- `cufft64_12.dll` (Note: cuFFT version jumped to 12 in CUDA 13)
- `curand64_10.dll` (Note: cuRAND still uses version 10)
- **cuDNN Family (8 DLLs listed in Section 2)**

### 3.3 Windows / MSVC Runtime

- `vcruntime140.dll`
- `vcruntime140_1.dll`
- `msvcp140.dll`

---

## 4. Preparation

### 4.1 Environment (Conda)

Recommended for managing CUDA 13 components:
```bash
conda create -n cuda13 python=3.13 -y
conda install -n cuda13 cuda-toolkit=13.2 cudnn -c nvidia -y
```

### 4.2 Binary Updates

If your `build` directory contains old binaries (e.g., error finding `cublasLt64_12.dll`), update via `lib_downloader.py`:
```bash
# Download CUDA 13 libraries
python src/lunavox/build/lib_downloader.py onnx win_cuda13 build
python src/lunavox/build/lib_downloader.py llama win_cuda13 build

# Move DLLs to execution directory
cp build/lib/onnx/lib/*.dll build/
cp build/lib/llama/*.dll build/
```

---

## 5. Dependency Checklist

1.  **Binary Version**: Ensure `onnxruntime_providers_cuda.dll` doesn't try to load `cublasLt64_12.dll`.
2.  **CUDA 13**: `cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll`.
3.  **cuDNN**: All 8 `cudnn_*.dll` files present.
4.  **Math**: `cufft64_12.dll`, `curand64_10.dll`.

---

## 6. Troubleshooting

1.  **Error 126**: Usually means a DLL is looking for CUDA 12 version (e.g., `cublasLt64_12.dll`). Update binaries.
2.  **Provider Check**: Ensure `decoder` is `CUDAExecutionProvider`.
3.  **Llama.cpp Warning**: If it falls back to CPU, ensure `ggml-cuda.dll` is the CUDA 13 version.
