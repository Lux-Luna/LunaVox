# LunaVox Windows CUDA Dependency Guide

Covers both **CUDA 12** and **CUDA 13** runtime sets on Windows x64. Pick the column that matches the binaries you downloaded with `lunavox download-libs --platform win_cuda12` (or `win_cuda13`).

## 1. Verified Combinations

| Component | CUDA 12 | CUDA 13 |
| :--- | :--- | :--- |
| ONNX Runtime | 1.24.4 (GPU) | 1.24.4 (GPU, CUDA 13 build) |
| llama.cpp | — | b8470 (CUDA 13.1) |
| CUDA Toolkit | 12.4.x | 13.2.0 |
| cuDNN | 9.1.1 | 9.20.0 |
| GPU under test | RTX 3090 (CC 8.6) | RTX 3090 |

## 2. cuDNN 9 Is Not a Single File

On Windows, `cudnn64_9.dll` is only an entry shim. The full set of eight DLLs **must** be present, regardless of CUDA version:

```
cudnn64_9.dll
cudnn_adv64_9.dll
cudnn_cnn64_9.dll
cudnn_engines_precompiled64_9.dll
cudnn_engines_runtime_compiled64_9.dll
cudnn_graph64_9.dll
cudnn_heuristic64_9.dll
cudnn_ops64_9.dll
```

Copying only `cudnn64_9.dll` is the most common cause of "CUDA acceleration silently disabled".

## 3. Minimum DLL Set

### Core (LunaVox-built)

- `onnxruntime.dll` (GPU build)
- `onnxruntime_providers_cuda.dll`
- `onnxruntime_providers_shared.dll`
- `ggml-cuda.dll`

### CUDA / Math

| | CUDA 12 | CUDA 13 |
| :--- | :--- | :--- |
| Runtime | `cudart64_12.dll` | `cudart64_13.dll` |
| cuBLAS | `cublas64_12.dll`, `cublasLt64_12.dll` | `cublas64_13.dll`, `cublasLt64_13.dll` |
| cuFFT | `cufft64_11.dll` | `cufft64_12.dll` |
| cuRAND | `curand64_10.dll` | `curand64_10.dll` |
| cuDNN | full 8-DLL set (Section 2) | full 8-DLL set (Section 2) |

### MSVC runtime

`vcruntime140.dll`, `vcruntime140_1.dll`, `msvcp140.dll`

## 4. Deployment

- **Conda (recommended)**: install `cuda-toolkit` + `cudnn` of the matching major into a dedicated env and ensure `Library\bin` is on `PATH`.
- **Portable**: drop every DLL listed above next to `lunavox-cli.exe`.
- After switching CUDA major version, run `lunavox download-libs --platform win_cuda13` (or `win_cuda12`) and rebuild — leftover binaries from the other major will fail with **error 126** because they reference the wrong `cublasLt64_*.dll`.

## 5. Troubleshooting

1. **Error 126 (module not found)**: a DLL is referencing the other CUDA major. Re-download the libs for the correct platform.
2. **Provider silently CPU**: run with `--verbose`; `decoder` should report `CUDAExecutionProvider`.
3. **llama.cpp falls back to CPU**: `ggml-cuda.dll` does not match the CUDA major. Replace it.
4. **Version skew**: `onnxruntime_providers_cuda.dll` and `onnxruntime.dll` must come from the same build.
