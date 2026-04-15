# LunaVox Windows CUDA 依赖指南

本文涵盖 Windows x64 下 **CUDA 12** 与 **CUDA 13** 两套运行时。请按你使用 `lunavox build libs --platform win_cuda12`（或 `win_cuda13`）下载到的二进制对照查表。

## 1. 实测组合

| 组件 | CUDA 12 | CUDA 13 |
| :--- | :--- | :--- |
| ONNX Runtime | 1.24.4 (GPU) | 1.24.4 (GPU, CUDA 13 构建) |
| llama.cpp | — | b8470 (CUDA 13.1) |
| CUDA Toolkit | 12.4.x | 13.2.0 |
| cuDNN | 9.1.1 | 9.20.0 |
| 测试 GPU | RTX 3090 (CC 8.6) | RTX 3090 |

## 2. cuDNN 9 不是单文件

Windows 上 `cudnn64_9.dll` 只是入口 shim，不论 CUDA 主版本，**必须** 同时存在以下 8 个 DLL：

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

只拷贝 `cudnn64_9.dll` 是 "CUDA 加速静默失效" 的最常见原因。

## 3. 最小 DLL 集

### 核心（LunaVox 自身）

- `onnxruntime.dll`（GPU 版）
- `onnxruntime_providers_cuda.dll`
- `onnxruntime_providers_shared.dll`
- `ggml-cuda.dll`

### CUDA / 数学库

| | CUDA 12 | CUDA 13 |
| :--- | :--- | :--- |
| Runtime | `cudart64_12.dll` | `cudart64_13.dll` |
| cuBLAS | `cublas64_12.dll`、`cublasLt64_12.dll` | `cublas64_13.dll`、`cublasLt64_13.dll` |
| cuFFT | `cufft64_11.dll` | `cufft64_12.dll` |
| cuRAND | `curand64_10.dll` | `curand64_10.dll` |
| cuDNN | 第 2 节列出的 8 个 DLL | 第 2 节列出的 8 个 DLL |

### MSVC 运行时

`vcruntime140.dll`、`vcruntime140_1.dll`、`msvcp140.dll`

## 4. 部署

- **Conda（推荐）**：在独立 env 中安装匹配主版本的 `cuda-toolkit` + `cudnn`，确保 `Library\bin` 在 `PATH` 中。
- **绿色部署**：把上面所有 DLL 与 `lunavox-cli.exe` 放在同一目录。
- 切换 CUDA 主版本后，执行 `lunavox build libs --platform win_cuda13`（或 `win_cuda12`）并重新构建。残留的旧版本二进制会因引用错位的 `cublasLt64_*.dll` 报 **错误 126**。

## 5. 排障

1. **错误 126（找不到模块）**：DLL 内部引用了另一个 CUDA 主版本的库。请重新下载对应平台的运行库。
2. **Provider 静默回落 CPU**：用 `--verbose` 运行，`decoder` 应当为 `CUDAExecutionProvider`。
3. **llama.cpp 回落 CPU**：`ggml-cuda.dll` 与 CUDA 主版本不匹配，需替换。
4. **版本错位**：`onnxruntime_providers_cuda.dll` 与 `onnxruntime.dll` 必须来自同一次构建。
