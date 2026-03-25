# LunaVox Windows CUDA 13 依赖集说明

本文档针对 LunaVox 在 Windows 平台下使用 CUDA 13 及其对应 cuDNN 9 推理时的依赖情况进行说明，旨在提供明确的最小依赖集指导。

---

## 1. 核心运行时组合 (已验证)

以下组合已在 Windows x64 平台下实测成功：

- **ONNX Runtime**: `1.24.4` (GPU 版, CUDA 13 构建)
- **Llama.cpp**: `b8470` (CUDA 13.1 构建)
- **CUDA Toolkit**: `13.2.0` (推荐版本)
- **cuDNN**: `9.20.0` (兼容 CUDA 13.x)

---

## 2. cuDNN 9 依赖特性

**特别注意**：在 Windows 上，cuDNN 9 并不是单一文件。虽然环境升级到了 CUDA 13，但配套的 cuDNN 9 仍然需要一整组 `cudnn_*64_9.dll`。

`cudnn64_9.dll` 仅作为入口，实际运行时必须包含以下完整的 cuDNN 9 家族：

- `cudnn64_9.dll`
- `cudnn_adv64_9.dll`
- `cudnn_cnn64_9.dll`
- `cudnn_engines_precompiled64_9.dll`
- `cudnn_engines_runtime_compiled64_9.dll`
- `cudnn_graph64_9.dll`
- `cudnn_heuristic64_9.dll`
- `cudnn_ops64_9.dll`

---

## 3. 最小依赖集 (DLL 清单)

为确保 LunaVox 各组件（包括 GGML/llama.cpp 和 ONNX Runtime）能成功启用 CUDA 13 加速，需要确保以下 DLL 在程序搜索路径内。

### 3.1 核心运行时 DLL (LunaVox 相关)

必须使用针对 CUDA 13 编译/构建的二进制库文件：

- `onnxruntime.dll` (CUDA 13 支持版)
- `onnxruntime_providers_cuda.dll` (需引用 `cublasLt64_13.dll` 而非 `_12.dll`)
- `onnxruntime_providers_shared.dll`
- `ggml-cuda.dll` (CUDA 13 支持版)

### 3.2 CUDA 13 / cuDNN 9 支撑 DLL

- `cudart64_13.dll`
- `cublas64_13.dll`
- `cublasLt64_13.dll`
- `cufft64_12.dll` (注意：CUDA 13 中的 cuFFT 版本号跳转至 12)
- `curand64_10.dll` (注意：cuRAND 仍沿用版本号 10)
- **cuDNN 家族 (见第 2 节列出的全套 8 个 DLL)**

### 3.3 Windows / MSVC 运行时

- `vcruntime140.dll`
- `vcruntime140_1.dll`
- `msvcp140.dll`

---

## 4. 准备方式

### 4.1 环境部署 (Conda)

推荐使用 conda 统一管理 CUDA 13 基础组件：

```bash
conda create -n cuda13 python=3.13 -y
conda install -n cuda13 cuda-toolkit=13.2 cudnn -c nvidia -y
```

### 4.2 二进制文件更新

如果当前 `build` 目录中仍是旧版本的二进制文件（如 `onnxruntime_providers_cuda.dll` 仍报找不到 `cublasLt64_12.dll` 的错误），需要通过 `lib_downloader.py` 下载对应的 CUDA 13 版运行库：

```bash
# 下载 CUDA 13 版本的库文件
python src/lunavox/build/lib_downloader.py onnx win_cuda13 build
python src/lunavox/build/lib_downloader.py llama win_cuda13 build

# 然后将下载好的 DLL 部署至执行目录
cp build/lib/onnx/lib/*.dll build/
cp build/lib/llama/*.dll build/
```

---

## 5. 依赖自查表

如果 CUDA 无法正常启用，请按此清单检查文件是否存在且版本匹配：

1. **核心库版本**: 确认 `onnxruntime_providers_cuda.dll` 不再尝试加载 `cublasLt64_12.dll` (可用 strings 检查)。
2. **CUDA 13 基础**: `cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll`。
3. **cuDNN 全家桶**: 确认 `cudnn64_9.dll` 等 8 个文件齐全。
4. **数学库**: `cufft64_12.dll`, `curand64_10.dll`。

---

## 6. 排障建议

1. **错误 126 (找不到指定模块)**: 通常是由于 DLL 内部存在对 CUDA 12 版本库名（如 `cublasLt64_12.dll`）的引用，而环境只有 CUDA 13。请通过更新二进制文件解决。
2. **Provider 检查**: 运行 TTS 后检查 `stats-json` 输出，确认 `decoder` 字段为 `CUDAExecutionProvider`。
3. **Llama.cpp 警告**: 若 `llama.cpp` 回退至 CPU，请确认 `ggml-cuda.dll` 是否已替换为针对 CUDA 13 构建的版本。

