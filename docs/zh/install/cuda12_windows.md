# LunaVox Windows CUDA 12 依赖集说明

本文档针对 LunaVox 在 Windows 平台下使用 CUDA 12 及其对应 cuDNN 9 推理时的依赖情况进行说明，旨在提供明确的最小依赖集指导。

---

## 1. 验证环境

本文档基于以下运行时组合的测试结果：

- **ONNX Runtime**: `1.24.4`
- **CUDA Runtime family**: `12.4.x`
- **cuDNN**: `9.1.1.17`
- **平台**: Windows x64
- **GPU**: NVIDIA RTX 3090 (Compute Capability 8.6)

实测环境中的关键包版本：
- `cuda-toolkit 12.4.1`
- `cudnn 9.1.1.17`
- `libcublas 12.4.5.8`
- `libcufft 11.2.1.3`
- `libcurand 10.3.5.147`

---

## 2. cuDNN 9 依赖特性

**特别注意**：Windows 上的 cuDNN 9 不再是单文件，不能只放一个 `cudnn64_9.dll`。

`cudnn64_9.dll` 仅作为入口 DLL，实际运行时需要一整组 `cudnn_*64_9.dll`。如果只拷贝入口文件，CUDA 加速将无法正常工作。完整的 cuDNN 9 DLL 家族包含：

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

为确保 LunaVox 各组件（包括 GGML/llama.cpp 和 ONNX Runtime）能成功启用 CUDA 加速，需要确保以下 DLL 在程序搜索路径内。

### 3.1 核心运行时 DLL (LunaVox 相关)

- `onnxruntime.dll` (GPU 版本)
- `onnxruntime_providers_cuda.dll`
- `onnxruntime_providers_shared.dll`
- `ggml-cuda.dll`

### 3.2 CUDA 12 / cuDNN 9 支撑 DLL

- `cudart64_12.dll`
- `cublas64_12.dll`
- `cublasLt64_12.dll`
- `cufft64_11.dll`
- `curand64_10.dll`
- **cuDNN 家族 (见第 2 节列出的全套 8 个 DLL)**

### 3.3 Windows / MSVC 运行时

LunaVox 及其依赖项需要标准 C/C++ 运行时支持：

- `vcruntime140.dll`
- `vcruntime140_1.dll`
- `msvcp140.dll`

---

## 4. 准备方式

### 4.1 使用 Conda 环境 (推荐)

在已安装 `cuda-toolkit 12.x` 和 `cudnn 9.x` 的 conda 环境中运行。应确保环境满足：
- 关键包：`cuda-toolkit`, `cudnn`, `vs2015_runtime` / `vc14_runtime`。
- 确保 `Library\bin` 路径已加入 `PATH`。

### 4.2 绿色部署 (脱离 Conda)

如果将 LunaVox 做成绿色软件包，必须确保上述所有 DLL（共约 20 个）物理存在于程序主执行文件同级目录，或存在于一个已加入 `PATH` 的固定路径中。

**严禁只拷贝 `cudnn64_9.dll` 而忽略其他 `cudnn_*.dll` 分量。**

---

## 5. 依赖自查表

如果 CUDA 无法正常启用，请按此清单检查文件是否存在且版本匹配：

1. **核心库**: `onnxruntime.dll`, `onnxruntime_providers_cuda.dll`, `ggml-cuda.dll`
2. **CUDA 基础**: `cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`
3. **cuDNN 全家桶**: 确认 `cudnn_adv64_9.dll` 至 `cudnn_ops64_9.dll` 共 8 个文件齐全
4. **数学库**: `cufft64_11.dll`, `curand64_10.dll`
5. **系统库**: `vcruntime140.dll`, `vcruntime140_1.dll`, `msvcp140.dll`

---

## 6. 排障建议

1. 确认 `metadata.json` 中 `onnx.provider` 为 `CUDAExecutionProvider`
2. 确认程序实际加载的是 GPU 版 ONNX Runtime，而不是 CPU 版
3. 确认 `onnxruntime_providers_cuda.dll` 与 `onnxruntime.dll` 版本一致
4. 确认 `PATH` 中存在完整 cuDNN 9 DLL 集，而不是仅有 `cudnn64_9.dll`
5. 确认 `PATH` 中存在 `cudart64_12.dll`、`cublas64_12.dll`、`cublasLt64_12.dll`、`cufft64_11.dll`、`curand64_10.dll`
6. 确认系统已安装可用的 MSVC runtime
7. 运行 TTS 时使用 `--verbose` 或 `--ort-debug-log` 观察 provider 实际启用情况

---

## 7. 当前推荐结论

对于 LunaVox Windows CUDA 12：

- 最稳妥的运行方式是使用包含 `cuda-toolkit 12.x` 和 `cudnn 9.x` 的 conda 环境
- 如果要脱离 conda 做“绿色部署”，不要只拷 `cudnn64_9.dll`
- 最小集应至少包含完整的 cuDNN 9 DLL 家族、CUDA runtime 相关 DLL、ORT GPU DLL、GGML CUDA DLL，以及 MSVC runtime
