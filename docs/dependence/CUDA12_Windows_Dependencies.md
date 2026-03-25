# LunaVox Windows CUDA 12 依赖调查报告

本报告基于对 `qwen3-tts-cli.exe`、`ggml-cuda.dll` 和 `onnxruntime_providers_cuda.dll` 的二进制扫描，总结了在 Windows 平台上启用 GPU 加速所需的最小系统级依赖项。

## 1. 调查背景
在执行 `qwen3-tts-cli.exe` 时，系统提示以下警告并回退到 CPU 模式：
- `[WARN] Llama.cpp dependency (cuda) is incomplete, falling back to CPU.`
- `[WARN] ONNX Runtime dependency (CUDA) is incomplete, falling back to CPU.`

为了解决此问题，需要确保系统 `PATH` 或程序运行目录（`build/`）中包含以下 CUDA 12 相关 DLL。

## 2. 核心组件依赖清单

### A. Llama.cpp 后端 (`ggml-cuda.dll`)
该组件负责 LLM 推理（Token 生成），主要依赖 CUDA 核心运行时和 BLAS 数学库。
- **必要 DLL**:
  - `cudart64_12.dll` (CUDA Runtime 12)
  - `cublas64_12.dll` (cuBLAS 12)
  - `cublasLt64_12.dll` (cuBLAS Lightweight 12)

### B. ONNX Runtime 后端 (`onnxruntime_providers_cuda.dll`)
该组件负责音频解码加速，对 cuDNN 和 CUDA 库有较多依赖。
- **必要 DLL**:
  - `cudart64_12.dll` (与 Llama.cpp 共用)
  - **`cudnn64_9.dll`** (cuDNN 9.x SDK，关键加速库)
  - `cublas64_12.dll`
  - `cublasLt64_12.dll`
  - `cufft64_11.dll` (快速傅里叶变换库)
  - `curand64_10.dll` (随机数生成库)

### C. 系统基础运行时 (Required for C++/Builds)
这些是几乎所有 C++ 编译程序在 Windows 上运行的基础。
- `vcruntime140.dll`
- `vcruntime140_1.dll` (64位运行时扩展)
- `msvcp140.dll` (C++ 标准库运行时)

---

## 3. 部署方案建议 (最小集)

若要实现“绿色版”或本地运行免去全局安装 CUDA Toolkit，建议将以下文件放入 `build/` 目录下：

| 文件名称 | 对应功能模块 | 建议来源 |
| :--- | :--- | :--- |
| **`cudart64_12.dll`** | 核心运行时 | CUDA 12.4+ bin/ |
| **`cublas64_12.dll`** | 矩阵计算 | CUDA 12.4+ bin/ |
| **`cublasLt64_12.dll`** | 矩阵计算扩展 | CUDA 12.4+ bin/ |
| **`cudnn64_9.dll`** | cuDNN 加速器 | cuDNN 9.x bin/ |
| **`cufft64_11.dll`** | FFT 音频处理 | CUDA 12.4+ bin/ |
| **`vcruntime140_1.dll`** | MSVC 运行时 | System32 或 Redist |

## 4. 驱动与环境要求
- **显卡驱动**: 必须支持 CUDA 12.0+ (通常建议 NVIDIA 驱动版本 > 525.xx)。
- **硬件**: NVIDIA GPU (RTX 30 系列或更高版本效果最佳)。
- **环境变量**: 如果未放置在 `build` 目录，则必须将包含上述 DLL 的文件夹路径加入用户变量 `PATH`。