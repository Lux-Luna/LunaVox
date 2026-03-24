# LunaVox Prebuilt Libraries (lib/)

This directory contains the binary dependencies required by LunaVox, specifically the ONNX Runtime SDK and Llama.cpp runtimes.

## Directory Structure

- `metadata.json`: 指令清单文件，记录当前 `lib/` 目录下库的明确后端意图（如 `CUDAExecutionProvider`）。

## 支持的后端与提供者 (Supported Backends/Providers)

LunaVox 通过 `metadata.json` 显式指定后端。以下是官方支持的名称列表：

### 1. ONNX Runtime (Audio Decoder)
| 提供者名称 (Provider) | 硬件平台 | 备注 |
| :--- | :--- | :--- |
| `CPUExecutionProvider` | 通用 CPU | 默认回退方案 |
| `CUDAExecutionProvider` | NVIDIA GPU | 需要 CUDA Toolkit |
| `DmlExecutionProvider` | Windows DirectX 12 | 适用于大部分 Windows 显卡 |
| `ROCmExecutionProvider` | AMD GPU (Linux) | |
| `CoreMLExecutionProvider` | macOS / iOS | |
| `VulkanExecutionProvider` | 通用 GPU | 通过 Vulkan 实现 |
| `OpenVINOExecutionProvider` | Intel CPU/GPU | |

### 2. Llama.cpp (LLM Engine)
| 后端名称 (Backend) | 硬件平台 |
| :--- | :--- |
| `cpu` | 通用 CPU |
| `cuda` | NVIDIA GPU |
| `vulkan` | 通用 GPU (Windows/Linux) |
| `metal` | macOS (Apple Silicon) |
| `rocm` | AMD GPU |
| `sycl` | Intel GPU |

## 如何管理依赖

### 官方下载 (推荐)
```bash
lunavox download-libs
```
这将自动下载库并生成正确的 `metadata.json` 指令文件。

### 手动安装 (自定义)
如果您手动替换了 `lib/` 下的库：
1. 请确保库文件放置在 `lib/onnx/` 或 `lib/llama/`。
2. **务必更新** `lib/metadata.json`，在 `provider` 或 `backend` 字段填入上述列表中的标准名称。
3. 推理阶段将严格遵循此文件的指令尝试加载硬件加速。

### Resource Links
- **Llama.cpp**: [https://github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases)
- **ONNX Runtime**: [https://github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases)
