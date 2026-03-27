# 🌌 LunaVox: Qwen3-TTS C++ 高性能推理引擎

![Version](https://img.shields.io/badge/version-2.1.0-blueviolet?style=for-the-badge)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-0078d7?style=for-the-badge&logo=windows&logoColor=white)
![CoreML](https://img.shields.io/badge/iOS-CoreML-000000?style=for-the-badge&logo=apple&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=c%2B%2B)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](../../LICENSE)

**LunaVox** 是专为 **Qwen3-TTS** 打造的高性能 C++ 推理引擎。它通过精简的架构设计和深度的硬件优化，提供了极致的语音合成速度与灵活性。无论是本地嵌入式设备、桌面应用还是高性能服务器，LunaVox 都能提供稳定、低延迟的 TTS 体感。

---

## 🚀 核心特性

- **轻量级运行**: 仅需 ONNX Runtime 与自定义 Llama 推理库，无需繁重的 Python 环境即可运行。
- **多语言原生支持**: 引擎链路内置自动语言检测，完美支持 **中、英、日、韩、俄、德、法、意、西、葡** 十种语言。
- **全模式支持**: 支持 基础合成 (Base)、声音克隆 (Clone)、定制定制声音 (Custom) 及 创意声音设计 (Design)。
- **现代构建系统**: 全自动工具链识别。支持 Windows (MSVC)、Linux (GCC) 及 macOS (Clang/Apple Silicon)。
- **跨平台硬件加速**: 深度集成 CUDA (NVIDIA), CoreML/Metal (Apple), DML (DirectX 12) 与 Vulkan 接口。

---

## 🛠️ 环境与构建要求

### 1. 系统环境
- **Windows**: Windows 10/11 (VS 2022/2025 支持)
- **Linux**: Ubuntu 22.04+ 或主流发行版 (GCC >= 9.0)
- **macOS**: Apple Silicon (M1/M2/M3), macOS 12+ (Metal 支持)
- **编译器**: MSVC (v143/v144)、GCC 10.0+ 或 Apple Clang
- **构建工具**: CMake 3.16+，建议安装 **Ninja** 提升构建速度

### 2. 依赖库
- **Python 3.10+**: 用于模型转换和自动化管理。
- **ONNX Runtime SDK**: 对应平台的 C++ 动态库。
- **Llama Runtime**: 预编译的后端二进程文件。

---

## 📊 性能评估

下表展示了 LunaVox 在不同后端配置下的平均性能表现。详细报告请参阅 **[Windows 性能评估报告](benchmark/windows_performance.md)**。

| 测试配置 | 平均 RTF | 峰值内存 (RAM) | 显存 (VRAM) | 相对加速比 (Speedup) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (CPU)** | 5.066 | 5.06 GB | — | 1.00x |
| **Baseline (GPU)** | 3.788 | 1.59 GB | 2.29 GB | 1.34x |
| **LunaVox (Full CPU)** | 1.152 | 1.06 GB | — | 4.40x |
| **LunaVox (CUDA13)** | 0.254 | 1.39 GB | 1.30 GB | 19.94x |
| **LunaVox (Llama.cpp (Vulkan) / ORT (DML))** | **0.206** | 0.91 GB | 1.05 GB | **24.59x** |

> [!NOTE]
> - **测试模型**: 基于 **Qwen3-TTS-12Hz-0.6B-Base**，开启声音克隆模式并在使用 `.json` 预计算特征文件作为参考。
> - **测试环境**: Intel i9-12900K + NVIDIA RTX 3090
> - **测试标准**: 在 **3 次预热**后，取 **10 次运行**的平均结果。

---

### 3. CLI 工具与依赖安装

```powershell
# 安装核心推理工具
pip install lunavox
```

> [!NOTE]
> **开发与脚本说明**: LunaVox 已发布至 PyPI，标准用户仅需执行 `pip install lunavox` 即可安装完整工具。若您需要深入研究模型转换、量化流水线或导出 Python 脚本，请切换至 **`dev`** 分支获取最新源码与内部工具。

## 📦 快速上手流程 (One-Key Setup)

LunaVox 推荐使用 `bootstrap` 指令一键完成 **模型拉取、运行库下载、项目构建及交互测试**。

### 1. 自动引导安装 (推荐)
```powershell
# 执行全自动引导设置
lunavox bootstrap
```

### 2. 本地构建 (从源码)
如果您需要精细化控制每个步骤，可以运行：
```powershell
# 1. 下载预转换模型 (或使用 convert 本地转换原始模型)
lunavox pull-model

# 2. 下载 C++ 运行库
lunavox download-libs

# 3. 自动编译项目
lunavox build --clean
```

> [!TIP]
> 更多详细命令和高级参数说明，请参阅 **[LunaVox CLI 指令汇总手册](guide/cli_reference.md)**。

---

## 🧱 运行库依赖 (Libraries)

LunaVox 自动下载 `lib/` 下相应的 ONNX Runtime 与 Llama.cpp。如果您需要针对 CUDA 环境进行精细化配置，请参阅：
- **[CUDA 12 Windows 依赖指南](install/cuda12_windows.md)**
- **[CUDA 13 Windows 依赖指南](install/cuda13_windows.md)**

---


## 🎙️ 推理测试与模式说明

编译完成后，可执行程序位于 `./build/qwen3-tts-cli.exe`。
> [!NOTE]
> - Linux/macOS 系统请使用 `./build/qwen3-tts-cli` 运行。
> - `--instruct` 仅对 **Custom** 和 **Design** 模式有效（Base 模式下禁用）。

详细教程请参阅：**[CLI 指令使用指南](guide/usage_tutorial.md)**。

### 1. 声音克隆 (Voice Cloning)
通过参考音频（.wav）或预计算特征（.json）模仿特定音色：
```bash
./build/qwen3-tts-cli.exe `
  -m models/base_small `
  -r ref/ref_0.6B.json `
  -t "Okay, fine, I'm just gonna leave this sock monkey here. Goodbye." `
  -o output/cloned.wav
```

### 2. 定制化声音 (Custom Voice)
使用系统内置的发音人 ID：
```bash
./build/qwen3-tts-cli.exe `
  -m models/custom `
  --speaker Vivian `
  --instruct "Use angry tone." `
  -t "She said she would be here by noon." `
  -o output/custom.wav
```

### 3. 声音设计 (Voice Design)
使用描述设计声音
```bash
.\build\qwen3-tts-cli.exe `
  -m models/design `
  -t "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!" `
  --instruct "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
  -o output/out.wav `

---

## 📈 性能监控与日志

- **详细统计**: 运行命令时添加 `--stats-json report.json` 即可获取 RTF（实时率）和内存占用分析。
- **日志查看**: 所有的构建和运行输出均实时记录在 `../../logs/latest.log` 中。
- **线程控制**: 使用 `-j` 参数（默认 4）调整 CPU 线程使用。

---

## 📜 更多信息

有关运行时详细配置及设计准则，请参阅:
- **[LunaVox CLI 指令汇总手册](guide/cli_reference.md)**
- **[运行时技术规范与约束](technical/runtime_specs.md)**
- **[合成链路编码器需求分析](technical/synthesis_pathway.md)**

---

## 🙏 致谢

本项目深受以下开源项目的启发或基于其成果：

- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)**: 提供强大的 base 模型权重与原始架构设计。
- **[onnxruntime](https://github.com/microsoft/onnxruntime)**: 驱动高性能音频解码后端。
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)**: 驱动 LLM 序列预测核心。
