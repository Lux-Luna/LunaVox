# 🌌 LunaVox: Qwen3-TTS C++ 高性能推理引擎

![Version](https://img.shields.io/badge/version-2.0.0-blueviolet?style=for-the-badge)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-0078d7?style=for-the-badge&logo=windows&logoColor=white)
![CoreML](https://img.shields.io/badge/iOS-CoreML-000000?style=for-the-badge&logo=apple&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=c%2B%2B)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)

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

下表展示了 LunaVox 在不同后端配置下的平均性能表现。

| 测试配置 | 平均 RTF | 峰值内存 (RAM) | 显存 (VRAM) | 相对加速比 (Speedup) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (CPU)** | 5.066 | 5.06 GB | — | 1.00x |
| **Baseline (GPU)** | 3.788 | 1.59 GB | 2.29 GB | 1.34x |
| **LunaVox (Full CPU)** | 0.935 | 1.54 GB | — | 5.42x |
| **LunaVox (Vulkan)** | 0.327 | 1.47 GB | 1.66 GB | 15.492x |
| **LunaVox (CUDA)** | **0.216** | 1.49 GB | 1.68 GB | **23.45x** |

> [!NOTE]
> - **测试模型**: 基于 **Qwen3-TTS-12Hz-0.6B-Base**，开启声音克隆模式并使用 `.json` 预计算特征文件作为参考。
> - **测试环境**: Intel i9-12900K + NVIDIA RTX 3090
> - **测试标准**: 在 **3 次预热**后，取 **100 次运行**的平均结果。
> - **RTF (Real-Time Factor)**: 数值越低表示速度越快（1.0 表示实时）。

---

### 3. CLI 工具与依赖安装
```powershell
# 以可编辑模式安装 CLI 工具及其转换依赖
pip install -e .[convert]
```

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
lunavox pull-model  # 或 lunavox convert

# 2. 下载 C++ 运行库
lunavox download-libs

# 3. 自动编译项目
lunavox build --clean
```

> [!TIP]
> 更多详细命令和高级参数说明，请参阅 **[LunaVox CLI 指令汇总手册](docs/LUNAVOX_COMMANDS_ZH.md)**。

---

## 🧱 运行库依赖 (Libraries)

LunaVox 自动下载 `lib/` 下合作 ONNX Runtime 与 Llama.cpp。若要手动更换，请将库放置在对应目录并确保 `lib/metadata.json` 指向正确的后端：

| 组件 | 官方支持的后端 (Backend / Provider) |
| :--- | :--- |
| **ONNX (音频)** | `CPU`, `CUDA`, `DML` (Windows), `ROCm` (Linux), `CoreML` (macOS), `Vulkan` |
| **Llama (预测)** | `cpu`, `cuda`, `vulkan`, `metal` (macOS), `rocm`, `sycl` (Intel) |

---


## 🎙️ 推理测试与模式说明

编译完成后，可执行程序位于 `./build/qwen3-tts-cli.exe`。
> [!NOTE]
> - Linux/macOS 系统请使用 `./build/qwen3-tts-cli` 运行。
> - `--instruct` 为可选参数，且**仅对 1.7B 系列模型有效** (0.6B 版本将忽略该参数)。

### 1. 声音克隆 (Voice Cloning)
通过参考音频（.wav）或预计算特征（.json）模仿特定音色：
```bash
./build/qwen3-tts-cli.exe `
  -m models/base `
  -r ref/ref_1.7B.json `
  --instruct "Maintain the speaker's original emotional tone and rhythm." `
  -t "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye." `
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
根据文字描述动态生成全新的声音：
```bash
./build/qwen3-tts-cli.exe `
  -m models/design `
  --instruct "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice." `
  -t "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!" `
  -o output/design.wav
```


---

## 📈 性能监控与日志

- **详细统计**: 运行命令时添加 `--stats-json report.json` 即可获取 RTF（实时率）和内存占用分析。
- **日志查看**: 所有的构建和运行输出均实时记录在 `logs/latest.log` 中。
- **线程控制**: 使用 `-j` 参数（默认 4）调整 CPU 线程使用。

---

## 🙏 致谢

本项目深受以下开源项目的启发或基于其成果：

- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)**: 提供强大的 base 模型权重与原始架构设计。
- **[onnxruntime](https://github.com/microsoft/onnxruntime)**: 驱动高性能音频解码后端。
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)**: 驱动 LLM 序列预测核心。

