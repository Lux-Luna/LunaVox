# 🌌 LunaVox: Qwen3-TTS C++ 高性能推理引擎

![Version](https://img.shields.io/badge/version-2.0.0-blueviolet?style=for-the-badge)
![Platform](https://img.shields.io/badge/platform-Windows-0078d7?style=for-the-badge&logo=windows)
![C++](https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=c%2B%2B)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

**LunaVox** 是专为 **Qwen3-TTS** 打造的高性能 C++ 推理引擎。它通过精简的架构设计和深度的硬件优化，提供了极致的语音合成速度与灵活性。无论是本地嵌入式设备还是高性能服务器，LunaVox 都能提供稳定、低延迟的 TTS 体感。

---

## 🚀 核心特性

- **轻量级运行**: 仅需 ONNX Runtime 与自定义 Llama 推理库，无需繁重的 Python 环境即可运行。
- **全模式支持**: 支持 基础合成 (Base)、声音克隆 (Clone)、定制定制声音 (Custom) 及 创意声音设计 (Design)。
- **现代构建系统**: 自动检测 Visual Studio 2022/2025 环境，支持 Ninja 高速编译，具备结构化的视觉反馈。
- **跨平台一致性**: 严格的模型转换流程，确保 C++ 推理结果与原始 Qwen 模型高度对齐。

---

## 🛠️ 环境与构建要求

### 1. 系统环境
- **操作系统**: Windows 10/11 (VS 2025/2022 支持)
- **编译器**: MSVC (v143/v144) 或 MinGW-w64 (GCC >= 7.0)
- **构建工具**: CMake 3.16+，建议安装 **Ninja** 提升构建速度

### 2. 依赖库
- **Python 3.10+**: 用于模型转换和自动化管理。
- **ONNX Runtime SDK**: 放置于 `./lib/onnx`。
- **Llama Runtime**: 预编译文件放置于 `./lib`。

### 3. CLI 工具与依赖安装
```powershell
# 以可编辑模式安装 CLI 工具及其转换依赖
pip install -e .[convert]
```

---

## 📦 快速上手流程

LunaVox 提供了统一的 `lunavox` CLI 工具来管理整个工作流。

### 1. 模型准备 (Setup & Convert)

您可以通过以下两种方式准备模型：

**方式 A：直接下载预转换模型 (推荐)**
直接从 HuggingFace 仓库下载已转换好的运行格式：
```powershell
lunavox pull-model --model base_small
```

**方式 B：本地从源权重转换 (Convert)**
下载 HuggingFace 原始权重并自动转换为 LunaVox 优化格式（GGUF/ONNX）：
```powershell
# 转换指定模型（会自动检测并下载缺失的源权重）
lunavox --project-root D:\TTS\lunavox --yes convert --model base_small
```

### 2. 下载运行库 (Download Libs)
下载推理引擎依赖的二进制库（ONNX Runtime / Llama）：
```powershell
# 下载 Windows CPU 后端的运行库
lunavox download libs llama win_cpu
lunavox download libs onnx win_cpu
```

### 3. 项目构建 (Build)
自动配置 CMake 并编译 C++ 推理引擎：
```powershell
# 执行清理并使用 8 线程加速构建
lunavox build --clean --j 8
```

> [!TIP]
> 使用 `lunavox doctor` 可以随时检查当前环境的依赖完整性。

---

## 🛠️ 进阶管理命令

通过 `lunavox --help` 查看所有可用选项。

### 环境诊断
```powershell
# 检查编译器、运行库及 Python 依赖状态
lunavox doctor
```

### 详尽模式构建
如果构建失败，可以使用详尽模式查看 CMake 输出：
```powershell
lunavox build --verbose
```

---

## 🎙️ 推理测试与模式说明

编译完成后，可执行程序位于 `./build/qwen3-tts-cli.exe`。

### 1. 基础模式 (Base Mode)
使用默认参数合成最纯粹的声音：
```powershell
./build/qwen3-tts-cli -m models/base_small -t "君不见黄河之水天上来，奔流到海不复回。" -o output.wav
```

### 2. 声音克隆 (Voice Cloning)
通过参考音频（.wav）或预计算特征（.json）模仿特定音色：
```powershell
./build/qwen3-tts-cli --mode clone -m models/base_small -r ref/sample.wav -t "你好，我是你的语音助手。" -o cloned.wav
```

### 3. 定制化声音 (Custom Voice)
使用系统内置的发音人 ID：
```powershell
./build/qwen3-tts-cli --mode custom -m models/base_small --speaker Vivian -t "Welcome to the future of voice synthesis." -o custom.wav
```

### 4. 声音设计 (Voice Design)
根据文字描述动态生成全新的声音：
```powershell
./build/qwen3-tts-cli --mode design -m models/base_small --instruct "A warm and gentle female voice" -t "你好呀！" -o design.wav
```

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
> - **测试环境**: Intel i9-12900K + NVIDIA RTX 3090
> - **测试标准**: 在 **3 次预热**后，取 **100 次运行**的平均结果。
> - **RTF (Real-Time Factor)**: 数值越低表示速度越快（1.0 表示实时）。

---

## 📈 性能监控与日志

- **详细统计**: 运行命令时添加 `--stats-json report.json` 即可获取 RTF（实时率）和内存占用分析。
- **日志查看**: 所有的构建和运行输出均实时记录在 `logs/latest.log` 中。
- **线程控制**: 使用 `-j` 参数（默认 4）调整 CPU 线程使用。

---

## 📜 开源协议

本项目采用 [MIT License](LICENSE) 开源。

