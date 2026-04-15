[**English**](https://github.com/Lux-Luna/LunaVox/blob/main/README.md) | [**中文**](README_ZH.md)

# 🌌 LunaVox: Qwen3-TTS C++ 高性能推理引擎

![Version](https://img.shields.io/badge/version-2.2.0-blueviolet?style=for-the-badge)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-0078d7?style=for-the-badge&logo=windows&logoColor=white)
![CoreML](https://img.shields.io/badge/iOS-CoreML-000000?style=for-the-badge&logo=apple&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=c%2B%2B)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](../../LICENSE)

**LunaVox** 是专为 **Qwen3-TTS** 打造的高性能 C++ 推理引擎。它通过精简的架构设计和深度的硬件优化，提供了极致的语音合成速度与灵活性。无论是本地嵌入式设备、桌面应用还是高性能服务器，LunaVox 都能提供稳定、低延迟的 TTS 体感。

---

## 🚀 核心特性

- **轻量级运行**: 仅需 ONNX Runtime 与自定义 Llama 推理库，无需繁重的 Python 环境即可运行。
- **多语言原生支持**: 引擎链路内置自动语言检测，完美支持 **中、英、日、韩、俄、德、法、意、西、葡** 十种语言。
- **统一的 `Voice` API**: 一个 `engine.synthesize(text, voice, params)` 调用即可覆盖 Base、声音克隆、内置发音人、声音设计；不再需要 6 种 `synthesize_*` 方法。
- **HTTP + WebSocket 服务层** (`lunavox serve`): FastAPI 应用，提供 `POST /v1/synth` 和流式 `WS /v1/stream`，底层与 CLI / GUI 共用同一进程内引擎 —— 详见 [服务层指南](guide/serve.md)。
- **桌面 GUI** (`lunavox gui`): 左侧栏三视图布局（合成 / 素材库 / 设置），与 CLI 共用同一进程内引擎。
- **Profile 驱动的 CLI**: `~/.lunavox/config.toml` profile 与环境变量、命令行开关分层合并，让 `lunavox --profile quality synth …` 一行搞定。
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

| 测试配置 | **TTFB (ms)** | RTF | 峰值 RAM | 显存 VRAM | 相对加速比 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 官方 PyTorch 基线 (CPU) | — | 5.066 | 5.06 GB | — | 1.00× |
| 官方 PyTorch 基线 (GPU) | — | 3.788 | 1.59 GB | 2.29 GB | 1.34× |
| **LunaVox (Full CPU)** | 1248 | 0.858 | 1.19 GB | — | 5.90× |
| **LunaVox (CUDA 13)** | 175 | 0.213 | 1.41 GB | 1.33 GB | 23.78× |
| **LunaVox (Vulkan + DML)** | **194** | **0.152** | **0.97 GB** | 1.00 GB | **33.33×** |

> [!NOTE]
> - **测试模型**: **Qwen3-TTS-12Hz-0.6B-Base**，语音克隆使用预计算特征文件 `ref/ref_0.6B.json`。
> - **测试环境**: Intel i9-12900K + NVIDIA RTX 3090，Windows 11。
> - **测试标准**: 每种后端 5 次预热（丢弃）+ **100 次正式测量**，固定 25 词英文句子。三种后端基于同一 commit 构建。
> - **TTFB**（time-to-first-byte）是流式管道从合成开始到首批 PCM 样本可用的墙钟时间 —— 流式调用方实际感受到的延迟。
> - 逐次分布（p50 / p95 / p99 / stddev）与原始数据见 [`benchmark/report.md`](https://github.com/Lux-Luna/LunaVox/blob/main/benchmark/report.md)。

---

### 3. CLI 工具与依赖安装

```powershell
pip install lunavox               # 核心 CLI
pip install "lunavox[serve]"      # + HTTP / WebSocket 服务层
pip install "lunavox[gui]"        # + 桌面 GUI
pip install "lunavox[convert]"    # + 原始权重 → GGUF 转换工具链
```

> [!NOTE]
> **开发与脚本说明**: LunaVox 已发布至 PyPI，标准用户仅需执行 `pip install lunavox` 即可安装完整工具。若您需要深入研究模型转换、量化流水线或导出 Python 脚本，请切换至 **[cli-only](https://github.com/Lux-Luna/LunaVox/tree/cli-only)** 分支获取最新源码与内部工具。

## 📦 快速上手流程 (One-Key Setup)

LunaVox 推荐使用 `bootstrap` 指令一键完成 **模型拉取、运行库下载、项目构建及冒烟合成**。

### 1. 自动引导安装 (推荐)
```powershell
# 执行全自动引导设置
lunavox bootstrap
```

### 2. 本地构建 (从源码)
如果您需要精细化控制每个步骤，可以运行：
```powershell
# 1. 下载预转换模型 (或使用 model convert 本地转换原始模型)
lunavox model pull

# 2. 下载 C++ 运行库
lunavox build libs

# 3. 自动编译项目
lunavox build --clean
```

> [!TIP]
> 更多详细命令和高级参数说明，请参阅 **[LunaVox CLI 指令汇总手册](guide/cli_reference.md)**。

---

## 🧱 运行库依赖 (Libraries)

LunaVox 自动下载 `lib/` 下相应的 ONNX Runtime 与 Llama.cpp。如果您需要针对 CUDA 环境进行精细化配置，请参阅：
- **[CUDA Windows 依赖指南 (CUDA 12 / 13)](install/cuda_windows.md)**

---


## 🎙️ 推理测试与模式说明

`lunavox synth` 会直接驱动 Python `Engine` 并写出 WAV —— GUI 和 benchmark
走的是同一条代码路径。若要进行深度性能分析或在无 Python 环境中运行，
仍可使用独立的 `./build/lunavox-cli` 可执行文件。

详细教程请参阅：**[CLI 指令使用指南](guide/usage_tutorial.md)**。

### 1. 声音克隆 (Voice Cloning)
通过参考音频（`.wav`）或预计算特征（`.json`）模仿特定音色：
```bash
lunavox synth "Okay, fine, I'm just gonna leave this sock monkey here. Goodbye." \
  --voice clone --ref ref/ref_0.6B.json \
  -o output/cloned.wav
```

### 2. 定制化声音 (Custom Voice)
使用系统内置的发音人 ID：
```bash
lunavox synth "She said she would be here by noon." \
  --voice custom --speaker Vivian --instruct "Use angry tone." \
  -o output/custom.wav
```

### 3. 声音设计 (Voice Design)
```bash
lunavox synth "It's in the top drawer... wait, it's empty? No way, that's impossible!" \
  --voice design --instruct "Speak in an incredulous tone, with a hint of panic." \
  -o output/designed.wav
```

### 4. 桌面 GUI
```bash
pip install "lunavox[gui]"
lunavox gui
```
GUI 是三视图左侧栏布局（合成 / 素材库 / 设置），底层直接复用 `Engine`
API，不再拼接 CLI 字符串让用户复制粘贴。

### 5. Python 嵌入式用法
```python
from lunavox.runtime import Engine, SynthesisParams, Voice

with Engine("models/base_small") as engine:
    result = engine.synthesize(
        "你好，来自 LunaVox。",
        voice=Voice.clone_file("ref/ref_0.6B.json"),
        params=SynthesisParams(temperature=0.7),
    )
    # result.audio 是 numpy.float32 单通道数组，范围 [-1, 1]
    print(f"RTF {result.stats.rtf:.3f}")
```

---

## 📈 性能监控与日志

- **详细统计**: 运行命令时添加 `--stats-json report.json` 即可获取 RTF（实时率）和内存占用分析。
- **日志查看**: 所有的构建和运行输出均实时记录在 `../../logs/latest.log` 中。
- **线程控制**: 使用 `-j` 参数（默认 4）调整 CPU 线程使用。

---

## 🙏 致谢

本项目深受以下开源项目的启发或基于其成果：

- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)**: 提供强大的 base 模型权重与原始架构设计。
- **[onnxruntime](https://github.com/microsoft/onnxruntime)**: 驱动高性能音频解码后端。
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)**: 驱动 LLM 序列预测核心。
