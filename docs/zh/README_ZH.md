[**English**](https://github.com/Lux-Luna/LunaVox/blob/main/README.md) | [**中文**](README_ZH.md)

# 🌌 LunaVox：Qwen3-TTS C++ 高性能推理引擎

![Version](https://img.shields.io/badge/version-2.2.2-blueviolet?style=for-the-badge)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-0078d7?style=for-the-badge&logo=windows&logoColor=white)
![CoreML](https://img.shields.io/badge/iOS-CoreML-000000?style=for-the-badge&logo=apple&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=c%2B%2B)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](../../LICENSE)

**LunaVox** 是专为 **Qwen3-TTS** 打造的高性能 C++ 推理引擎，通过精简架构与深度硬件优化，在嵌入式设备、桌面应用与服务器上都能保持稳定低延迟的 TTS 表现。

## 🚀 核心特性

- **轻量运行时** —— 仅需 ONNX Runtime 与自定义 llama.cpp 封装，推理阶段不依赖重型 Python 环境。
- **多语言原生支持** —— 引擎内置自动语言检测，覆盖中 / 英 / 日 / 韩 / 俄 / 德 / 法 / 意 / 西 / 葡。
- **统一 `Voice` API** —— 一次 `engine.synthesize(text, voice, params)` 覆盖 Base / 克隆 / 定制 / 声音设计。
- **HTTP + WebSocket 服务** (`lunavox serve`)：FastAPI 应用，提供 `POST /v1/synth` 与流式 `WS /v1/stream`，详见 [服务层指南](guide/serve.md)。
- **桌面 GUI** (`lunavox gui`)：customtkinter 三视图（合成 / 素材库 / 设置），与 CLI 共用同一进程引擎。
- **Profile 驱动 CLI** —— `~/.lunavox/config.toml` / 环境变量 / 命令行分层合并，`lunavox --profile quality synth …` 一行搞定。
- **跨平台硬件加速** —— CUDA (NVIDIA)、CoreML/Metal (Apple)、DML (DirectX 12)、Vulkan。

## 🛠️ 运行要求

- **Windows** 10/11（VS 2022/2025）、**Linux** Ubuntu 22.04+（GCC ≥ 9.0）或 **macOS** 12+ Apple Silicon
- **CMake 3.16+**（推荐 Ninja）+ C++17 编译器
- **Python 3.10+**（CLI 与转换工具链）

## 📊 性能基线

| 测试配置 | **TTFB (ms)** | RTF | 峰值 RAM | 显存 VRAM | 相对加速比 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 官方 PyTorch 基线 (CPU) | — | 5.066 | 5.06 GB | — | 1.00× |
| 官方 PyTorch 基线 (GPU) | — | 3.788 | 1.59 GB | 2.29 GB | 1.34× |
| **LunaVox (Full CPU)** | 1248 | 0.858 | 1.19 GB | — | 5.90× |
| **LunaVox (CUDA 13)** | 175 | 0.213 | 1.41 GB | 1.33 GB | 23.78× |
| **LunaVox (Vulkan + DML)** | **194** | **0.152** | **0.97 GB** | 1.00 GB | **33.33×** |

模型 `Qwen3-TTS-12Hz-0.6B-Base`，使用 `ref/ref_0.6B.json` 克隆；硬件 Intel i9-12900K + RTX 3090 / Windows 11；5 次预热 + 100 次测量，25 词固定英文句子。逐次分布见 [`benchmark/report.md`](https://github.com/Lux-Luna/LunaVox/blob/main/benchmark/report.md)，详细分析见 [Windows 性能评估报告](benchmark/windows_performance.md)。

## 📦 安装与快速上手

```powershell
pip install lunavox               # 核心 CLI + GUI + HTTP/WebSocket 服务（默认全含）
pip install "lunavox[convert]"    # + 原始权重 → GGUF 转换工具链（重型，按需）
```

```powershell
lunavox bootstrap                 # 一键：拉模型 + 下运行库 + 构建 + 冒烟测试
```

若需分步控制，可依次 `lunavox model pull`、`lunavox build libs`、`lunavox build --clean`。CUDA 配置见 [CUDA Windows 指南](install/cuda_windows.md)；完整命令参考：**[CLI 指令手册](guide/cli_reference.md)**。

## 🎙️ 合成示例

```bash
lunavox synth "Hello from LunaVox." -o out.wav                       # 基础音色
lunavox synth "…" --voice clone  --ref ref/ref_0.6B.json -o out.wav  # 声音克隆
lunavox synth "…" --voice custom --speaker Vivian --instruct "…"     # 内置发音人
lunavox synth "…" --voice design --instruct "A warm, calm narrator." # 文本驱动声音设计
lunavox gui                                                          # 桌面 GUI
```

独立可执行 `./build/lunavox-cli` 与 Python 路径共用同一引擎。完整模式说明见 **[使用教程](guide/usage_tutorial.md)**。

### Python 嵌入式用法

```python
from lunavox.runtime import Engine, SynthesisParams, Voice

with Engine("models/base_small") as engine:
    result = engine.synthesize(
        "你好，来自 LunaVox。",
        voice=Voice.clone_file("ref/ref_0.6B.json"),
        params=SynthesisParams(temperature=0.7),
    )
    print(f"RTF {result.stats.rtf:.3f}")  # result.audio 是 float32 [-1, 1] 单通道数组
```

## 📈 性能监控与日志

- `--stats-json report.json` —— 单次合成的 RTF 与内存分解
- `logs/latest.log` —— 构建与运行输出
- `-j N` —— CPU 线程数（默认 4）

## 🙏 致谢

- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** —— base 权重与架构原型
- **[onnxruntime](https://github.com/microsoft/onnxruntime)** —— 音频解码后端
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** —— LLM 序列预测核心
