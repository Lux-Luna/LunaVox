[**English**](README.md) | [**中文**](docs/zh/README_ZH.md)

# 🌌 LunaVox: High-Performance C++ Inference Engine for Qwen3-TTS

![Version](https://img.shields.io/badge/version-2.1.6-blueviolet?style=for-the-badge)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-0078d7?style=for-the-badge&logo=windows&logoColor=white)
![CoreML](https://img.shields.io/badge/iOS-CoreML-000000?style=for-the-badge&logo=apple&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=c%2B%2B)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)

**LunaVox** is a high-performance C++ inference engine specifically designed for **Qwen3-TTS**. Through streamlined architecture and deep hardware optimization, it provides extreme speech synthesis speed and flexibility. Whether for local embedded devices, desktop applications, or high-performance servers, LunaVox delivers stable, low-latency TTS experience.

---

## 🚀 Key Features

- **Lightweight Runtime**: Runs with only ONNX Runtime and a custom Llama inference library, no heavy Python environment required.
- **Native Multi-language Support**: Built-in automatic language detection, supporting **Chinese, English, Japanese, Korean, Russian, German, French, Italian, Spanish, and Portuguese**.
- **Full Mode Support**: Supports Base synthesis, Voice Cloning, Custom Voice, and Voice Design (Prompt-to-Voice).
- **Modern Build System**: Automatic toolchain detection. Supports Windows (MSVC), Linux (GCC), and macOS (Clang/Apple Silicon).
- **Cross-platform Hardware Acceleration**: Deeply integrated with CUDA (NVIDIA), CoreML/Metal (Apple), DML (DirectX 12), and Vulkan.

---

## 🛠️ Environment & Build Requirements

### 1. System Environment
- **Windows**: Windows 10/11 (VS 2022/2025 supported)
- **Linux**: Ubuntu 22.04+ or mainstream distributions (GCC >= 9.0)
- **macOS**: Apple Silicon (M1/M2/M3), macOS 12+ (Metal support)
- **Compiler**: MSVC (v143/v144), GCC 10.0+, or Apple Clang
- **Build Tools**: CMake 3.16+, **Ninja** is recommended for faster builds.

### 2. Dependencies
- **Python 3.10+**: For model conversion and automation.
- **ONNX Runtime SDK**: Platform-specific C++ dynamic libraries.
- **Llama Runtime**: Pre-compiled backend binaries.

---

## 📊 Performance Benchmarks

The following table shows the average performance of LunaVox across different backend configurations. For detailed reports, see the **[Windows Performance Evaluation Report](docs/en/benchmark/windows_performance.md)**.

| Configuration | **TTFB (ms)** | RTF | Peak RAM | VRAM | Speedup |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Official PyTorch Baseline (CPU) | — | 5.066 | 5.06 GB | — | 1.00× |
| Official PyTorch Baseline (GPU) | — | 3.788 | 1.59 GB | 2.29 GB | 1.34× |
| **LunaVox (Full CPU)** | 1248 | 0.858 | 1.19 GB | — | 5.90× |
| **LunaVox (CUDA 13)** | 175 | 0.213 | 1.41 GB | 1.33 GB | 23.78× |
| **LunaVox (Vulkan + DML)** | **194** | **0.152** | **0.97 GB** | 1.00 GB | **33.33×** |

> [!NOTE]
> - **Test Model**: **Qwen3-TTS-12Hz-0.6B-Base** with voice cloning using the pre-computed `ref/ref_0.6B.json` feature file.
> - **Test Environment**: Intel i9-12900K + NVIDIA RTX 3090, Windows 11.
> - **Test Standard**: 5 warm-up runs (discarded) + **100 measurement runs** per backend, fixed 25-word English sentence. All three backends built from the same commit.
> - **TTFB** (time-to-first-byte) is the streaming-pipeline delay from synth start to the first PCM sample becoming available — the latency a streaming caller actually observes.
> - Full per-run distribution (p50 / p95 / p99 / stddev) and raw stats in [`benchmark/report.md`](benchmark/report.md).

---

### 3. CLI Tool & Dependency Installation

```powershell
# Install core inference tooling
pip install lunavox
```

> [!NOTE]
> **Developer Note**: LunaVox is published on PyPI. Standard users only need to run `pip install lunavox`. For research into model conversion or quantization pipelines, switch to the **[cli-only](https://github.com/Lux-Luna/LunaVox/tree/cli-only)** branch to get the latest source and internal tools.

## 📦 Quick Setup (One-Key Setup)

LunaVox recommends using the `bootstrap` command to complete **Model Pulling, Runtime Library Download, Project Build, and Interactive Testing** in one go.

### 1. Automatic Guided Setup (Recommended)
```powershell
# Execute full automatic setup
lunavox bootstrap
```

### 2. Local Build (From Source)
If you need fine-grained control:
```powershell
# 1. Download pre-converted models (or use 'convert' for local weights)
lunavox pull-model

# 2. Download C++ runtime libraries
lunavox download-libs

# 3. Compile the project
lunavox build --clean
```

> [!TIP]
> For detailed commands and advanced parameters, see the **[LunaVox CLI Reference Manual](docs/en/guide/cli_reference.md)**.

---

## 🧱 Runtime Libraries

LunaVox automatically downloads appropriate ONNX Runtime and Llama.cpp into the `lib/` directory. For CUDA configurations, see:
- **[CUDA Windows Dependency Guide (CUDA 12 / 13)](docs/en/install/cuda_windows.md)**

---

## 🎙️ Inference Testing & Modes

After building, the executable is located at `./build/lunavox-cli.exe`.
> [!NOTE]
> - On Linux/macOS, use `./build/lunavox-cli`.
> - `--instruct` is only valid for **Custom** and **Design** modes (disabled in Base mode).

Detailed tutorial: **[CLI Usage Tutorial](docs/en/guide/usage_tutorial.md)**.

### 1. Voice Cloning
Mimic a specific voice using reference audio (.wav) or pre-computed features (.json):
```bash
./build/lunavox-cli.exe `
  -m models/base_small `
  -r ref/ref_0.6B.json `
  -t "Okay, fine, I'm just gonna leave this sock monkey here. Goodbye." `
  -o output/cloned.wav
```

### 2. Custom Voice
Use built-in expert speaker IDs:
```bash
./build/lunavox-cli.exe `
  -m models/custom `
  --speaker Vivian `
  --instruct "Use angry tone." `
  -t "She said she would be here by noon." `
  -o output/custom.wav
```

### 3. Voice Design
Design voice using text descriptions:
```bash
.\build\lunavox-cli.exe `
  -m models/design `
  -t "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!" `
  --instruct "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
  -o output/out.wav
```

---

## 📈 Monitoring & Logging

- **Detailed Stats**: Add `--stats-json report.json` to get RTF and memory analysis.
- **Logs**: All build and runtime output is logged to `../../logs/latest.log`.
- **Thread Control**: Use `-j` (default 4) to adjust CPU thread usage.

---

## 🙏 Acknowledgements

Inspired by or based on:
- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)**: Powerful base weights and architecture design.
- **[onnxruntime](https://github.com/microsoft/onnxruntime)**: High-performance audio decoding backend.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)**: Core for LLM sequence prediction.
