[**English**](README.md) | [**中文**](docs/zh/README_ZH.md)

# 🌌 LunaVox: High-Performance C++ Inference Engine for Qwen3-TTS

![Version](https://img.shields.io/badge/version-2.1.0-blueviolet?style=for-the-badge)
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

The following table shows the average performance of LunaVox across different backend configurations. For detailed reports, see the **[Windows Performance Evaluation Report](benchmark/windows_performance.md)**.

| Configuration | Average RTF | Peak RAM | VRAM | Relative Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (CPU)** | 5.066 | 5.06 GB | — | 1.00x |
| **Baseline (GPU)** | 3.788 | 1.59 GB | 2.29 GB | 1.34x |
| **LunaVox (Full CPU)** | 1.152 | 1.06 GB | — | 4.40x |
| **LunaVox (CUDA13)** | 0.254 | 1.39 GB | 1.30 GB | 19.94x |
| **LunaVox (Vulkan + DML)**| **0.206** | 0.91 GB | 1.05 GB | **24.59x** |

> [!NOTE]
> - **Test Model**: Based on **Qwen3-TTS-12Hz-0.6B-Base**, with Voice Cloning enabled using pre-computed `.json` feature files.
> - **Test Environment**: Intel i9-12900K + NVIDIA RTX 3090
> - **Test Standard**: Average of **10 runs** after **3 warmup runs**.

---

### 3. CLI Tool & Dependency Installation

```powershell
# Install core inference tooling
pip install lunavox
```

> [!NOTE]
> **Developer Note**: LunaVox is published on PyPI. Standard users only need to run `pip install lunavox`. For research into model conversion or quantization pipelines, switch to the **`dev`** branch to get the latest source and internal tools.

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
> For detailed commands and advanced parameters, see the **[LunaVox CLI Reference Manual](guide/cli_reference.md)**.

---

## 🧱 Runtime Libraries

LunaVox automatically downloads appropriate ONNX Runtime and Llama.cpp into the `lib/` directory. For CUDA configurations, see:
- **[CUDA 12 Windows Dependency Guide](install/cuda12_windows.md)**
- **[CUDA 13 Windows Dependency Guide](install/cuda13_windows.md)**

---

## 🎙️ Inference Testing & Modes

After building, the executable is located at `./build/qwen3-tts-cli.exe`.
> [!NOTE]
> - On Linux/macOS, use `./build/qwen3-tts-cli`.
> - `--instruct` is only valid for **Custom** and **Design** modes (disabled in Base mode).

Detailed tutorial: **[CLI Usage Tutorial](guide/usage_tutorial.md)**.

### 1. Voice Cloning
Mimic a specific voice using reference audio (.wav) or pre-computed features (.json):
```bash
./build/qwen3-tts-cli.exe `
  -m models/base_small `
  -r ref/ref_0.6B.json `
  -t "Okay, fine, I'm just gonna leave this sock monkey here. Goodbye." `
  -o output/cloned.wav
```

### 2. Custom Voice
Use built-in expert speaker IDs:
```bash
./build/qwen3-tts-cli.exe `
  -m models/custom `
  --speaker Vivian `
  --instruct "Use angry tone." `
  -t "She said she would be here by noon." `
  -o output/custom.wav
```

### 3. Voice Design
Design voice using text descriptions:
```bash
.\build\qwen3-tts-cli.exe `
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

## 📜 More Information

See also:
- **[LunaVox CLI Reference Manual](guide/cli_reference.md)**
- **[Runtime Technical Specifications & Constraints](technical/runtime_specs.md)**
- **[Synthesis Pathway Encoder Requirements](technical/synthesis_pathway.md)**

---

## 🙏 Acknowledgements

Inspired by or based on:
- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)**: Powerful base weights and architecture design.
- **[onnxruntime](https://github.com/microsoft/onnxruntime)**: High-performance audio decoding backend.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)**: Core for LLM sequence prediction.
