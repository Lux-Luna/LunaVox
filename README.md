<div align="center">

# 🔮 LunaVox

**Lightweight, high-quality, and high-speed inference engine for [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS).**

[简体中文](./README_zh.md) | [English](./README.md)

</div>

---

## 🚀 Introduction

**LunaVox** is a streamlined inference engine dedicated to the GPT-SoVITS project. By decoupling dependencies and utilizing pure ONNX Runtime, it provides a portable, fast, and easy-to-integrate solution for cross-platform speech synthesis.

- **High-speed Inference**: Deeply optimized for ONNX Runtime. Implements **I/O Binding** for zero-copy GPU memory access and **KV-Caching** to significantly reduce autoregressive loop latency.
- **Lightweight**: Minimum dependencies; automatic resource management.
- **Versatile**: Supports GPT-SoVITS V2, Pro, and Pro Plus models in English, Japanese, and Chinese.

---

## 📂 Project Structure

- **`src/lunavox_tts`**: Core runtime and inference logic.
- **`GUI`**: Desktop application for character management and inference.
- **`Tutorial`**: Comprehensive examples for inference and model conversion.
- **`CharacterData`**: Local storage for your character models.
- **`TTSData`**: Shared resources (G2P, Chinese-HuBERT, Speaker-Vector). *Auto-downloaded on first run.*
- **`RoBERTa`**: Chinese RoBERTa resources. *Auto-downloaded on first run.*

> 📖 [View Code Architecture Guide (EN)](docs/en/project_structure.md)

---

## 🎙 Voice Cloning Guide

For perfect character voice cloning, you should use the [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) project to train your own models or download third-party PyTorch models (`.ckpt` and `.pth`), then convert them using LunaVox's conversion tools.

However, if you don't mind a less-than-perfect restoration, LunaVox provides pretrained V2 and V2 Pro Plus models. You only need to provide a `.wav` reference audio to start performing inference for any character.

> [!IMPORTANT]
> **Important Tip**: To achieve the best inference results, it is strongly recommended to use a reference audio that matches the **target language** and set the **correct reference audio text**.

> [!TIP]
> **Voice Persona**: This project provides a **Persona tool** that allows you to "solidify" the timbre of a specific audio. You won't need to provide reference audio repeatedly in subsequent inferences, reducing both disk space and memory overhead.

---

## 🏁 Quick Start

### 1. Installation

```bash
pip install lunavox-tts
```

### 2. Desktop GUI

Launch the aesthetic desktop application for a user-friendly experience:

```bash
python GUI/main.py
```

### 3. Python API

```python
import lunavox_tts as lunavox

# 1. Load a character model bundle (v2 / v2Pro / v2ProPlus)
lunavox.load_character('my_char', './CharacterData/character_model/v2/pretrained')

# 2. Set reference audio for cloning
lunavox.set_reference_audio('my_char', './path/to/ref.wav', 'The reference text.')

# 3. Text-to-Speech
lunavox.tts('my_char', 'Hello, this is LunaVox!', play=True, language='en')
```

---

## 🛠 Advanced Features

### Model Conversion
Transform your PyTorch `.ckpt` and `.pth` models into efficient ONNX/BIN bundles.
See: `Tutorial/pytorch_to_onnx_bin_demo.py`

### V2 Pro Plus Support
Low-latency inference with advanced speaker embedding support.
See: `Tutorial/tts_tryout/v2pp_en.py`

### Headless Server
Deploy as a high-performance FastAPI backend.
```python
import lunavox_tts as lunavox
lunavox.start_server(host="0.0.0.0", port=8000)
```

---

## 📦 Minimal Deployment

LunaVox allows for extreme modularity. For disk-space sensitive environments:

*   **Example (V2 English TTS)**: Only **~307 MB** required (Base Engine + V2 Model + English Resources).

> 📖 [See Detailed Dependency Size Analysis](docs/en/dependency_size.md)

---

## ⚙ Runtime Configuration

LunaVox managed via `env_manager`. By default, it detects your hardware (CPU/GPU) and installs the optimized ONNX Runtime and portable CUDA libraries automatically. You can switch modes via GUI or script:

```python
from lunavox_tts.Utils.EnvManager import env_manager
env_manager.set_mode("cpu") # or "gpu"
env_manager.ensure_environment() # Trigger auto-install if missing
```

---

## 📅 Roadmap & Progress

### ✅ Completed
- **Language Core**
  - [x] Simplified Chinese Support
  - [x] English Support
- **Model Architecture**
  - [x] V2 Pro Plus Inference Support
- **Performance**
  - [x] GPU Acceleration & Optimization (CUDA/DirectML)
  - [x] IO Binding Zero-copy Optimization
  - [x] KV Cache Acceleration (On-device Persistence)
- **Advanced Features**
  - [x] Persona Support (Speaker Embedding Fixing)

### 🚀 Planned
- **User Experience**
  - [ ] Windows All-in-one Portable Package
- **Advanced Features**
  - [ ] Enhanced Emotion Control
