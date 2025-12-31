<div align="center">

# 🔮 LunaVox

**Lightweight, high-quality, and high-speed inference engine for [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS).**

[简体中文](./README_zh.md) | [English](./README.md)

</div>

---

## 🚀 Introduction

**LunaVox** is a streamlined inference engine dedicated to the GPT-SoVITS project. By decoupling dependencies and utilizing pure ONNX Runtime, it provides a portable, fast, and easy-to-integrate solution for cross-platform speech synthesis.

- **Fast**: Specialized for ONNX Runtime with optimized I/O binding and KV-Caching.
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

## ⚙ Runtime Configuration

LunaVox managed via `env_manager`. By default, it detects your hardware (CPU/GPU) and installs the optimized ONNX Runtime and portable CUDA libraries automatically. You can switch modes via GUI or script:

```python
from lunavox_tts.Utils.EnvManager import env_manager
env_manager.set_mode("cpu") # or "gpu"
env_manager.ensure_environment() # Trigger auto-install if missing
```

---
