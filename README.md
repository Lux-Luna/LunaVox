<div align="center">
<pre>
 ___       ___  ___  ________   ________  ___      ___ ________     ___    ___ 
|\  \     |\  \|\  \|\   ___  \|\   __  \|\  \    /  /|\   __  \   |\  \  /  /|
\ \  \    \ \  \\\  \ \  \\ \  \ \  \|\  \ \  \  /  / | \  \|\  \  \ \  \/  / /
 \ \  \    \ \  \\\  \ \  \\ \  \ \   __  \ \  \/  / / \ \  \\\  \  \ \    / / 
  \ \  \____\ \  \\\  \ \  \\ \  \ \  \ \  \ \    / /   \ \  \\\  \  /     \/  
   \ \_______\ \_______\ \__\\ \__\ \__\ \__\ \__/ /     \ \_______\/  /\   \  
    \|_______|\|_______|\|__| \|__|\|__|\|__|\|__|/       \|_______/__/ /\ __\ 
                                                                   |__|/ \|__| 
</pre>
</div>

<div align="center">

# 🔮 LunaVox: [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) Lightweight Inference Engine

**Experience near-instantaneous speech synthesis on your CPU**

[简体中文](./README_zh.md) | [English](./README.md)

</div>

---

**LunaVox** is a lightweight inference engine built on the open-source TTS
project [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS). It integrates TTS inference, ONNX model conversion, API
server, and other core features, aiming to provide ultimate performance and convenience.

* **✅ Supported Model Version:** GPT-SoVITS V2
* **✅ Supported Language:** Japanese
* **✅ Supported Python Version:** >= 3.9

---

## 🚀 Performance Advantages

LunaVox optimizes the original model for outstanding CPU performance.

| Feature                     |  🔮 LunaVox | Official PyTorch Model | Official ONNX Model |
|:----------------------------|:-----------:|:----------------------:|:-------------------:|
| **First Inference Latency** |  **1.13s**  |         1.35s          |        3.57s        |
| **Runtime Size**            | **\~200MB** |      \~several GB      |  Similar to LunaVox |
| **Model Size**              | **\~230MB** |    Similar to LunaVox  |       \~750MB       |

> 📝 **Note:** Since GPU inference latency does not significantly improve over CPU for the first packet, we currently
> only provide a CPU version to ensure the best out-of-the-box experience.
>
> 📝 **Latency Test Info:** All latency data is based on a test set of 100 Japanese sentences (\~20 characters each),
> averaged. Tested on CPU i7-13620H.

---

## 🏁 QuickStart

> **⚠️ Important:** It is recommended to run LunaVox in **Administrator mode** to avoid potential performance degradation.

### 📦 Installation

Install via pip:

```bash
pip install lunavox-tts
```

> 📝 **You may encounter an installation failure when trying to install pyopenjtalk. This is because pyopenjtalk
> is a library that includes C extensions, and the publisher does not currently provide pre-compiled binary packages (
> wheels).
> For Windows users, this requires
installing [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). Specifically, you
must select the "Desktop
> development with C++" workload during the installation process.**

### ⚡️ Quick Tryout

No GPT-SoVITS model yet? No problem!
LunaVox includes predefined speaker characters for immediate use without any model
files. Run the script below to hear it in action:

```bash
python Tutorial/quick_tryout.py
```

This script will automatically download required dependencies and play a sample audio.

### 🎤 TTS Best Practices

A simple TTS inference example:

```python
import lunavox_tts as lunavox

# Step 1: Load character voice model
lunavox.load_character(
    character_name='<CHARACTER_NAME>',  # Replace with your character name
    onnx_model_dir=r"<PATH_TO_CHARACTER_ONNX_MODEL_DIR>",  # Folder containing ONNX model
)

# Step 2: Set reference audio (for emotion and intonation cloning)
lunavox.set_reference_audio(
    character_name='<CHARACTER_NAME>',  # Must match loaded character name
    audio_path=r"<PATH_TO_REFERENCE_AUDIO>",  # Path to reference audio
    audio_text="<REFERENCE_AUDIO_TEXT>",  # Corresponding text
)

# Step 3: Run TTS inference and generate audio
lunavox.tts(
    character_name='<CHARACTER_NAME>',  # Must match loaded character
    text="<TEXT_TO_SYNTHESIZE>",  # Text to synthesize
    play=True,  # Play audio directly
    save_path="<OUTPUT_AUDIO_PATH>",  # Output audio file path
)

print("🎉 Audio generation complete!")
```

---

## 🔧 Model Conversion

To convert original GPT-SoVITS models for LunaVox, ensure `torch` is installed:

```bash
pip install torch
```

Use the built-in conversion tool:

> **Tip:** `convert_to_onnx` currently supports only V2 models.

```python
import lunavox_tts as lunavox

lunavox.convert_to_onnx(
    torch_pth_path=r"<YOUR .PTH MODEL FILE>",  # Replace with your .pth file
    torch_ckpt_path=r"<YOUR .CKPT CHECKPOINT FILE>",  # Replace with your .ckpt file
    output_dir=r"<ONNX MODEL OUTPUT DIRECTORY>"  # Directory to save ONNX model
)
```

---

## 🌐 Launch FastAPI Server

LunaVox includes a lightweight FastAPI server:

```python
import lunavox_tts as lunavox

# Start server
lunavox.start_server(
    host="0.0.0.0",  # Host address
    port=8000,  # Port
    workers=1  # Number of workers
)
```

> For request formats and API details, see our [API Server Tutorial](./Tutorial/English/API%20Server%20Tutorial.py).

---

## ⌨️ Launch CMD Client

LunaVox provides a simple command-line client for quick testing and interactive use:

```python
import lunavox_tts as lunavox

# Launch CLI client
lunavox.launch_command_line_client()
```

---

## 📝 Roadmap

* [ ] **🌐 Language Expansion**

    * [ ] Add support for **Chinese** and **English**.

* [ ] **🚀 Model Compatibility**

    * [ ] Support for `V2Proplus`, `V3`, `V4`, and more.

* [ ] **📦 Easy Deployment**

    * [ ] Release **Docker images**.
    * [ ] Provide out-of-the-box **Windows / Linux bundles**.

---
