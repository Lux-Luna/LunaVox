<div align="center">

# LunaVox: Lightweight Inference Engine for [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

**A high-performance, lightweight inference engine purpose-built for GPT-SoVITS**

[简体中文](./README_zh.md) | [English](./README.md)

</div>

---

**LunaVox** is a lightweight inference engine based on the open-source TTS project [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS). It bundles speech synthesis, ONNX model conversion, an API server, and other conveniences to deliver faster deployment and better ergonomics.

- **Supported model versions:** GPT-SoVITS V2, GPT-SoVITS V2 Pro Plus  
- **Supported languages:** Japanese, Chinese, English

LunaVox preserves the core GPT-SoVITS inference pipeline: multilingual front-ends (e.g., Open JTalk) convert text to phonemes → HuBERT extracts reference audio features → a three-stage T2S stack (Encoder / First-Stage Decoder / Stage Decoder) produces speech tokens → the VITS vocoder renders the final waveform. All of these components—including the Chinese HuBERT and speaker vector models—are provided as ONNX graphs and paired with caching so that pure ONNX Runtime inference remains fast and resource friendly.

---

## Quick Start

### Installation

Install via pip:

```bash
pip install lunavox-tts
```

> **Note:** Installing `pyopenjtalk` may fail because it ships native extensions without prebuilt wheels. On Windows you must install the [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and enable the “Desktop development with C++” workload.

### Quick Tryout

All demo scripts live under `Tutorial/` and will automatically pull missing models and dictionaries on demand.

#### GPT-SoVITS v2 preset (no speaker vector required)

```bash
python Tutorial/v2_quick_tryout/quick_tryout_en.py  # English prompt + output
python Tutorial/v2_quick_tryout/quick_tryout_zh.py  # Chinese prompt + output
python Tutorial/v2_quick_tryout/quick_tryout_ja.py  # Japanese prompt + output
```

#### GPT-SoVITS v2 Pro Plus preset (requires speaker embedding)

```bash
python Tutorial/v2_pro_plus_quick_tryout/quick_tryout_v2proplus_en.py
python Tutorial/v2_pro_plus_quick_tryout/quick_tryout_v2proplus_zh.py
python Tutorial/v2_pro_plus_quick_tryout/quick_tryout_v2proplus_ja.py
```

> The v2 Pro Plus scripts need the ERes2NetV2 speaker embedding model exported to `TTSData/sv/eres2netv2.onnx`; follow the documentation’s export steps before running them.

### Recommended Downloads

For users in mainland China we recommend downloading the required models and dictionaries manually and placing them inside the root `CharacterData`, `TTSData`, and `RoBERTa` directories.

| Source        | Link                                                                                           |
|:--------------|:-----------------------------------------------------------------------------------------------|
| Hugging Face  | [https://huggingface.co/Lux-Luna/LunaVox/tree/main](https://huggingface.co/Lux-Luna/LunaVox)   |

After downloading, point to the assets with environment variables (`os.environ`).

### Optional Dependencies

- **Chinese text pipeline (`lunavox_tts.Chinese.ZhBert`)**  
  Install with `pip install "lunavox-tts[zh]"` to pull in `torch` and `transformers`. Without the extra, Chinese inputs fall back to zero BERT embeddings while Japanese/English inference keeps working.
- **Model conversion utilities (`lunavox.convert_to_onnx`)**  
  Install with `pip install "lunavox-tts[convert]"` to enable the PyTorch-based converter.

### Best Practices for TTS Inference

Example for multilingual synthesis:

```python
import os

# Optional: point to the Chinese HuBERT model. If omitted, the script will try to download it from Hugging Face.
os.environ['HUBERT_MODEL_PATH'] = r"C:\path\to\your\chinese-hubert-base.onnx"

# Optional: point to the Open JTalk dictionary. If omitted, the script will try to download it from GitHub.
os.environ['OPEN_JTALK_DICT_DIR'] = r"C:\path\to\your\open_jtalk_dic_utf_8-1.11"

import lunavox_tts as lunavox

# Step 1: load the character ONNX bundle
lunavox.load_character(
    character_name='<CHARACTER_NAME>',
    onnx_model_dir=r"<PATH_TO_CHARACTER_ONNX_MODEL_DIR>",
)

# Step 2: set the reference audio (voice cloning prompt)
lunavox.set_reference_audio(
    character_name='<CHARACTER_NAME>',
    audio_path=r"<PATH_TO_REFERENCE_AUDIO>",
    audio_text="<REFERENCE_AUDIO_TEXT>",
    audio_language='ja',  # ja / zh / en
)

# Step 3: synthesise speech
lunavox.tts(
    character_name='<CHARACTER_NAME>',
    text="<TEXT_TO_SYNTHESIZE>",
    play=True,
    save_path="<OUTPUT_AUDIO_PATH>",
    language='ja',  # Target language
)

print("Audio generated.")
```

## Performance Baseline (Intel Core i9-12900K)

The following numbers were collected with `benchmark/scripts/tts_benchmark.py` on Windows 11, Python 3.12, 32 GB RAM, and an Intel Core i9-12900K. Each run used 3 warm-up iterations plus 100 measured loops with the fixed text “This is LunaVox speaking English.”

| Model version | Model size (MB) | First packet latency (s) | End-to-end latency (s) | Throughput (iter/s) | RSS delta after load (MB) |
|---|---|---|---|---|---|
| v2 | 683.54 | 1.15 | 1.15 | 0.96 | 2151.46 |
| v2_pro_plus | 1256.14 | 1.38 | 1.38 | 0.76 | 2917.04 |

- Both models achieve a real-time factor of roughly 0.54, producing audio faster than real time.
- Full metrics and per-iteration logs are stored in `benchmark/results/v2_results.json` and `benchmark/results/v2_pro_plus_results.json`.

## Model Conversion

Install the optional converter dependencies first:

```bash
pip install "lunavox-tts[convert]"
```

```python
import lunavox_tts as lunavox

lunavox.convert_to_onnx(
    torch_pth_path=r"<PATH_TO_PTH>",
    torch_ckpt_path=r"<PATH_TO_CKPT>",
    output_dir=r"<OUTPUT_ONNX_DIR>",
)
```

The converter decomposes the GPT-SoVITS pipeline into multiple ONNX graphs: `t2s_encoder_fp32.onnx`, `t2s_first_stage_decoder_fp32.onnx`, `t2s_stage_decoder_fp32.onnx`, and `vits_fp32.onnx`, while bundling the Chinese HuBERT model and speaker vector network. During conversion the original FP16 weights are temporarily promoted to FP32 so that ONNX Runtime delivers stable numerical behavior on CPU-only hosts.

## Runtime Configuration

- `LUNAVOX_ORT_PROVIDERS`: override the preferred ONNX Runtime providers (comma-separated). Example: `CUDAExecutionProvider,CPUExecutionProvider`.
- `LUNAVOX_USE_IO_BINDING=1`: enable experimental IO binding for the vocoder step (can reduce host/device copies when GPU providers are available).

## Launch the FastAPI Server

```python
import os

os.environ['HUBERT_MODEL_PATH'] = r"C:\path\to\your\chinese-hubert-base\chinese-hubert-base.onnx"
# No need to set OPEN_JTALK_DICT_DIR as it's now handled by pyopenjtalk-plus

import lunavox_tts as lunavox

lunavox.start_server(
    host="0.0.0.0",
    port=8000,
    workers=1,
)
```

> See [Tutorial/English/API Server Tutorial.py](./Tutorial/English/API%20Server%20Tutorial.py) for request formats and endpoint details.

## Launch the WebUI

LunaVox includes a Gradio-based web interface for browser-based synthesis.

### Quick start

```bash
# Windows
start_webui.bat

# Or run directly
python WebUI/webui.py
```

### Features

- Character management: automatically scans `CharacterData/character_model`
- Reference audio: upload custom prompts or reuse the included samples
- Text synthesis: enter Japanese text and generate speech with one click
- In-browser playback: listen instantly within the UI
- File saving: generated audio is saved under `Output`

### Usage

1. After launching, the browser opens `http://127.0.0.1:7860`
2. Select a character model (the ONNX bundle loads automatically)
3. Provide a reference audio clip (upload or choose from presets)
4. Enter the text to synthesise
5. Click “Generate” to produce and preview the audio

## Launch the Command-Line Client

```python
import lunavox_tts as lunavox

lunavox.launch_command_line_client()
```

## Roadmap

- [x] Language expansion  
  - [x] Chinese support  
  - [x] English support

- [x] Model compatibility  
  - [x] GPT-SoVITS V2 Pro support  
  - [x] GPT-SoVITS V2 Pro Plus support

- [ ] Performance improvements  
  - [ ] Publish a GPU-oriented build  
  - [ ] Implement text-splitting utilities for long-form synthesis

- [ ] Easier deployment  
  - [ ] Publish a Docker image  
  - [ ] Provide ready-to-use Windows / Linux bundles

---

