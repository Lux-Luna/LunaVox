# LunaVox

LunaVox is a C++ inference runtime for Qwen3-TTS.

Current runtime architecture:
- Talker + Predictor: `llama.cpp` official runtime from `./lib`
- Speaker encoder / codec encoder / decoder: ONNX Runtime (CPU)
- Model layout: 2 GGUF + 3 ONNX + `embeddings/` + `tokenizer.json`

## Requirements

- CMake 3.14+
- Python 3.10+
- A C++ toolchain (on Windows, `m2w64-toolchain` in conda is supported)
- Prebuilt runtime files in `./lib` (already expected by this repo)
- Python dependencies:
  - `pip install -r requirements.txt`

## Model Setup

```bash
python manage.py setup
```

Generated model layout (`models/base_small/`):
- `qwen3_tts_talker.q5_k.gguf`
- `qwen3_tts_predictor.q8_0.gguf`
- `qwen3_tts_speaker_encoder.fp16.onnx`
- `qwen3_tts_codec_encoder.fp16.onnx`
- `qwen3_tts_decoder.fp16.onnx`
- `embeddings/text_embedding_projected.npy`
- `embeddings/codec_embedding_0.npy` ... `embeddings/codec_embedding_15.npy`
- optional `embeddings/proj_weight.npy`, `embeddings/proj_bias.npy`
- `tokenizer.json`

Convert only (reuse already-downloaded model assets):

```bash
python manage.py convert --force
```

If local ONNX export is unavailable in your environment, you can download prebuilt ONNX files:

```bash
python manage.py setup --skip-download --onnx-prebuilt-repo cgisky/qwen3-tts-custom-gguf --onnx-prebuilt-only
```

## Build

```bash
python manage.py build --backend cpu
```

`tools/build_manager.py` will:
- configure CMake to use prebuilt llama runtime in `./lib`
- configure CMake to use ONNX Runtime SDK in `./lib/onnxruntime`
- build into `build-cpu/`

## Run

```bash
./build-cpu/qwen3-tts-cli -m models/base_small -t "Hello from LunaVox" -o output.wav
```

Voice cloning:

```bash
./build-cpu/qwen3-tts-cli -m models/base_small -t "Hello" -r output.wav -o cloned.wav
```

## Tests

```bash
ctest --test-dir build-cpu -R "cli_.*smoke.*" --output-on-failure
```
