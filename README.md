# LunaVox

LunaVox is a C++ inference runtime for Qwen3-TTS.

Current runtime architecture:
- Talker + Predictor: `llama.cpp` official runtime from `./lib`
- Speaker encoder / codec decoder: existing GGML C++ logic
- Model layout: 5 GGUF files + `embeddings/` + `tokenizer.json`

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
- `qwen3_tts_speaker_encoder.gguf`
- `qwen3_tts_codec_encoder.gguf`
- `qwen3_tts_codec_decoder.gguf`
- `embeddings/text_embedding_projected.npy`
- `embeddings/codec_embedding_0.npy` ... `embeddings/codec_embedding_15.npy`
- optional `embeddings/proj_weight.npy`, `embeddings/proj_bias.npy`
- `tokenizer.json`

Convert only (reuse already-downloaded model assets):

```bash
python manage.py convert --force
```

## Build

```bash
python manage.py build --backend cpu
```

`tools/build_manager.py` will:
- auto-generate MinGW import libs from `lib/ggml*.dll` on Windows
- configure CMake to use prebuilt runtime in `./lib`
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
