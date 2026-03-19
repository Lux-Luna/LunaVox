# LunaVox Engineering Notes

## Runtime Contract

LunaVox now targets a single runtime path:
- Talker + Predictor: llama.cpp official binaries from `lib/`
- Speaker encoder / codec decoder: existing GGML C++ implementation

Legacy two-file model layout and transformer fallback are removed.

## Model Layout

`models/` must contain:
- `qwen3_tts_talker.q5_k.gguf`
- `qwen3_tts_predictor.q8_0.gguf`
- `qwen3_tts_speaker_encoder.gguf`
- `qwen3_tts_codec_encoder.gguf`
- `qwen3_tts_codec_decoder.gguf`
- `embeddings/`
- `tokenizer.json`

## Build Contract

- CMake consumes prebuilt runtime from `lib/`.
- `tools/build_manager.py` auto-generates MinGW import libs (`libggml*.a`) from DLLs when needed.
- `ggml/` submodule is intentionally removed; only minimal headers are kept under `third_party/ggml/include`.

## Commands

```bash
python manage.py setup
python manage.py build --backend cpu
ctest --test-dir build-cpu --output-on-failure
```

