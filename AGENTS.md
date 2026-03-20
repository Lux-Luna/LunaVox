# LunaVox Engineering Notes

## Runtime Contract

LunaVox now targets a single runtime path:
- Talker + Predictor: llama.cpp official binaries from `lib/`
- Speaker encoder / codec encoder / decoder: ONNX Runtime C++ inference

Legacy two-file model layout and transformer fallback are removed.

## Model Layout

`models/` must contain:
- `qwen3_tts_talker.q5_k.gguf`
- `qwen3_tts_predictor.q8_0.gguf`
- `qwen3_tts_speaker_encoder.fp16.onnx`
- `qwen3_tts_codec_encoder.fp16.onnx`
- `qwen3_tts_decoder.fp16.onnx`
- `embeddings/`
- `tokenizer.json`

## Build Contract

- CMake consumes prebuilt runtime from `lib/`.
- CMake consumes ONNX Runtime SDK from `lib/onnxruntime`.
- No project dependency on `third_party/ggml` (removed).

## Commands

```bash
python manage.py bootstrap --backend cpu
python manage.py preflight
ctest --test-dir build-cpu -R cli_help_smoke --output-on-failure
```
