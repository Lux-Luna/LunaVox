# LunaVox Runtime Models

This directory contains converted runtime artifacts (ONNX and GGUF) for various Qwen3-TTS model variants. These artifacts are generated from original Hugging Face checkpoints to be used by the LunaVox inference engine.

## Downloading and Setup

### 1. Manual Download Command
You can download specific model sources (raw Hugging Face checkpoints) using the following command:
```bash
python manage.py download --model <name>  # e.g., base_small, base, custom...
python manage.py download --all           # Download all supported models
```

### 2. Automatic Download
The `manage.py convert` (and `setup`) commands are integrated with an automatic downloader. If the required Hugging Face source weights are missing from your local cache, they will be automatically pulled from the Hugging Face Hub (using `snapshot_download`) before the conversion begins.

### 3. Model Cache
Original model weights are cached in the standard Hugging Face directory:
`~/.cache/huggingface/hub/models--Qwen--...`

## Directory Structure

Each model variant subfolder (e.g., `models/base_small/`) typically contains:

- `qwen3_tts_talker.q5_k.gguf`: Quantized Talker model (Llama-based).
- `qwen3_tts_predictor.q8_0.gguf`: Quantized Predictor model (Llama-based).
- `qwen3_tts_codec_encoder.fp16.onnx`: Audio Tokenizer (Mimi-based).
- `qwen3_tts_speaker_encoder.fp16.onnx`: Reference Audio Speaker Encoder.
- `qwen3_tts_decoder.fp16.onnx`: Audio Decoder (Mimi-based).
- `embeddings/`: Projected text and codec embeddings.
- `tokenizer.json`: Hugging Face text tokenizer configuration.

## Available Variants
- `base`: Qwen3-TTS-12Hz-1.7B-Base
- `base_small`: Qwen3-TTS-12Hz-0.6B-Base
- `custom`: Qwen3-TTS-12Hz-1.7B-CustomVoice
- `custom_small`: Qwen3-TTS-12Hz-0.6B-CustomVoice
- `design`: Qwen3-TTS-12Hz-1.7B-VoiceDesign
