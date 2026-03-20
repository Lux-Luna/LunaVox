# LunaVox

LunaVox is a C++ inference runtime for Qwen3-TTS.

Current runtime architecture:
- Talker + Predictor: `llama.cpp` official runtime from `./lib`
- Speaker encoder / codec encoder / decoder: ONNX Runtime (CPU)
- Model layout: 2 GGUF + 3 ONNX + `embeddings/` + `tokenizer.json`

## Requirements

- CMake 3.16+
- Python 3.10+
- A C++ toolchain (on Windows, conda toolchains are supported)
- Prebuilt llama runtime files in `./lib`
- ONNX Runtime SDK in `./lib/onnxruntime` (`include/` + `lib/`)
- Python dependencies:
  - `pip install -r requirements.txt`
  - `pip install -r requirements-convert-onnx.txt` (needed for local ONNX export)

## Important Constraints

- Inference has no Python dependency.
- ONNX artifacts are exported locally only.
- Online prebuilt ONNX download is intentionally disabled.

## One-Command Bootstrap (recommended)

```bash
conda run -n lunavox python manage.py bootstrap --backend cpu
```

Optional:

```bash
conda run -n lunavox python manage.py bootstrap --backend cpu --timeout-sec 170 --skip-quant
```

This command runs:
1. preflight checks (`git safe.directory`, conda env, conversion deps, ORT SDK)
2. local model conversion/export
3. CMake configure + build
4. built CLI verification (`--help`)

## Manual Steps

Run preflight only:

```bash
conda run -n lunavox python manage.py preflight
```

Run setup (download + conversion):

```bash
conda run -n lunavox python manage.py setup
```

Convert only (reuse local model assets):

```bash
conda run -n lunavox python manage.py convert --force
```

Build only:

```bash
conda run -n lunavox python manage.py build --backend cpu --timeout-sec 170
```

Run alignment/performance compare (base + clone):

```bash
conda run -n lunavox python manage.py compare --mode both --timeout-sec 170 --report-out logs/compare/latest_compare_report.json
```

Notes:
- `compare` writes per-stage command logs under `logs/manage/` and detailed artifacts under `logs/compare/_artifacts/`.
- `--qwen-model-dir` now defaults to `models/base_small` to avoid stale external paths.
- `--strict-qwen-model` defaults to `true`: if Qwen model artifacts are incomplete, compare fails immediately (no fallback).
- `--qwen-clone-anchor-seconds` controls **Qwen side only** (default `0.0`); clone compare also writes a fixed anchor matrix (`0.0` and `1.0`) for long-WAV root-cause isolation.
- For clone compare, you can pass separate references:
  - `--reference-audio` for lunavox (WAV)
  - `--qwen-reference-audio` for Qwen side (WAV or JSON)
- Compare defaults now include fixed seeds (`--seed`, `--predictor-seed`) for reproducibility.

## ONNX Export Diagnostics

- ONNX export runs by stage: `codec_encoder`, `speaker_encoder`, `decoder` (+ optional `quantize`)
- ONNX export completion is followed by a local ORT validation stage.
- Per-stage timeout default: `170s` (configurable by `--timeout-sec`)
- Stage logs:
  - `logs/convert_onnx/codec_encoder.log`
  - `logs/convert_onnx/speaker_encoder.log`
  - `logs/convert_onnx/decoder.log`
  - `logs/convert_onnx/quantize.log` (if enabled)
  - `logs/convert_onnx/validate.log`

## Runtime Model Layout (`models/base_small/`)

- `qwen3_tts_talker.q5_k.gguf`
- `qwen3_tts_predictor.q8_0.gguf`
- `qwen3_tts_speaker_encoder.fp16.onnx`
- `qwen3_tts_codec_encoder.fp16.onnx`
- `qwen3_tts_decoder.fp16.onnx`
- `embeddings/text_embedding_projected.npy`
- `embeddings/codec_embedding_0.npy` ... `embeddings/codec_embedding_15.npy`
- optional `embeddings/proj_weight.npy`, `embeddings/proj_bias.npy`
- `tokenizer.json`

## Run

```bash
./build-cpu/qwen3-tts-cli -m models/base_small -t "Hello from LunaVox" -o output.wav
```

Voice cloning:

```bash
./build-cpu/qwen3-tts-cli -m models/base_small -t "Hello" -r output.wav -o cloned.wav
```

Runtime notes:
- Clone reference audio is full-length by default (no forced 1-second truncation).
- Optional manual cap is supported via env var `QWEN3_TTS_CLONE_MAX_REF_SAMPLES`.
- Clone synthesis defaults to x-vector-only prompting for stability. Enable ICL fusion (`ref_codes + spk_emb`) explicitly with `QWEN3_TTS_CLONE_USE_ICL=1`.
- ORT logs are error-only by default; pass `--ort-debug-log` to surface warning logs during debugging.

Export timing/memory JSON:

```bash
./build-cpu/qwen3-tts-cli -m models/base_small -t "Hello" --stats-json logs/compare/hello_stats.json
```

## Tests

```bash
ctest --test-dir build-cpu -R cli_help_smoke --output-on-failure
```
