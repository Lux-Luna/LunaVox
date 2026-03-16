# Lunavox

C++ inference for [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) using the [GGML](https://github.com/ggml-org/ggml) tensor library.

Runs the full TTS pipeline in pure C++17, including text tokenization, speaker encoding, transformer code generation, and vocoder decoding, without Python or PyTorch at inference time.

## Features

- Pure C++17 inference pipeline
- Zero-shot voice cloning from reference audio
- GGUF model support
- CLI executable and shared-library / C API output
- Flexible CMake build with integrated or external GGML
- Optional Apple CoreML code predictor path

## Build Requirements

- C++17 compiler: GCC 9+, Clang 10+, or MSVC 2022
- CMake 3.14+
- Python 3.10+ for setup and conversion tooling
- Optional: initialized `ggml/` submodule, or an external GGML build/package

## Quickstart

```bash
git clone https://github.com/Lux-Luna/LunaVox
cd LunaVox
git submodule update --init --recursive

# 1. Build the project
python manage.py build --backend auto

# 2. Set up models
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install numpy tqdm safetensors gguf huggingface_hub torch
python manage.py setup

# 3. Generate speech
./build-cpu/qwen3-tts-cli -m models -t "Hello from qwen3-tts.cpp!" -o output.wav
```

On macOS with `--backend auto`, the build directory is typically `build-metal`.

## Build System

Top-level `CMakeLists.txt` supports three GGML integration modes:

1. `GGML_BUILD_DIR=<configured-build-tree>` if provided
2. integrated build from the vendored `ggml/` subdirectory
3. fallback `find_package(ggml CONFIG)`

Preferred build entry point:

```bash
python manage.py build --backend auto
```

Useful variants:

```bash
python manage.py build --backend cpu
python manage.py build --backend cuda
python manage.py build --backend metal
python manage.py build --backend coreml
python manage.py build --backend auto --clean --j 8
```

Direct CMake is also supported:

```bash
cmake -S . -B build-cpu
cmake --build build-cpu --config Release -j 4
```

Use an external GGML build tree when needed:

```bash
cmake -S . -B build -DGGML_BUILD_DIR=/path/to/ggml/build
cmake --build build --config Release -j 4
```

Relevant options:

| Option | Description | Default |
| :--- | :--- | :--- |
| `GGML_BUILD_DIR` | Path to a configured GGML build tree | empty |
| `QWEN3_TTS_TIMING` | Enable detailed timing instrumentation | `OFF` |
| `QWEN3_TTS_COREML` | Enable CoreML code predictor bridge on Apple platforms | `ON` |
| `QWEN3_TTS_USE_OPENMP` | Enable OpenMP when available | `ON` |
| `QWEN3_TTS_NATIVE` | Enable host-specific native optimizations | `OFF` |
| `QWEN3_TTS_COPY_GGML_RUNTIME` | Copy GGML runtime DLLs next to built binaries on Windows | `ON` |

## Model Setup

The `manage.py setup` command handles model download and GGUF conversion.

```bash
python manage.py setup
python manage.py setup --force
python manage.py setup --skip-download
python manage.py setup --tokenizer-type q8_0
```

Required output files in `models/`:

- `qwen3-tts-0.6B-base.gguf`
- `qwen3-tts-tokenizer-f16.gguf` (preferred)
- `qwen3-tts-tokenizer-q8_0.gguf` (runtime fallback when the F16 file is absent)

For voice-cloning smoke tests, also keep:

- `reference/ref-audio.wav`

## Usage

Basic synthesis:

```bash
./build-cpu/qwen3-tts-cli -m models -t "Your text here" -o output.wav
```

Voice cloning:

```bash
./build-cpu/qwen3-tts-cli -m models -t "Synthesize with my voice." -r reference.wav -o cloned.wav
```

Main CLI options:

- `-m, --model`: directory containing GGUF files
- `-t, --text`: text to synthesize
- `-r, --reference`: reference WAV for voice cloning
- `-o, --output`: output WAV path
- `-j, --threads`: number of compute threads
- `-l, --language`: force language (`en`, `ru`, `zh`, `ja`, `ko`, `de`, `fr`, `es`, `it`, `pt`)
- `--no-auto-language`: disable text-based language auto-detection
- `--temperature`: sampling temperature
- `--top-k`: top-k sampling
- `--top-p`: top-p sampling
- `--max-tokens`: maximum audio tokens
- `--repetition-penalty`: repetition penalty

## Outputs

Primary build outputs:

- `qwen3-tts-cli`
- `qwen3tts` shared library
- static component libraries for tokenizer, transformer, encoder, decoder, and pipeline

Installed public headers include:

- `src/qwen3_tts.h`
- `src/qwen3tts_c_api.h`
- component headers under `src/`

## Testing

CTest currently registers:

- `tokenizer_test`
- `encoder_test`
- `transformer_test`
- `decoder_test`
- `tts_template_chinese_test`
- `cli_basic_smoke_test`
- `cli_clone_smoke_test`

Run tests with the build directory you used:

```bash
ctest --test-dir build-cpu --output-on-failure
```

Notes:

- tests are smoke/invariant oriented
- tests do not require the upstream Python project at runtime
- tests do require local model files
- voice-cloning related tests require `reference/ref-audio.wav`

## Acknowledgments

- Qwen Team for the original [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) model
- Georgi Gerganov and contributors for [GGML](https://github.com/ggml-org/ggml)
- WavTokenizer for the underlying vocoder architecture
