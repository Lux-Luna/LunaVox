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

### Minimal Stable Windows Combo

- OS: Windows 10/11 x64
- Compiler toolchain: Visual Studio Build Tools 2022 (MSVC x64 + Windows SDK)
- Conda env packages: `python=3.10`, `cmake`, `ninja`, `pip`
- Recommended clean env setup:

```powershell
conda env remove -n lunavox -y
conda create -n lunavox python=3.10 pip cmake ninja -y
conda activate lunavox
```

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

Recommended stable CPU build on Windows:

```powershell
python manage.py build --backend cpu
```

Notes:

- `tools/build_manager.py` auto-detects and reuses `ggml/build_cpu` when available.
- The build manager performs generator health checks on Windows and falls back safely if a cached generator is unusable.
- With a healthy environment and reused GGML build tree, CPU build should normally finish within 3 minutes.

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
- `--backend`: global backend policy (`auto`, `gpu`, `igpu`, `accel`, `cpu`)
- `--backend-speaker`: override Speaker Encoder backend
- `--backend-transformer`: override Talker/Code Predictor backend
- `--backend-talker`: Talker backend override
- `--backend-code-predictor`: Code Predictor backend override
- `--backend-decoder`: override Codec Decoder backend
- `--streaming-decode`: enable experimental generate/decode overlap
- `--decode-chunk-frames`: decoder chunk size (frames) for streaming mode
- `--streaming-max-queued-chunks`: max queued decode chunks in streaming mode
- `--streaming-decode-batch-chunks`: decode worker batch size (chunks) in streaming mode

Threading notes:

- `--threads 0` (or omitted) now uses an automatic thread policy tuned for throughput.
- Set `QWEN3_TTS_THREADS=<n>` to override the auto policy for deployment or reproducible benchmarks.

Component backend notes:

- Current runtime grouping is:
  - `Speaker Encoder` -> `AudioTokenizerEncoder`
  - `Talker` -> `TTSTransformer` Talker engine
  - `Code Predictor` -> `TTSTransformer` Code Predictor engine
  - `Codec Decoder` -> `AudioTokenizerDecoder`
  - `Codec Encoder` is not used in the current inference path
- Component-level environment overrides:
  - `QWEN3_TTS_BACKEND_SPEAKER_ENCODER`
  - `QWEN3_TTS_BACKEND_TRANSFORMER`
  - `QWEN3_TTS_BACKEND_TALKER`
  - `QWEN3_TTS_BACKEND_CODE_PREDICTOR`
  - `QWEN3_TTS_BACKEND_CODEC_DECODER`

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

## Performance Benchmark

Use the built-in benchmark helper to run reproducible CPU/GPU end-to-end checks:

```bash
python tools/perf_benchmark.py --max-tokens 256
```

Component override example:

```bash
python tools/perf_benchmark.py --max-tokens 256 --backend-talker gpu --backend-code-predictor gpu --backend-decoder cpu
```

Split-engine strategy example (GPU Talker + CPU Code Predictor):

```bash
python tools/perf_benchmark.py --max-tokens 256 --backend-talker gpu --backend-code-predictor cpu --backend-decoder cpu
```

Windows tip:

- If GPU benchmark runs fail under `conda run -n <env> ...` with a non-zero process exit, run the same command from a regular shell Python to avoid runtime DLL path conflicts.

Outputs:

- `perf/summary.json`
- `perf/summary.md`
- `perf/cpu.log`, `perf/gpu.log` (when GPU case is enabled)

Decoder stage timing (debug/profiling):

- Set `QWEN3_TTS_DECODER_TIMING=1` to print decoder sub-stage timings (`graph-build`, `graph-alloc`, `compute`, etc.).

Streaming decode controls (experimental):

- CLI: `--streaming-decode --decode-chunk-frames 32`
- Env: `QWEN3_TTS_STREAMING_DECODE=1`
- Optional queue tuning: `QWEN3_TTS_STREAMING_MAX_QUEUED_CHUNKS=<n>`
- Optional decode batch tuning: `--streaming-decode-batch-chunks <n>`

Parameter sweep helper (threads/chunk/queue/batch):

```bash
python tools/perf_sweep.py --max-tokens 128 --threads 0,8,12 --chunks 16,32,64 --queues 2,4 --batches 1,2,4 --out-root perf/phaseD_sweep128
```

Current optimization roadmap and bottleneck analysis are tracked in:

- `docs/performance_blueprint.md`
- `docs/phaseB_engine_split_report_2026-03-17.md`
- `docs/phaseC_streaming_decode_report_2026-03-17.md`
- `docs/phaseD_architecture_optimization_report_2026-03-17.md`

## Acknowledgments

- Qwen Team for the original [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) model
- Georgi Gerganov and contributors for [GGML](https://github.com/ggml-org/ggml)
- WavTokenizer for the underlying vocoder architecture
