[**English**](README.md) | [**中文**](docs/zh/README_ZH.md)

# 🌌 LunaVox: High-Performance C++ Inference Engine for Qwen3-TTS

![Version](https://img.shields.io/badge/version-2.2.0-blueviolet?style=for-the-badge)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-0078d7?style=for-the-badge&logo=windows&logoColor=white)
![CoreML](https://img.shields.io/badge/iOS-CoreML-000000?style=for-the-badge&logo=apple&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=c%2B%2B)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)

**LunaVox** is a high-performance C++ inference engine built specifically for **Qwen3-TTS**. A streamlined architecture and deep hardware optimization deliver stable, low-latency TTS for embedded devices, desktop apps, and servers alike.

## 🚀 Key Features

- **Lightweight runtime** — ONNX Runtime + a custom llama.cpp wrapper, no heavy Python required at inference time.
- **Native multi-language** — automatic language detection across Chinese, English, Japanese, Korean, Russian, German, French, Italian, Spanish, Portuguese.
- **Unified `Voice` API** — one `engine.synthesize(text, voice, params)` covers Base, Voice Cloning, Custom Voice, and Voice Design.
- **HTTP + WebSocket serving** (`lunavox serve`): FastAPI app with `POST /v1/synth` and streaming `WS /v1/stream` — see [serve guide](docs/en/guide/serve.md).
- **Desktop GUI** (`lunavox gui`): customtkinter app (Synthesize / Library / Settings) driving the same in-process engine.
- **Profile-driven CLI** — layered `~/.lunavox/config.toml` / env / flag precedence so `lunavox --profile quality synth …` is a one-liner.
- **Cross-platform hardware acceleration** — CUDA (NVIDIA), CoreML/Metal (Apple), DML (DirectX 12), and Vulkan.

## 🛠️ Requirements

- **Windows** 10/11 (VS 2022/2025), **Linux** Ubuntu 22.04+ (GCC ≥ 9.0), or **macOS** 12+ on Apple Silicon
- **CMake 3.16+** (Ninja recommended) and a compatible C++17 compiler
- **Python 3.10+** for the CLI and conversion toolchain

## 📊 Performance

| Configuration | **TTFB (ms)** | RTF | Peak RAM | VRAM | Speedup |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Official PyTorch Baseline (CPU) | — | 5.066 | 5.06 GB | — | 1.00× |
| Official PyTorch Baseline (GPU) | — | 3.788 | 1.59 GB | 2.29 GB | 1.34× |
| **LunaVox (Full CPU)** | 1248 | 0.858 | 1.19 GB | — | 5.90× |
| **LunaVox (CUDA 13)** | 175 | 0.213 | 1.41 GB | 1.33 GB | 23.78× |
| **LunaVox (Vulkan + DML)** | **194** | **0.152** | **0.97 GB** | 1.00 GB | **33.33×** |

Model `Qwen3-TTS-12Hz-0.6B-Base` with `ref/ref_0.6B.json` cloning on Intel i9-12900K + RTX 3090 / Windows 11 — 5 warmup + 100 measurement runs on a fixed 25-word English sentence. Full per-run distribution in [`benchmark/report.md`](benchmark/report.md); detailed analysis in [Windows performance report](docs/en/benchmark/windows_performance.md).

## 📦 Install & quick start

```powershell
pip install lunavox               # core CLI + GUI + HTTP/WebSocket server (default)
pip install "lunavox[convert]"    # + source → GGUF conversion toolchain (heavy, optional)
```

```powershell
lunavox bootstrap                 # one-key: pull model + libs + build + smoke test
```

Prefer step-by-step? Run `lunavox model pull`, `lunavox build libs`, then `lunavox build --clean`. For CUDA see [CUDA on Windows](docs/en/install/cuda_windows.md). Full command reference: **[CLI manual](docs/en/guide/cli_reference.md)**.

## 🎙️ Synthesis

```bash
lunavox synth "Hello from LunaVox." -o out.wav                       # base voice
lunavox synth "…" --voice clone  --ref ref/ref_0.6B.json -o out.wav  # voice cloning
lunavox synth "…" --voice custom --speaker Vivian --instruct "…"     # catalog speaker
lunavox synth "…" --voice design --instruct "A warm, calm narrator." # text-to-voice design
lunavox gui                                                          # desktop GUI
```

The standalone `./build/lunavox-cli` works the same way in Python-free environments. Full mode documentation: **[usage tutorial](docs/en/guide/usage_tutorial.md)**.

### Embedded Python

```python
from lunavox.runtime import Engine, SynthesisParams, Voice

with Engine("models/base_small") as engine:
    result = engine.synthesize(
        "Hello from LunaVox.",
        voice=Voice.clone_file("ref/ref_0.6B.json"),
        params=SynthesisParams(temperature=0.7),
    )
    print(f"RTF {result.stats.rtf:.3f}")  # result.audio is a float32 [-1, 1] mono array
```

## 📈 Observability

- `--stats-json report.json` — RTF + memory breakdown per synthesis
- `logs/latest.log` — build and runtime output
- `-j N` — CPU thread count (default 4)

## 📖 Documentation

Full bilingual docs — guide, CLI reference, technical details, benchmarks, Python API — are published at **https://lux-luna.github.io/LunaVox/**. Local preview:

```bash
pip install -e ".[dev]"
mkdocs serve
```

Release history: **[CHANGELOG.md](CHANGELOG.md)**.

## 🙏 Acknowledgements

- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** — base weights and architecture
- **[onnxruntime](https://github.com/microsoft/onnxruntime)** — audio decoding backend
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — LLM sequence prediction core
