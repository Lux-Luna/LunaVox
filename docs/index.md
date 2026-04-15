# LunaVox

> High-performance C++ inference engine for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

LunaVox is a specialized inference engine that takes a single TTS model family and pushes it to the theoretical limit on commodity hardware. Through a layered architecture — C++ core, ctypes-bound Python orchestration, desktop GUI shell — and deep integration with ONNX Runtime + llama.cpp, it delivers **sub-200 ms TTFB** and **RTF 0.152** on an RTX 3090 for the 0.6B model.

## Quick links

<div class="grid cards" markdown>

- :material-rocket-launch: **[Usage Tutorial](en/guide/usage_tutorial.md)**
  Walk through Base, Voice Cloning, Custom Voice, and Voice Design modes.

- :material-console: **[CLI Reference](en/guide/cli_reference.md)**
  Every flag of `lunavox-cli` with examples.

- :material-chart-bar: **[Benchmarks](en/benchmark/windows_performance.md)**
  Reproducible 100-run perf sweeps across CUDA / Vulkan+DML / CPU.

- :material-code-braces: **[Python API](en/api/runtime.md)**
  Embed LunaVox in your own apps via `lunavox.runtime.Engine`.

- :material-cog: **[Synthesis Pathway](en/technical/synthesis_pathway.md)**
  How text becomes audio end-to-end inside the C++ engine.

- :material-file-chart: **[Stats Schema](en/technical/stats_schema.md)**
  Contract between `--stats-json` and downstream consumers.

</div>

## Performance snapshot

| Configuration | TTFB | RTF | Peak RAM | Speedup vs. PyTorch |
| :--- | ---: | ---: | ---: | ---: |
| Official PyTorch (CPU) | — | 5.066 | 5.06 GB | 1.00× |
| Official PyTorch (GPU) | — | 3.788 | 1.59 GB | 1.34× |
| **LunaVox (Full CPU)** | 1248 ms | 0.858 | 1.19 GB | 5.90× |
| **LunaVox (CUDA 13)** | 175 ms | 0.213 | 1.41 GB | 23.78× |
| **LunaVox (Vulkan + DML)** | **194 ms** | **0.152** | **0.97 GB** | **33.33×** |

Measured on an Intel i9-12900K + RTX 3090 with `Qwen3-TTS-12Hz-0.6B-Base`, voice cloning mode, 100 measurement runs per backend after 5 warmups. Full distribution and raw stats in [`benchmark/report.md`](https://github.com/Lux-Luna/LunaVox/blob/main/benchmark/report.md).

## License

MIT. See [LICENSE](https://github.com/Lux-Luna/LunaVox/blob/main/LICENSE).
