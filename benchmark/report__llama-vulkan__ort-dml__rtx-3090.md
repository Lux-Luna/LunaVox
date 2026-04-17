# LunaVox 0.6B Benchmark — llama.cpp/vulkan + ORT/DirectML on NVIDIA GeForce RTX 3090

Generated: 2026-04-17T01:18:30+00:00  
Git commit: `6c53207ddee9`  
Run tag: `llama-vulkan__ort-dml__rtx-3090`

## Active Backend Combo

| Component | Backend | Version | Provider / EP | Build platform |
| :--- | :--- | :--- | :--- | :--- |
| llama.cpp (talker + predictor GGUF) | `vulkan` | `b8470` | — | windows |
| ONNX Runtime (codec encoder + decoder) | onnxruntime | `1.24.4` | `DmlExecutionProvider` (DirectML) | windows |

## Host

- **CPU**: 12th Gen Intel(R) Core(TM) i9-12900K
- **GPU**: NVIDIA GeForce RTX 3090
- **OS**: Windows 11 (AMD64)
- **Python**: 3.13.11
- **Conda env**: `base`
- **CLI binary**: `build/lunavox-cli.exe` (mtime: 2026-04-17T00:22:37+00:00)

## Configuration

- **Model**: `models/base_small` (Qwen3-TTS-12Hz-0.6B-Base)
- **Voice reference**: `ref/ref_0.6B.json` (pre-encoded codes)
- **Sample rate**: 24000 Hz
- **Warm-up**: 5 runs (excluded from stats)
- **Repeat**: 100 runs
- **First chunk frames**: 8
- **Text** (25 words):

  > LunaVox is a lightweight on-device text-to-speech engine built on Qwen3, optimized for low-latency streaming synthesis on consumer GPUs.

## 1. Latency (total wall time per synth, ms)

| Metric | mean | p50 | p95 | p99 | min | max | stddev |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| total_ms | 1221.7 | 1224.0 | 1242.1 | 1248.0 | 1182.0 | 1248.0 | 14.5 |
| ttfb_ms | 177.8 | 174.5 | 197.0 | 198.0 | 170.0 | 202.0 | 8.0 |

## 2. Real-Time Factor

- **Mean RTF**: 0.1380 → **7.27× realtime**
- **p50 / p95 / p99**: 0.1380 / 0.1400 / 0.1410

## 3. Stage Breakdown (per-run mean, ms)

| Tokenize | Encode | Generate (LLM) | Decode (codec) |
| ---: | ---: | ---: | ---: |
| 0.3 | 0.0 | 1138.5 | 376.3 |

## 4. Memory Footprint

| Peak RSS max (MB) | Peak RSS mean (MB) | Peak physical (MB) | Peak VRAM Δ (MB) | Steady VRAM Δ (MB) |
| ---: | ---: | ---: | ---: | ---: |
| 995.1 | 993.2 | 995.1 | 1245.8 | 1173.8 |

## 5. Throughput

- Audio duration mean: **8.88 s**
- Chars/sec mean / p95: **111.3 / 113.7**
- Load time: 1562 ms (in-load warm-up: 330 ms)
- Wall elapsed (load + warmup + repeats): 130.76 s

## Notes

- **RTF** = synth wall time / generated audio duration. Lower is better; <1.0 means faster than realtime.
- **TTFB** is the wall-clock delay from synth start to the first PCM sample becoming available via the streaming decoder.
- **VRAM Δ** is sampled externally via NVML at 100 ms intervals, minus the pre-run baseline; only meaningful when an NVIDIA GPU is present.
- Raw per-run records: `benchmark/results/stats__llama-vulkan__ort-dml__rtx-3090.json`
- Aggregated metrics: `benchmark/results/summary__llama-vulkan__ort-dml__rtx-3090.json`
