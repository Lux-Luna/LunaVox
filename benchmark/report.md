# LunaVox 0.6B — 100-Run Benchmark Report

Generated: 2026-04-15T03:43:53+00:00  
Git commit: `2fb5887769f2`  
Host: Windows 11 (AMD64)  
CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel  
Python: 3.13.11

## Configuration

- **Model**: `models/base_small` (Qwen3-TTS-12Hz-0.6B-Base)
- **Voice reference**: `ref/ref_0.6B.json` (pre-encoded codes)
- **Sample rate**: 24000 Hz
- **Warm-up**: 5 runs (not included in stats)
- **Repeat**: 100 runs per backend
- **First chunk frames**: 8
- **Text** (25 words, fixed across backends):

  > LunaVox is a lightweight on-device text-to-speech engine built on Qwen3, optimized for low-latency streaming synthesis on consumer GPUs.

## 1. Overview

| Backend | Total mean (ms) | TTFB mean (ms) | RTF mean | ×Realtime | Peak RSS (MB) | Peak VRAM (MB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Universal GPU (Vulkan + DirectML)** | 1351.0 | 194.4 | 0.1520 | 6.57× | 971.1 | 1296.8 |
| **CUDA 13** | 2108.1 | 175.2 | 0.2130 | 4.71× | 1446.8 | 1450.8 |
| **Full CPU** | 8033.4 | 1248.4 | 0.8580 | 1.17× | 1214.7 | 839.1 |

## 2. Latency Distribution (total wall time per synth, ms)

| Backend | mean | p50 | p95 | p99 | min | max | stddev |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Universal GPU (Vulkan + DirectML)** | 1351.0 | 1341.0 | 1414.2 | 1479.3 | 1301.0 | 1605.0 | 42.3 |
| **CUDA 13** | 2108.1 | 2110.0 | 2135.1 | 2145.1 | 2029.0 | 2151.0 | 18.9 |
| **Full CPU** | 8033.4 | 7969.5 | 8829.5 | 9582.9 | 7368.0 | 9774.0 | 448.7 |

## 3. Time-to-First-Byte (first audio chunk, ms)

TTFB = wall-clock delay from synth start to the first PCM sample becoming available via the streaming decoder.

| Backend | mean | p50 | p95 | p99 | min | max | stddev |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Universal GPU (Vulkan + DirectML)** | 194.4 | 191.0 | 215.2 | 236.1 | 182.0 | 246.0 | 11.5 |
| **CUDA 13** | 175.2 | 175.0 | 182.1 | 184.0 | 164.0 | 187.0 | 3.8 |
| **Full CPU** | 1248.4 | 1229.5 | 1388.5 | 1507.5 | 1137.0 | 1561.0 | 80.2 |

## 4. Real-Time Factor (RTF = synth_wall / audio_duration)

| Backend | mean | p50 | p95 | ×Realtime |
| :--- | ---: | ---: | ---: | ---: |
| **Universal GPU (Vulkan + DirectML)** | 0.1520 | 0.1510 | 0.1590 | 6.57× |
| **CUDA 13** | 0.2130 | 0.2130 | 0.2150 | 4.71× |
| **Full CPU** | 0.8580 | 0.8510 | 0.9430 | 1.17× |

## 5. Stage Breakdown (per-run mean, ms)

| Backend | Tokenize | Encode | Generate (LLM) | Decode (codec) |
| :--- | ---: | ---: | ---: | ---: |
| **Universal GPU (Vulkan + DirectML)** | 0.4 | 0.0 | 1263.0 | 403.3 |
| **CUDA 13** | 0.4 | 0.0 | 1997.6 | 613.0 |
| **Full CPU** | 0.4 | 0.0 | 7243.7 | 4105.9 |

## 6. Memory Footprint

| Backend | Peak RSS max (MB) | Peak RSS mean (MB) | Peak physical (MB) | Peak VRAM (MB) | Steady VRAM (MB) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Universal GPU (Vulkan + DirectML)** | 971.1 | 970.0 | 971.1 | 1296.8 | 1027.9 |
| **CUDA 13** | 1446.8 | 1446.3 | 1446.8 | 1450.8 | 1364.3 |
| **Full CPU** | 1214.7 | 1184.4 | 1214.7 | 839.1 | 43.5 |

## 7. Throughput

| Backend | Audio duration (s) | Chars/sec mean | Chars/sec p95 | Load time (ms) | In-load warmup (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Universal GPU (Vulkan + DirectML)** | 8.88 | 100.8 | 103.9 | 2562 | 770 |
| **CUDA 13** | 9.92 | 64.5 | 65.5 | 2471 | 687 |
| **Full CPU** | 9.36 | 17.0 | 18.2 | 1683 | 630 |

## 8. Per-Backend Metadata

### Universal GPU (Vulkan + DirectML)
- Git commit: `2fb5887769f2`
- Conda env: `base`
- CLI binary mtime: 2026-04-15T03:43:48+00:00
- Wall elapsed: 145.8 s

### CUDA 13
- Git commit: `2fb5887769f2`
- Conda env: `cuda13`
- CLI binary mtime: 2026-04-15T03:49:31+00:00
- Wall elapsed: 225.09 s

### Full CPU
- Git commit: `2fb5887769f2`
- Conda env: `base`
- CLI binary mtime: 2026-04-15T03:53:53+00:00
- Wall elapsed: 844.52 s

## 9. Notes

- **RTF** = synth wall time / generated audio duration. Lower is better; <1.0 means faster than realtime.
- **TTFB** reflects the streaming pipeline (threaded decoder overlapped with talker+predictor). A batch caller that waits for `result.audio` still observes `total_ms`; a streaming caller starts consuming PCM at `t_first_audio_ms`.
- **VRAM** is sampled externally via pynvml at 100 ms intervals; values are delta above the pre-run baseline, so other GPU workloads on the same device do not skew the numbers.
- Raw per-run records are in `benchmark/results/stats_<backend>.json`; aggregated metrics are in `benchmark/results/summary_<backend>.json`.
