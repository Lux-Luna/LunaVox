# 📊 Qwen3-TTS LunaVox Performance Benchmark Summary (Windows)

This report summarizes the performance of the LunaVox inference engine on Windows across three backend configurations: **Full CPU**, **CUDA 13**, and **Universal GPU (Vulkan + DirectML)**. All three were measured with the same binary version and the same 100-run protocol. Raw per-run records and the full statistical distribution live in `benchmark/report.md`.

## 1. Test Environment & Configuration

- **OS**: Windows 11 (AMD64)
- **CPU**: 12th Gen Intel(R) Core(TM) i9-12900K
- **GPU**: NVIDIA GeForce RTX 3090
- **Python**: 3.13.11
- **Test Model**: `models/base_small` (Qwen3-TTS-12Hz-0.6B-Base)
- **Voice reference**: `ref/ref_0.6B.json` (pre-encoded codes, skips the speaker/codec encoder)
- **Benchmark Standard**: 5 warm-up runs (excluded from stats) followed by **100 measurement runs** per backend. Each run synthesizes the same fixed 25-word English sentence:

  > LunaVox is a lightweight on-device text-to-speech engine built on Qwen3, optimized for low-latency streaming synthesis on consumer GPUs.

- **Git commit**: `2fb5887769f2` (dev branch)
- **Generated**: 2026-04-15

## 2. Performance Overview (100-run mean)

| Mode | Backend (LLM / Audio) | Total (ms) | **TTFB (ms)** | RTF | ×Realtime | Peak RSS | VRAM (steady / peak) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Full CPU** | llama.cpp (CPU) / ORT (CPU) | 8033.4 | 1248.4 | 0.858 | 1.17× | 1215 MB | — |
| **CUDA 13** | llama.cpp (CUDA) / ORT (CUDA) | 2108.1 | 175.2 | 0.213 | 4.69× | 1447 MB | 1364 / 1451 MB |
| **Universal GPU** | llama.cpp (Vulkan) / ORT (DML) | **1351.0** | **194.4** | **0.152** | **6.57×** | **971 MB** | **1028 / 1297 MB** |

**TTFB** (time-to-first-byte) is the wall-clock delay from synth start to the first PCM sample becoming available via the streaming decoder. This is what a streaming caller actually observes — total latency only matters for batch callers that wait for the full audio.

## 3. Latency Distribution (total wall time per synth, ms)

| Mode | mean | p50 | p95 | p99 | min | max | stddev |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Full CPU** | 8033.4 | 7969.5 | 8829.5 | 9582.9 | 7368 | 9774 | 448.7 |
| **CUDA 13** | 2108.1 | 2110.0 | 2135.1 | 2145.1 | 2029 | 2151 | 18.9 |
| **Universal GPU** | 1351.0 | 1341.0 | 1414.2 | 1479.3 | 1301 | 1605 | 42.3 |

## 4. Time-to-First-Byte Distribution (ms)

| Mode | mean | p50 | p95 | p99 | min | max | stddev |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Full CPU** | 1248.4 | 1229.5 | 1388.5 | 1507.5 | 1137 | 1561 | 80.2 |
| **CUDA 13** | 175.2 | 175.0 | 182.1 | 184.0 | 164 | 187 | 3.8 |
| **Universal GPU** | 194.4 | 191.0 | 215.2 | 236.1 | 182 | 246 | 11.5 |

## 5. Stage Latency Analysis (per-run mean, ms)

| Mode | Tokenize | Encode | Generate (LLM) | Decode (codec) |
| :--- | ---: | ---: | ---: | ---: |
| **Full CPU** | 0.4 | 0.0 | 7243.7 | 4105.9 |
| **CUDA 13** | 0.4 | 0.0 | 1997.6 | 613.0 |
| **Universal GPU** | 0.4 | 0.0 | 1263.0 | 403.3 |

`encode` is 0 because `ref_0.6B.json` bundles pre-computed speaker/codec embeddings and bypasses the encoder ONNX sessions. `tokenize` is sub-millisecond for plain-ASCII English.

## 6. Comprehensive Conclusion

1. **Optimal configuration**: **Universal GPU (Vulkan + DirectML)** is the fastest overall at mean **1351 ms total / 194 ms TTFB / 6.57× realtime**, with the lowest peak RSS (971 MB) and steady-state VRAM of just 1028 MB. The streaming pipeline reaches audible output in under 200 ms on average — within the range that users perceive as instant. All three backends were re-measured after fixing the `repeat_results` audio-buffer accumulation in `src/main.cpp` (previously inflating 100-run RSS by ~85–90 MB per backend).
2. **CUDA 13**: Produces the lowest TTFB (175.2 ms, also the most stable at stddev 3.8 ms) but significantly higher total latency (2108 ms), because the CUDA build of llama.cpp and ORT-CUDA has steady-state generate/decode throughput ~1.6× slower than the Vulkan+DML path on this particular hardware. Peak RSS 1447 MB, steady VRAM 1364 MB.
3. **CPU viability**: On an i9-12900K the Full CPU build stays sub-realtime at RTF 0.858 (1.17× realtime). Usable for batch/offline synthesis, but TTFB of ~1.25 s makes it unsuitable for interactive streaming.
4. **Resource efficiency**: All three backends keep RSS under **1.5 GB** and steady-state VRAM under **1.4 GB**, well within the envelope for consumer hardware. The Full CPU build does not touch GPU memory — the VRAM sampler's 43 MB steady reading is baseline desktop noise.
5. **Streaming vs batch**: The threaded decoder pipeline overlaps codec decoding with talker+predictor generation, so TTFB is consistently 6–11× lower than total latency on GPU backends. Batch callers that await `result.audio` still observe the total; streaming callers start consuming PCM at the TTFB mark.

## 7. Reproducing

```powershell
# 1. For each backend: set the target provider, clean-build, and run 100 reps.
lunavox build libs --platform win_vulkan  # or win_cuda13 / win_cpu
lunavox build --clean
python benchmark/run_benchmark.py --backend vulkan   # or cuda / cpu

# 2. After all three are done:
python benchmark/run_benchmark.py --aggregate
```

Raw stats per run are in `benchmark/results/stats_<backend>.json`; aggregated metrics are in `benchmark/results/summary_<backend>.json`; the full 9-section report (including per-backend metadata and notes on sampling methodology) is `benchmark/report.md`.
