# LunaVox (Qwen3-TTS) Inference Performance Report

## Overview
This report summarizes the performance metrics for the `qwen3-tts-cli` engine running in the **CUDA 13** environment. The benchmark consists of 10 consecutive inference runs using the `base_small` model.

### Test Environment
- **Environment:** Conda `cuda13`
- **Model:** `models\base_small` (0.6B)
- **Reference Voice:** `ref\ref_0.6B.json`
- **Inference Hardware:** NVIDIA GPU (Detected RTX 3090, as per system logs)
- **Input Text:** "Hi, this is lunavox speaking English."
- **Instruction:** "Natural and clear speech with a pleasant tone."

## Performance Metrics

| Run | Latency (ms) | RTF (Real-Time Factor) | Peak RAM (MB) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: |
| 1 | 918.00 | 0.3702 | 1425.25 | 4111 |
| 2 | 910.00 | 0.3669 | 1425.07 | 4105 |
| 3 | 1034.00 | 0.3231 | 1428.32 | 4125 |
| 4 | 963.00 | 0.3540 | 1425.49 | 4118 |
| 5 | 1068.00 | 0.3513 | 1428.47 | 4134 |
| 6 | 924.00 | 0.3397 | 1425.99 | 4116 |
| 7 | 896.00 | 0.3394 | 1425.97 | 4111 |
| 8 | 854.00 | 0.3558 | 1425.12 | 4108 |
| 9 | 928.00 | 0.3867 | 1424.88 | 4104 |
| 10 | 940.00 | 0.3672 | 1425.25 | 4092 |

---

### Summary Statistics

| Metric | Average Value |
| :--- | :--- |
| **Average Latency** | **943.50 ms** |
| **Average RTF** | **0.3554** |
| **Average Peak RAM** | **1425.98 MB** |
| **Average Peak VRAM** | **4112.40 MB** |

> **Notes:**
> - Real-Time Factor (RTF) indicates the ratio of processing time to audio duration. Lower values are better.
> - Peak VRAM includes the model weights loaded into GPU memory and the overhead of the LLM engine.
> - Data collected across 10 trials to ensure statistical variance is accounted for.

---
*Report generated on: 2026-03-24*

---

## CPU Only (Fallthrough) Benchmark

### CPU Performance Metrics

| Run | Latency (ms) | RTF | Peak RAM (MB) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: |
| 1 | 3436.00 | 0.9544 | 1522.01 | 2377 |
| 2 | 3092.00 | 0.9663 | 1521.60 | 2360 |
| 3 | 2368.00 | 0.9867 | 1494.49 | 2343 |
| 4 | 2824.00 | 0.9806 | 1520.14 | 2343 |
| 5 | 2932.00 | 0.9905 | 1521.26 | 2323 |
| 6 | 2674.00 | 0.9550 | 1495.57 | 2311 |
| 7 | 3312.00 | 0.9628 | 1522.32 | 2307 |
| 8 | 2591.00 | 0.9526 | 1495.56 | 2307 |
| 9 | 2908.00 | 0.9824 | 1520.98 | 2307 |
| 10 | 2527.00 | 0.9572 | 1494.90 | 2312 |

#### CPU Summary Statistics
| Metric | Average Value |
| :--- | :--- |
| **Average Latency** | **2866.40 ms** |
| **Average RTF** | **0.9688** |
| **Average Peak RAM** | **1510.88 MB** |
| **Average Peak VRAM** | **2329.00 MB** |

---

## 📈 Comparative Analysis (CUDA 13 vs CPU Only)

| Metric | CUDA 13 | CPU Only | Difference |
| :--- | :---: | :---: | :---: |
| **Avg Latency** | 943.50 ms | 2866.40 ms | +203.8% |
| **Avg RTF** | 0.3554 | 0.9688 | +172.6% |
| **Peak RAM** | 1425.98 MB | 1510.88 MB | +5.9% |
| **Peak VRAM** | 4112.40 MB | 2329.00 MB | -43.4% |

### Key Takeaways
- **Compute Efficiency:** The CUDA 13 build is approximately **3x faster** than the CPU-only fallthrough, bringing the Real-Time Factor (RTF) down from nearly 1.0 (real-time) to 0.35 (3x speed).
- **Resource Trade-off:** While the CPU build significantly reduces VRAM usage (as expected), it incurs a substantial latency penalty.
- **Stability:** Both environments showed consistent performance across 10 runs with minimal jitter.

---
*Report updated on: 2026-03-24*
