# 📊 Qwen3-TTS LunaVox Performance Benchmark Summary (Windows)

This report summarizes the performance data of the LunaVox inference engine in Windows environment across three backend configurations: **CPU**, **CUDA**, and **Universal GPU (Vulkan+DML)**.

## 1. Test Environment & Configuration
- **OS**: Windows 11
- **CPU**: 12th Gen Intel(R) Core(TM) i9-12900K
- **GPU**: NVIDIA GeForce RTX 3090
- **Conda Env**: `cuda13`
- **Test Model**: `models/base_small` (0.6B)
- **Test Text**: "Hi, this is lunavox speaking English."
- **Benchmark Standard**: Average of 10 runs after 3 warmup runs.

## 2. Performance Overview

| Mode | Backend Details (LLM / Audio) | Avg Latency (ms) | Avg RTF | RAM Usage | VRAM Usage |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Full CPU** | llama.cpp (Native) / ORT (CPU) | 3592.8 | 1.1515 | 1.06 GB | 0.00 GB |
| **CUDA 13** | llama.cpp (CUDA) / ORT (CUDA) | 751.8 | 0.2540 | 1.39 GB | 1.30 GB |
| **Universal GPU** | llama.cpp (Vulkan) / ORT (DML) | **625.4** | **0.2057** | **0.91 GB** | **1.05 GB** |

## 3. Stage Latency Analysis (Average)

| Mode | Tokenizer (ms) | Encoder (ms) | Generation (ms) | Decoding (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **CUDA 13** | 0.3 | 0.0 | 453.6 | 297.9 |
| **Universal GPU** | 0.1 | 0.0 | 399.6 | 225.4 |
| **Full CPU** | 0.3 | 0.0 | 2223.7 | 1368.7 |

## 4. Comprehensive Conclusion
1.  **Optimal Configuration**: **Universal GPU (Vulkan + DirectML)** performed best in this test, achieving an RTF of **0.206** (~4.8x real-time) with the lowest RAM and VRAM consumption. This indicates high execution efficiency in Windows development environments.
2.  **CUDA Performance**: Native CUDA backend is stable, but for this specific hardware and small model (0.6B), its scheduling overhead is slightly higher than the Vulkan/DML combination.
3.  **CPU Viability**: On high-performance CPUs like i9-12900K, RTF is close to real-time (1.15), sufficient for non-real-time or background synthesis tasks.
4.  **Resource Efficiency**: LunaVox keeps RAM usage under **1.4 GB** and VRAM usage under **1.3 GB** in all modes, making it suitable for low-resource deployment.
