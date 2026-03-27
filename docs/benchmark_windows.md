# Qwen3-TTS LunaVox 性能测试汇总报告 (Windows)

本报告汇总了 LunaVox 推理引擎在 Windows 环境下，针对 **CPU**、**CUDA** 及 **Universal GPU (Vulkan+DML)** 三种后端配置的性能测试数据。

## 1. 测试环境与配置
- **操作系统**: Windows 11
- **处理器 (CPU)**: 12th Gen Intel(R) Core(TM) i9-12900K
- **显卡 (GPU)**: NVIDIA GeForce RTX 3090
- **Conda 环境**: `cuda13`
- **测试模型**: `models/base_small` (0.6B)
- **测试文本**: "Hi, this is lunavox speaking English."
- **测试标准**: 3 次预热 + 10 次正式测试取平均值

## 2. 性能表现总览

| 优化模式 | 后端详情 (LLM / Audio) | 平均延迟 (ms) | 平均 RTF | 内存占用 | 显存占用 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Full CPU** | llama.cpp (Native) / ORT (CPU) | 3592.8 | 1.1515 | 1.06 GB | 0.00 GB |
| **CUDA 13** | llama.cpp (CUDA) / ORT (CUDA) | 751.8 | 0.2540 | 1.39 GB | 1.30 GB |
| **Universal GPU** | llama.cpp (Vulkan) / ORT (DML) | **625.4** | **0.2057** | **0.91 GB** | **1.05 GB** |

## 3. 分阶段延迟分析 (平均值)

| 模式 | 分词 (ms) | 编码 (ms) | 生成 (ms) | 解码 (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **CUDA 13** | 0.3 | 0.0 | 453.6 | 297.9 |
| **Universal GPU** | 0.1 | 0.0 | 399.6 | 225.4 |
| **Full CPU** | 0.3 | 0.0 | 2223.7 | 1368.7 |

## 4. 综合结论
1. **最优选配置**: **Universal GPU (Vulkan + DirectML)** 在本次测试中表现最佳，其 RTF 达到 **0.206** (约 4.8 倍实时)，且内存与显存消耗最低。这表明在 Windows 开发环境下，该方案拥有极高的执行效率。
2. **CUDA 表现**: 原生 CUDA 后端表现稳定，但在此特定硬件与小模型（0.6B）配置下，其调度开销略高于 Vulkan/DML 组合。
3. **CPU 可行性**: 在 i9-12900K 等高性能 CPU 上，RTF 接近实时 (1.15)，足以应对非实时或背景合成任务，展现了良好的通用性。
4. **资源优势**: LunaVox 引擎在所有模式下内存占用均控制在 **1.4 GB 以内**，显存控制在 **1.3 GB 以内**，非常适合低资源部署。
