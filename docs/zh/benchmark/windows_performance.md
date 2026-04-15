# 📊 Qwen3-TTS LunaVox 性能测试汇总报告 (Windows)

本报告汇总 LunaVox 推理引擎在 Windows 下三种后端配置的性能数据：**Full CPU**、**CUDA 13**、**Universal GPU (Vulkan + DirectML)**。三组数据来自同一个二进制版本、同一套 100 次采样流程。原始逐次数据与完整统计分布在 `benchmark/report.md`。

## 1. 测试环境与配置

- **操作系统**: Windows 11 (AMD64)
- **处理器**: 12th Gen Intel(R) Core(TM) i9-12900K
- **显卡**: NVIDIA GeForce RTX 3090
- **Python**: 3.13.11
- **测试模型**: `models/base_small` (Qwen3-TTS-12Hz-0.6B-Base)
- **语音参考**: `ref/ref_0.6B.json`（预计算 codes，跳过 speaker/codec encoder）
- **测试标准**: 每个后端 5 次预热（不计入统计） + **100 次正式测量**。每次合成同一句 25 词英文：

  > LunaVox is a lightweight on-device text-to-speech engine built on Qwen3, optimized for low-latency streaming synthesis on consumer GPUs.

- **Git commit**: `2fb5887769f2` (dev 分支)
- **生成日期**: 2026-04-15

## 2. 性能总览（100 次平均）

| 模式 | 后端 (LLM / Audio) | 总延迟 (ms) | **TTFB (ms)** | RTF | ×实时 | 峰值 RSS | 峰值 VRAM |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Full CPU** | llama.cpp (CPU) / ORT (CPU) | 8033.4 | 1248.4 | 0.858 | 1.17× | 1215 MB | — |
| **CUDA 13** | llama.cpp (CUDA) / ORT (CUDA) | 2108.1 | 175.2 | 0.213 | 4.69× | 1447 MB | 1364 / 1451 MB |
| **Universal GPU** | llama.cpp (Vulkan) / ORT (DML) | **1351.0** | **194.4** | **0.152** | **6.57×** | **971 MB** | **1028 / 1297 MB** |

**TTFB**（time-to-first-byte）是从合成开始到流式 decoder 输出第一批 PCM 样本的墙钟时间。这是流式调用方实际感受到的延迟 —— 总延迟只对批量调用方（等待 `result.audio` 完成）才重要。

## 3. 总延迟分布（每次合成墙钟时间，毫秒）

| 模式 | mean | p50 | p95 | p99 | min | max | stddev |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Full CPU** | 8033.4 | 7969.5 | 8829.5 | 9582.9 | 7368 | 9774 | 448.7 |
| **CUDA 13** | 2108.1 | 2110.0 | 2135.1 | 2145.1 | 2029 | 2151 | 18.9 |
| **Universal GPU** | 1351.0 | 1341.0 | 1414.2 | 1479.3 | 1301 | 1605 | 42.3 |

## 4. TTFB 分布（毫秒）

| 模式 | mean | p50 | p95 | p99 | min | max | stddev |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Full CPU** | 1248.4 | 1229.5 | 1388.5 | 1507.5 | 1137 | 1561 | 80.2 |
| **CUDA 13** | 175.2 | 175.0 | 182.1 | 184.0 | 164 | 187 | 3.8 |
| **Universal GPU** | 194.4 | 191.0 | 215.2 | 236.1 | 182 | 246 | 11.5 |

## 5. 分阶段延迟分析（每次平均，毫秒）

| 模式 | 分词 | 编码 | 生成 (LLM) | 解码 (codec) |
| :--- | ---: | ---: | ---: | ---: |
| **Full CPU** | 0.4 | 0.0 | 7243.7 | 4105.9 |
| **CUDA 13** | 0.4 | 0.0 | 1997.6 | 613.0 |
| **Universal GPU** | 0.4 | 0.0 | 1263.0 | 403.3 |

`encode` 为 0 是因为 `ref_0.6B.json` 把 speaker / codec embedding 都预算好了，合成路径完全跳过 encoder ONNX session。英文短文本 `tokenize` 低于测量粒度。

## 6. 综合结论

1. **最优配置**: **Universal GPU (Vulkan + DirectML)** 综合表现最佳，平均 **1351 ms 总延迟 / 194 ms TTFB / 6.57× 实时**，峰值 RSS 最低（971 MB），VRAM 稳态只有 1028 MB。流式管道平均 200 ms 内就能出首帧，已经落在用户感知"即时"的区间。三种后端均在修复 `src/main.cpp` 的 `repeat_results` audio 累积后重测，之前 100-run 下每种后端 RSS 都被多算了 ~85-90 MB。
2. **CUDA 13**: TTFB 最低（175.2 ms，且最稳定，stddev 仅 3.8 ms），但总延迟显著更高（2108 ms）——在这套硬件上 CUDA 版 llama.cpp + ORT-CUDA 的稳态 generate/decode 吞吐比 Vulkan+DML 路径慢约 1.6×。峰值 RSS 1447 MB，VRAM 稳态 1364 MB。
3. **CPU 可行性**: i9-12900K 上 Full CPU 构建 RTF 0.858（1.17× 实时），仍然快于实时，适合批量/离线合成。但 ~1.25 s 的 TTFB 不适合交互式流式场景。
4. **资源效率**: 三种后端 RSS 全部控制在 **1.5 GB 以内**、VRAM 稳态控制在 **1.4 GB 以内**，消费级硬件完全可承载。Full CPU 构建不占 GPU 内存 —— 采样器记录的 43 MB 稳态值只是桌面基线噪声。
5. **流式 vs 批量**: 线程化 decoder 管道把 codec 解码和 talker+predictor 生成重叠起来，GPU 后端 TTFB 比总延迟低 6–11×。批量调用方等 `result.audio` 时看到的是总延迟；流式调用方在 TTFB 时就开始消费 PCM。

## 7. 数据复现

```powershell
# 1. 对每种后端：设置目标 provider、clean-build、跑 100 次。
lunavox build libs --platform win_vulkan  # 或 win_cuda13 / win_cpu
lunavox build --clean
python benchmark/run_benchmark.py --backend vulkan   # 或 cuda / cpu

# 2. 三种后端都跑完后：
python benchmark/run_benchmark.py --aggregate
```

逐次原始数据在 `benchmark/results/stats_<backend>.json`；聚合指标在 `benchmark/results/summary_<backend>.json`；完整 9 节报告（含每后端元数据、采样方法说明）在 `benchmark/report.md`。
