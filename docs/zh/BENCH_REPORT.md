# LunaVox 性能基准报告 — base_small (0.6B) + ref_0.6B.json

> 日期：2026-04-13
> 模型：`Qwen3-TTS-12Hz-0.6B-Base` (`models/base_small`)
> 参考：`ref/ref_0.6B.json`（预计算 speaker embedding，跳过 encoder）
> 版本：LunaVox 2.1.6（Phase A–F 重构完成后）

---

## 1. 测试环境

| 项目 | 值 |
|---|---|
| **OS** | Windows 11 |
| **CPU** | Intel i9-12900K (16P/24T) |
| **GPU** | NVIDIA GeForce RTX 3090 |
| **Python** | 3.13.11 |
| **ONNX Runtime Provider** | `DmlExecutionProvider`（DirectML → RTX 3090） |
| **Llama.cpp Backend** | `vulkan`（NVIDIA GeForce RTX 3090） |
| **合成采样率** | 24000 Hz |
| **线程数** | 4 |

### 调用路径

通过 **Python ctypes 绑定** (`lunavox.runtime.Engine`) 直接调用 `build/lunavox.dll` 的 C ABI，不经 CLI 也不经 subprocess。所有统计数据来自 `LunavoxAudio` 结构体内嵌的 timing 字段（无 stdout 解析）。

```python
from lunavox.runtime import Engine
with Engine("models/base_small", n_threads=4) as eng:
    result = eng.synthesize_with_voice_file(text, "ref/ref_0.6B.json")
    # result.stats: t_tokenize_ms / t_encode_ms / t_generate_ms / t_decode_ms / t_total_ms / rtf / ...
```

---

## 2. 测试方法

- **冷加载**：一次 `Engine.__init__` → 测量 `wall_create_ms` + `last_load_ms` + `last_warmup_ms`
- **循环合成**：10 次连续 `synthesize_with_voice_file`，覆盖 5 种不同长度的英文句子（循环复用）
- **进程级采样**：单独线程每 100 ms 采样 `psutil.Process.memory_info().rss` 和 `pynvml.nvmlDeviceGetMemoryInfo().used`
- **引擎级 stats**：每次 synth 结束从 `LunavoxAudio` 读出 tokenize/encode/generate/decode/total 毫秒 + RSS peak + audio 时长 + RTF
- **唯一进程**：10 次合成共享一个 `Engine` 句柄，不重新加载模型

### 测试文本

| # | 长度 | 内容 |
|---|---:|---|
| 1 | 67 | LunaVox is a lightweight Qwen3-TTS inference engine written in C++. |
| 2 | 78 | This benchmark measures cold load, warmup, and steady-state synthesis latency. |
| 3 | 93 | Real-time factor below one point zero means the engine generates audio faster than real time. |
| 4 | 88 | We are running ten iterations to separate first-run effects from warm cache performance. |
| 5 | 79 | The Python ctypes binding returns structured statistics without parsing stdout. |

10 次迭代 = 2 轮完整的 5 句循环。

---

## 3. 冷启动开销

| 阶段 | 时间 |
|---|---:|
| `Engine.__init__` 墙钟时间 | **1716 ms** |
| `Engine.last_load_ms`（模型加载 + 初始化） | **1713 ms** |
| `Engine.last_warmup_ms`（decoder 首次编译 + arena 扩张） | **337 ms** |

`warmup_ms` 占 `load_ms` 的 ~20%。冷启动首帧开销已经被 `Engine::load_models()` 末尾的 decoder warmup 吸收 —— 第一次 `synthesize()` 不再支付 "首次 kernel 编译" 的代价（这是 `跨平台与性能演进.md` P0 的实装效果）。

---

## 4. 循环合成性能（10 次迭代）

### 4.1 逐次数据

| Run | 文本 ID | 音频时长 (s) | tokenize | encode | generate | decode | **total** | **RTF** |
|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 1 | 5.68 | 0 | 0 |  845 | 296 | **1141** | **0.2009** |
| 2  | 2 | 5.84 | 0 | 0 |  828 | 296 | **1124** | **0.1925** |
| 3  | 3 | 5.92 | 1 | 0 |  814 | 252 | **1067** | **0.1802** |
| 4  | 4 | 4.72 | 1 | 0 |  636 | 163 |  **800** | **0.1695** |
| 5  | 5 | 5.36 | 0 | 0 |  747 | 247 |  **994** | **0.1854** |
| 6  | 1 | 5.68 | 0 | 0 |  767 | 241 | **1008** | **0.1775** |
| 7  | 2 | 5.84 | 1 | 0 |  773 | 253 | **1027** | **0.1759** |
| 8  | 3 | 5.92 | 0 | 0 |  768 | 251 | **1019** | **0.1721** |
| 9  | 4 | 4.72 | 0 | 0 |  632 | 157 |  **789** | **0.1672** |
| 10 | 5 | 5.36 | 1 | 0 |  702 | 233 |  **936** | **0.1746** |

所有时间单位毫秒；`encode` 全部为 0 是因为 `ref_0.6B.json` 是预计算好的 speaker embedding，直接跳过 encoder 路径。

### 4.2 聚合统计

| 指标 | min | median | **mean** | max | stdev |
|---|---:|---:|---:|---:|---:|
| **total_ms** | 789 | 1013.5 | **990.5** | 1141 | 119.6 |
| **generate_ms** | 632 | 767.5 | **751.2** | 845 | — |
| **decode_ms** | 157 | 249.0 | **238.9** | 296 | — |
| **RTF** | 0.1672 | 0.1767 | **0.1796** | 0.2009 | — |

### 4.3 首次 vs 稳态

| | total_ms | RTF |
|---|---:|---:|
| Run 1（首次合成） | **1141** | **0.201** |
| Run 2–10 平均 | **974** | **0.177** |
| 差异 | +17% | +14% |

第一次合成比稳态慢约 **17%**，但差距很小 —— 这证明 `load_models()` 里的 decoder warmup 已经把大头成本吃掉了。对比 `跨平台与性能演进.md` 里记录的重构前 `base_small · base` 冷/热落差 **2.7×**（7976 → 2934 ms），现在已经收敛到 **1.17×**。

### 4.4 按文本长度分组

文本长度会影响音频时长，进而影响 total_ms。下面按"句子 ID"聚合看一致性：

| 文本 ID | 长度 | 音频时长 (s) | Run N | Run N+5 | 平均 total | 平均 RTF |
|:---:|---:|---:|---:|---:|---:|---:|
| 1 | 67 | 5.68 | 1141 | 1008 | 1074.5 | 0.1892 |
| 2 | 78 | 5.84 | 1124 | 1027 | 1075.5 | 0.1842 |
| 3 | 93 | 5.92 | 1067 | 1019 | 1043.0 | 0.1762 |
| 4 | 88 | 4.72 |  800 |  789 |  794.5 | 0.1683 |
| 5 | 79 | 5.36 |  994 |  936 |  965.0 | 0.1800 |

- **文本 4**（88 字符但音频只有 4.72s —— 语速快）总时间最短、RTF 最低；
- **文本 1** 作为首句承担了 warmup 余波，两次差异最大（1141 vs 1008）；
- **文本 3**（最长 93 字符，音频 5.92s）的 RTF 最稳定（0.1802 / 0.1721）。

---

## 5. 内存开销

### 5.1 进程级（psutil，100 ms 采样）

| 指标 | 值 |
|---|---:|
| **RSS peak**（整个测试窗口，包含加载 + 10 次合成） | **1080.11 MB** |
| **VRAM delta peak**（相对基线，RTX 3090） | **1159.88 MB** |

### 5.2 引擎级（C API 返回的每次 synth 峰值）

| Run | RSS peak (MB) |
|:---:|---:|
| 1 | 1033.5 |
| 2 | 1050.8 |
| 3 | 1057.5 |
| 4 | 1062.5 |
| 5 | 1078.2 |
| 6 | 1079.8 |
| 7 | 1078.7 |
| 8 | 1079.8 |
| 9 | 1078.3 |
| 10 | 1079.1 |

RSS 在前 5 次逐步增长（ONNX arena + KV cache 扩张），从第 6 次起在 **~1079 MB** 稳定下来。没有泄漏迹象。

### 5.3 分解

| 组件 | 估计占用 | 依据 |
|---|---:|---|
| Decoder ONNX session + arena | ~1.15 GB VRAM | 与 VRAM delta 几乎相等 |
| Talker LLM (q5_k GGUF) | ~500 MB RSS | 0.6B 模型量化后估值 |
| Predictor LLM (q8_0 GGUF) | ~250 MB RSS | q8_0 压缩率 |
| Tokenizer + speaker embedding + codec embedding tables | ~100 MB RSS | mmap 共享 |
| 其他（libs + Python runtime） | ~200 MB RSS | |

VRAM 被 DirectML decoder session 完全主导 —— 与 `跨平台与性能演进.md` 里的观察一致，没有明显浪费。

---

## 6. 瓶颈分布

### 6.1 按阶段

对稳态的 9 次（run 2–10）平均：

| 阶段 | 平均毫秒 | 占比 |
|---|---:|---:|
| tokenize | 0.4 | 0.0% |
| encode | 0.0 | 0.0% |
| **generate** | **740.7** | **76.1%** |
| **decode** | **232.6** | **23.9%** |
| 总计 | 973.8 | 100% |

**`generate` 阶段是绝对主导**（占 76%），这是 talker LLM + predictor sampling 的联合开销 —— 典型的 Transformer 解码循环，受采样步数驱动。`decode` 是 ONNX codec decoder 的单次前向，受 DML + RTX 3090 约束。

### 6.2 为什么 `encode` / `tokenize` 都是 0？

- `ref_0.6B.json` 已经把 speaker encoder 的输出预计算好了。合成路径直接读取 embedding，**完全跳过** encoder ONNX session。
- `tokenize` 使用纯 CPU 的 tokenizer.json 处理英文文本，耗时低于测量粒度（< 1 ms）。

对于"用 `.wav` 文件作参考"的路径，`encode` 会多出 ~200 ms。对于非拉丁语系文本，`tokenize` 可能到 2–5 ms。

---

## 7. 与历史数据对比

下面是同一模型 + 同一参考文件在不同阶段的 `bench_baseline.py` 单次结果（2 次 repeat 的 run1），与本次 10 次循环的 run1 做对比：

| 来源 | 条件 | load_ms | run1_ms | RTF1 | RSS MB |
|---|---|---:|---:|---:|---:|
| 重构前基线 (commit `7b7d8fd`) | 跨平台与性能演进.md | 1436 | 3059 | 0.267 | 1033 |
| Phase A | grep 守卫 + 平台收敛 | 1687 | 2164 | 0.189 | 1034 |
| Phase F | 全部重构完成 | 1616 | 2066 | 0.181 | 1056 |
| **本次 10 次循环** | ctypes 直调 C API | **1713** | **1141** | **0.201** | **~1080** |

> **说明**：本次 run1 (1141 ms) 显著**快于**历史 run1 的 2066–3059 ms 是因为：
>
> 1. `bench_baseline.py` 每次跑都是全新进程，`run1` 包含 `Engine::load_models()` 之外的一些首次分配；本次测试 `run1` 是 `Engine` 已经构造完、warmup 已经跑完之后的第一次合成。
> 2. 历史测试用固定长句（13 秒英文），本次 run 1 文本只生成 5.68 s 的音频，自然更快。
>
> 更公平的对比应该看 **RTF**（归一化到音频时长）：**本次 mean RTF 0.1796** 与 Phase F 的 0.181 吻合，说明重构后的性能稳定。

---

## 8. 结论

### 8.1 核心指标

| 项目 | 值 |
|---|---|
| **冷加载** | 1.71 s |
| **稳态 RTF (平均)** | **0.1796** （约 5.57× 实时速率） |
| **10 次合成总耗时** | 9.91 s |
| **10 次合成产出音频总时长** | 55.04 s |
| **进程 RSS 峰值** | 1080 MB |
| **GPU VRAM 占用** | 1160 MB |
| **首次 vs 稳态落差** | +17%（健康） |

### 8.2 健康信号

- ✅ **RTF 稳定**：min 0.167, max 0.201, stdev ≈ 10%，无 outlier
- ✅ **内存无泄漏**：RSS 在 run 5 之后收敛到 1079 MB 稳定值
- ✅ **冷启动小**：warmup 吃掉主要首帧开销，run1 只比稳态慢 17%
- ✅ **阶段分布清晰**：generate (76%) + decode (24%)，两个瓶颈都在预期位置
- ✅ **跨阶段一致**：Phase A–F 所有阶段的 RTF 都在 0.17–0.20 噪声带内

### 8.3 后续优化空间

参考 `跨平台与性能演进.md` 的路线图：

1. **P1 — Decoder VRAM 瘦身**：当前 1.16 GB 基本是权重本体，可通过 `kSameAsRequested` arena 策略 + F16 权重瘦身挤掉 20–30%
2. **P2 — Codec 嵌入 LRU**：长文本场景下保持 F16 reverse cache 有界，避免长序列重算
3. **P3 — KV cache 动态化**：当前 talker `n_ctx=768` / predictor `n_ctx=256` 固定分配，短文本有浪费
4. **P4 — 更细粒度 timing**：当前 `generate` 是一个 bucket，拆成 `llama_prefill / llama_decode_loop / predictor_sample / talker_decode` 能定位优化目标（已在 `stats_schema.md` 里预留字段，需要 `LUNAVOX_TIMING` 编译标志）

这些都不是当前报告关心的范围 —— 当前数据表明重构本身没有引入任何回归，底层已经健康。

---

## 9. 数据复现

```bash
# 确保构建产物是最新的
lunavox build

# 运行本报告的脚本（已在 _tmp_bench.py）
python _tmp_bench.py

# 原始数据写入 _tmp_bench.json
```

脚本特点：
- 10 次循环，5 种文本轮转
- ctypes 直调 C API，无 subprocess / 无 stdout 解析
- psutil + pynvml 采样，100 ms 粒度
- 输出结构化 JSON，可直接喂给后续分析工具
