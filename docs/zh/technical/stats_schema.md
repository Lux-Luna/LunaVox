# Stats Schema

三处生产者输出同一份结构化 stats：

| 生产者 | 表达形式 | 源码 |
| --- | --- | --- |
| `lunavox-cli --stats-json report.json` | JSON 文件 | `src/main.cpp` |
| C API 返回的 `LunavoxAudio` | 内存结构体 | `src/lunavox_c_api.h` |
| `lunavox.runtime.SynthesisStats` | Python dataclass | `src/lunavox/runtime/binding.py` |

共享 schema 钉在 `src/lunavox/core/stats_schema.py`。新增字段必须 **同一次提交内** 同步修改该模块加上 `src/main.cpp` + `src/lunavox_c_api.h` + `src/lunavox/runtime/binding.py`。

## 顶层 `StatsJSON`

```jsonc
{
  "t_load_ms":    1714,    // Engine::load_models 内的墙钟时间
  "t_warmup_ms":   565,    // 其中 decoder warmup 的时间
  "runs": [ ... RunStats ... ]
}
```

## `RunStats`

```jsonc
{
  "run_id":              1,
  "sample_rate":         24000,
  "n_samples":           71040,
  "audio_duration_s":    2.96,
  "rtf":                 0.175,
  "effective_language_id": -1,
  "timing_ms": { ... TimingMs ... },
  "stream":    { ... StreamStats ... },
  "mem":       { ... MemoryBytes ... }
}
```

`rtf = timing_ms.total / 1000 / audio_duration_s`，越小越快；`< 1.0` 表示快于实时。

## `TimingMs`（毫秒）

| 字段 | 必填 | 含义 |
| --- | :---: | --- |
| `tokenize` | ✓ | 文本 → token IDs |
| `encode` | ✓ | Speaker encoder（使用预计算 embedding JSON 时为 0） |
| `generate` | ✓ | LLM 生成（talker + predictor + 采样） |
| `decode` | ✓ | ONNX decoder + 后处理 |
| `total` | ✓ | 上述总和加 overhead |
| `first_audio` | ✓ | 流式管道首块 PCM 出片墙钟时间 |
| `llama_prefill` / `llama_decode_loop` / `talker_post` / `predictor_sample` / `talker_decode` |   | 细分子项，需要 `LUNAVOX_TIMING` 编译标志 |
| `decoder_tensor_prep` / `decoder_ort_run` / `decoder_tensor_extract` / `decoder_state_trim` / `pcm_gather` |   | 同上 |

## `StreamStats`

| 字段 | 含义 |
| --- | --- |
| `first_chunk_frames` | 首块 decoder 的帧数（TTFB 调优旋钮） |
| `t_first_audio_ms` | C API 中 `audio.first_audio_ms` 的同名值 |

## `MemoryBytes`（字节）

| 字段 | 含义 |
| --- | --- |
| `rss_start` / `rss_end` | 合成开始 / 结束时的进程 RSS |
| `rss_peak` | 合成期间 RSS 高水位 |
| `phys_start` / `phys_end` / `phys_peak` | macOS `phys_footprint`（Windows / Linux 上等于 RSS） |

## C API 子集

`LunavoxAudio` 把上面字段中的常用子集直接平铺在结构体里，避免 Python 绑定再 round-trip：

```c
typedef struct LunavoxAudio {
    const float* samples;
    int32_t  n_samples;
    int32_t  sample_rate;
    int64_t  t_tokenize_ms;
    int64_t  t_encode_ms;
    int64_t  t_generate_ms;
    int64_t  t_decode_ms;
    int64_t  t_total_ms;
    int64_t  audio_duration_ms;
    float    rtf;
    uint64_t rss_peak_bytes;
    uint64_t rss_end_bytes;
} LunavoxAudio;
```

Python 绑定以 `SynthesisStats` 镜像同一子集。更细的 `llama_*` / `decoder_*` 仅出现在 JSON 文件输出中。

## 使用方

- `benchmark/run_benchmark.py` —— 读取 `--stats-json`，计算 100 次跑分的 latency / TTFB / RTF / 内存分布，产出 `benchmark/report.md`。
- `lunavox.gui.widgets.stats_card::StatsCard.update_stats` —— 在 GUI 的 stats 卡片上直接渲染 `SynthesisStats`（不再经过 legacy dict 转换）。
- 新工具应当 `from lunavox.core.stats_schema import ...`，而不是去戳裸 dict。
