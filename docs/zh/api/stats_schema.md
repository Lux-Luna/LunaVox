# 统计 Schema

统计 schema 是 `lunavox-cli --stats-json` 输出的**生产者-消费者契约**。
三个生产者发射同一份结构：

1. `src/main.cpp` 在用户传 `--stats-json report.json` 时写 JSON 文件
2. `src/lunavox_c_api.cpp::to_c_audio` 用同样字段填充 `LunavoxAudio`
   结构体，Python ctypes 绑定以结构体成员的形式拿到
3. `src/lunavox/runtime/binding.py::SynthesisStats` 是 GUI 和嵌入脚本
   消费的 Python dataclass

消费者（`benchmark/run_benchmark.py`、GUI）可以
`from lunavox.core.stats_schema import StatsJSON`，而不是对自由字典硬
索引。

> **英文 API 参考是唯一真源**：TypedDict 字段、`ParsedStats` 方法详见
> [**English → Stats Schema**](../../en/api/stats_schema.md)。

## 导航

- **[`TimingMs`](../../en/api/stats_schema.md#lunavox.core.stats_schema.TimingMs)**
  ——每一阶段 wall-clock 毫秒
- **[`StreamStats`](../../en/api/stats_schema.md#lunavox.core.stats_schema.StreamStats)**
  ——流式管线诊断（first chunk、TTFB）
- **[`MemoryBytes`](../../en/api/stats_schema.md#lunavox.core.stats_schema.MemoryBytes)**
  ——RSS 起点 / 终点 / 高水位
- **[`RunStats`](../../en/api/stats_schema.md#lunavox.core.stats_schema.RunStats)**
  ——单次 synth 的完整快照
- **[`StatsJSON`](../../en/api/stats_schema.md#lunavox.core.stats_schema.StatsJSON)**
  ——`--stats-json` 顶层 payload
- **[`ParsedStats` / `parse_stats_json`](../../en/api/stats_schema.md#lunavox.core.stats_schema.ParsedStats)**
  ——下游消费者解析器
