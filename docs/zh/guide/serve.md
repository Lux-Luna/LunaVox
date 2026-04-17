# `lunavox serve` —— HTTP / WebSocket 服务层

`lunavox serve` 启动一个 FastAPI 应用，把 `BatchEngine` 并发池封装
成 HTTP + WebSocket API。与 `lunavox synth`、GUI 完全是同一条 `Engine`
代码路径 —— 不开子进程，没有第二条合成实现。N 个 Engine 的 context
pool 让多个客户端真正并行合成，流式支持四种 voice 模式
（`base` / `clone` / `custom` / `design`）。

安装：`pip install lunavox`（详见 [CLI 参考](cli_reference.md)）。
FastAPI、uvicorn、pydantic、prometheus-client 均已在基础包里。

## 启动服务

```bash
lunavox serve --host 127.0.0.1 --port 8000
lunavox serve --model base_small --port 8080 --batch-size 4
lunavox --profile quality serve --batch-size 2
```

| 开关 | 默认 | 说明 |
| :--- | :--- | :--- |
| `--host` | `127.0.0.1` | 监听地址，`0.0.0.0` 监听所有网卡。 |
| `--port` | `8000` | 监听端口。 |
| `--model` | profile 默认值 | `models/` 下的模型目录名。 |
| `--batch-size` | `4` | 池大小：整数 1–16，或 `auto` 通过 pynvml 探测空闲 VRAM。每个槽位加载一份独立 Engine —— 按 `N ×` 单 Engine VRAM 预算。 |
| `--log-level` | `info` | uvicorn 日志级别。 |

当前 profile、线程数、采样默认值来自 `~/.lunavox/config.toml`，与其他
`lunavox` 命令完全一致。

## 并发模型

请求从 `asyncio.Queue` 抢一个空闲 engine，在后台线程上合成，完成后
释放回池。多余的客户端在队列上阻塞等待，而不是抢夺同一个 GPU。

| 配置 | VRAM | 并发 | 吞吐 |
| :--- | :--- | :--- | :--- |
| `--batch-size 1` | 1× engine | 1（排队） | baseline |
| `--batch-size 2` | 2× engine | 2 | ~1.7× |
| `--batch-size 4`（默认） | 4× engine | 4 | ~2.5× |

每个槽位各自持有 KV cache 和 ONNX decoder state，所以 N=4 在 0.6B
模型上约 ~800 MB 额外 VRAM —— 24 GB 显卡忽略不计,8 GB 卡建议
`--batch-size 2`。

## 长文本自动切分

所有端点都走同一个 `AsyncSynthesisPipeline`,进入时先看 `text`:
超过**自动切分阈值**(默认 240 字符)的输入会在标点边界处切成多段,
再分发给 engine 池。客户端看到的始终是一条连续的响应
(完整 WAV 或连续 PCM 流),服务器内部做了几段合成对客户端透明。

级联策略(单一实现,无按语言分支):

1. **强终止符** —— `.` `!` `?`(需后跟空白)、以及 CJK / 印度系 /
   阿拉伯系 / 缅甸 / 泰文自终止的 `。` `！` `？` `…` `।` `؟` 等。
2. **弱终止符** —— `,` `;` `:` `、` `,` `;` `:` `،` `؛`
   (从句边界,当强切分后仍有超 `max_chars` 的段时再细化)。
3. **空白** —— 没有标点的长句的第三档回退。
4. **硬切** —— 最后手段,直接按 `max_chars` 截断
   (比如无标点、无空格的纯 CJK 长串、URL、长标识符)。

也就是说,2000 字的中英混合长文直接丢到 `POST /v1/synth` 就能正常工作,
客户端无需自己预切。流式端点会把每段的 chunk 压平成一条流,整段只有
一个 chunk 带 `is_last=true` 并附汇总 stats。

语言覆盖对齐模型当前支持的 10 种语言(中英日韩俄德法意西葡)。新增
语言只改 `lunavox.core.text.punctuation` 里的标点表,不碰算法。

## 接口列表

### `POST /v1/synth`

一次性合成。支持四种 voice 模式，WAV 字节通过 body 返回，stats 结构
通过 `X-Lunavox-Stats` header 提供。

```json
{
  "text": "你好，来自 LunaVox。",
  "voice": "base",
  "temperature": 0.7,
  "top_p": 0.9
}
```

模式专属字段：

- `voice=clone` —— `reference`：`.wav` 或预计算 `.json` 路径
- `voice=custom` —— `speaker`（可选 `instruct`）
- `voice=design` —— `instruct`（必填）

```
HTTP/1.1 200 OK
Content-Type: audio/wav
X-Lunavox-Stats: {"sample_rate":24000,"n_samples":...,"mode":"base","stats":{...}}

<WAV 字节>
```

```bash
curl -X POST http://127.0.0.1:8000/v1/synth \
  -H "Content-Type: application/json" \
  -d '{"text":"你好，来自 LunaVox。","voice":"base"}' \
  --output out.wav
```

### `WS /v1/stream`

WebSocket 流式合成，支持全部四种 voice 模式。协议：

1. 客户端发送一个 JSON 文本帧，结构与上面的 `SynthRequest` 一致。
2. 服务端以二进制帧推送 **int16 小端序** PCM，采样率为引擎采样率
   （通常 24 kHz）。
3. 服务端最后发送一个结束文本帧后关闭连接：
   ```json
   {"done": true, "sample_rate": 24000, "stats": {"t_total_ms": ..., "rtf": ..., ...}}
   ```

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8000/v1/stream") as ws:
        await ws.send(json.dumps({"text": "你好，来自 LunaVox。", "voice": "base"}))
        pcm_chunks: list[bytes] = []
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                pcm_chunks.append(msg)
            else:
                print("done:", json.loads(msg)["stats"])
                break

asyncio.run(main())
```

TTFB 由 C++ 解码器流水线控制（`first_chunk_frames` 默认 8）。RTX 3090
+ Vulkan+DML 下首包通常在 ~200 ms 到达。

### `WS /v1/stream/text` —— 句级文本流式输入

voice agent 场景的输入流式端点：上游 LLM 把 token 通过文本通道流式
推进 LunaVox，LunaVox 按句输出音频 —— 端到端延迟从"完整 LLM 回复
+ 首句 TTFB"降到"首句 LLM 时长 + 首句 TTFB"。

协议：

1. **Init** —— 一个 JSON 帧，voice / 采样字段（无 `text`）：
   ```json
   {"voice": "base", "temperature": 0.7}
   ```
2. **文本分片** —— 客户端边接 LLM 输出边发送 N 个 JSON 帧：
   ```json
   {"text": "你好。"}
   {"text": "今天天气"}
   {"text": "怎么样？"}
   ```
   每个分片喂进 `StreamingSentenceBuffer`,遇到终止符后完整句子吐给合成器。
3. **音频** —— 每个完整句子的 PCM 通过二进制帧推送
   （int16 LE、引擎采样率），按句子顺序交错到达。
4. **End** —— 客户端发送 `{"end": true}`，残留（无终止符）作为
   最后一个合成单位 flush。
5. **Terminal** —— 服务端发送 1 个 JSON 帧后关闭连接：
   ```json
   {
     "done": true, "sample_rate": 24000, "sentences": 3,
     "stats": {"t_total_ms": 1240, "audio_duration_ms": 4500, "rtf": 0.275, "mem": {"rss_start_bytes": 500000000, "rss_end_bytes": 1400000000, "rss_peak_bytes": 1500000000, "vram_start_bytes": 0, "vram_end_bytes": 0, "vram_peak_bytes": 0, "vram_measured": false}}
   }
   ```
   `stats` 是**最后一个句子**的时延 / 内存快照（最反映尾包延迟）。

句子边界检测复用上节长文本切分同一张多语言标点表,新增语言在这里
自动生效。
短于 4 字符的片段（如 "Mr."）会留在缓冲区不立刻 flush。

### `GET /health`

存活探针。返回 `{"status": "ok" | "loading" | "error", ...}`。

### `GET /v1/models`

`lunavox.model.config.MODELS` 中所有条目，并带 `installed` 字段
标识本地 `models/` 下是否存在。

### `GET /metrics`

Prometheus 抓取端点，返回 `text/plain; version=0.0.4`：

| 指标 | 类型 | 标签 | 含义 |
| :--- | :--- | :--- | :--- |
| `lunavox_pool_size` | gauge | — | BatchEngine pool 总槽位数 |
| `lunavox_pool_idle` | gauge | — | 当前空闲槽位数 |
| `lunavox_requests_total` | counter | `voice`, `status` | 累计合成请求数 |
| `lunavox_request_duration_seconds` | histogram | `voice` | 服务端单请求墙钟时长 |
| `lunavox_rtf` | histogram | `voice` | 引擎报告的实时率 |

每次抓取都会刷新 pool gauge。Histogram bucket 按 RTX 3090 Vulkan
典型负载（25 词英文 RTF ~0.15、延迟 ~1.3 s）调过。

## Stats 信封

所有成功合成响应都附带一份 `SynthStatsResponse`：

- `t_total_ms` —— 请求进入到完整音频输出的墙钟时间
- `audio_duration_ms` —— 生成音频的总时长
- `rtf` —— 实时率（`t_total_ms / audio_duration_ms`）
- `mem` —— 嵌套 `MemStatsResponse`，包含 RSS 与 VRAM 的 `start` / `end` / `peak` 字节数。用 `peak - start` 得到本次合成的实际增长；`end - start` 是合成结束后未回收的残留。
- `mem.vram_measured` —— **VRAM 是否可用的权威标志**。为 `false` 时 `vram_*` 字段未定义，客户端**不要**渲染。`vram_peak_bytes > 0` 不是替代标志：CPU-only 场景下零值是真实读数。`vram_*` 按本进程 PID 聚合（跨所有可见 NVIDIA 设备，走 `nvmlDevice*RunningProcesses`），同机其他进程/其他卡的波动不会污染读数。
