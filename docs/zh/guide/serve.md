# `lunavox serve` —— HTTP / WebSocket 服务层

`lunavox serve` 启动一个 FastAPI 应用，把进程内的 `BatchEngine` 并发
请求池封装成 HTTP + WebSocket API。底层与 `lunavox synth`、桌面 GUI
完全是同一条 `Engine` 代码路径 —— 不开子进程，不拼接 CLI 字符串，
没有第二条合成实现。

从 v2.2.0（Phase 5B）起，服务端使用 **N 个 Engine 组成的 context
pool**，让多个客户端真正并行合成，而不是在单个 GPU 上排队。流式
接口 (`WS /v1/stream`) 现在也支持四种 voice 模式，不再限制为
base。

## 安装

```bash
pip install "lunavox[serve]"
```

该 extra 会安装 `fastapi`、`uvicorn[standard]` 和 `pydantic>=2`。

## 启动服务

```bash
lunavox serve --host 127.0.0.1 --port 8000
lunavox serve --model base_small --port 8080 --batch-size 4
lunavox --profile quality serve --batch-size 2
```

命令行开关：

| 开关 | 默认 | 说明 |
| :--- | :--- | :--- |
| `--host` | `127.0.0.1` | 监听地址，使用 `0.0.0.0` 监听所有网卡。 |
| `--port` | `8000` | 监听端口。 |
| `--model` | 当前 profile 默认值 | `models/` 下的模型目录名。 |
| `--batch-size` | `4` | 并发请求池大小。可传整数（1–16）或 `auto`：`auto` 会通过 pynvml 探测空闲 VRAM 并自动算一个安全值。每个槽位加载一份独立的 Engine —— 按 `N ×` 单 Engine VRAM 预算；低显存部署设为 `1`。 |
| `--log-level` | `info` | uvicorn 日志级别（`critical`/`error`/`warning`/`info`/`debug`）。 |

当前生效的 profile、线程数、采样默认值全部来自 `~/.lunavox/config.toml`，
与其他 `lunavox` 命令完全一致。

## 并发模型

Phase 5B 在 `BatchEngine` 类背后使用 **N 个独立 `Engine` 实例组成的
context pool**。进来的请求从 `asyncio.Queue` 里抢一个空闲 engine，
在后台线程上跑合成，完成后释放回池子。多余的并发客户端会在队列
上阻塞等待，而不是抢夺同一个 GPU。

| 配置 | VRAM 占用 | 并发请求 | 吞吐目标 |
| :--- | :--- | :--- | :--- |
| `--batch-size 1` | 1 × engine | 1（排队） | baseline |
| `--batch-size 2` | 2 × engine | 2 | ~1.7× baseline |
| `--batch-size 4`（默认） | 4 × engine | 4 | ~2.5× baseline |

代价是 VRAM —— 每个 pool 槽位各自持有 KV cache 和 ONNX decoder
state，所以 N=4 在 0.6B 模型上约占 ~800 MB 额外 VRAM。24 GB 显卡
上忽略不计；8 GB 卡建议 `--batch-size 2`。Phase 5C 会探索 llama.cpp
多序列升级，在不改本页 API 的前提下把 N× 的 KV cache 代价合并成 1×。

## 接口列表

### `POST /v1/synth`

一次性合成。支持四种 voice 模式，WAV 字节通过 body 返回，stats 结构
作为 JSON 序列化后放入 `X-Lunavox-Stats` header。

```json
{
  "text": "你好，来自 LunaVox。",
  "voice": "base",
  "temperature": 0.7,
  "top_p": 0.9
}
```

模式专属字段：

- `voice=clone` —— 设置 `reference` 为 `.wav` 或预计算 `.json` 路径
- `voice=custom` —— 设置 `speaker`（可选 `instruct`）
- `voice=design` —— 设置 `instruct`（必填）

响应：

```
HTTP/1.1 200 OK
Content-Type: audio/wav
X-Lunavox-Stats: {"sample_rate":24000,"n_samples":...,"mode":"base","stats":{...}}

<WAV 字节>
```

cURL 示例：

```bash
curl -X POST http://127.0.0.1:8000/v1/synth \
  -H "Content-Type: application/json" \
  -d '{"text":"你好，来自 LunaVox。","voice":"base"}' \
  --output out.wav
```

### `WS /v1/stream`

WebSocket 流式合成。自 Phase 5B 起支持全部四种 voice 模式
（`base` / `clone` / `custom` / `design`）—— handler 调用
`BatchEngine.synthesize_stream`，按 voice 模式 dispatch 到对应的
`_streaming` C API 符号。

协议：

1. 客户端发送一个 JSON 文本帧，结构与上面的 `SynthRequest` 一致。
2. 服务端以若干个二进制帧推送音频片段，格式为
   **int16 小端序** PCM，采样率就是引擎采样率（通常为 24 kHz）。
3. 服务端最后发送一个结束文本帧：
   ```json
   {"done": true, "sample_rate": 24000, "stats": {"t_total_ms": ..., "rtf": ..., ...}}
   ```
   然后关闭连接。

Python 客户端示例：

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
                terminal = json.loads(msg)
                print("done:", terminal["stats"])
                break

asyncio.run(main())
```

TTFB 由现有 C++ 解码器流水线控制（`first_chunk_frames` 默认 8）。
RTX 3090 + Vulkan+DML 配置下首包通常在 ~200 ms 到达，后续 chunk
按解码器稳态节奏持续到达。

### `GET /health`

存活探针。返回 `{"status": "ok" | "loading" | "error", ...}`。

### `GET /v1/models`

目录列表 —— `lunavox.model.config.MODELS` 中所有条目，并带
`installed` 字段标识本地 `models/` 下是否存在。

### `GET /metrics`

Phase 5C Prometheus 抓取端点，返回标准
`text/plain; version=0.0.4` 格式：

| 指标 | 类型 | 标签 | 含义 |
| :--- | :--- | :--- | :--- |
| `lunavox_pool_size` | gauge | — | BatchEngine pool 总槽位数 |
| `lunavox_pool_idle` | gauge | — | 当前空闲槽位数 |
| `lunavox_requests_total` | counter | `voice`, `status` | 累计合成请求数 |
| `lunavox_request_duration_seconds` | histogram | `voice` | 服务端单请求墙钟时长 |
| `lunavox_rtf` | histogram | `voice` | 引擎报告的实时率 |

每次抓取都会刷新 pool gauge 数据，所以低流量部署也能反映当前
状态。Histogram 的 bucket 是按 RTX 3090 Vulkan 典型负载（25
词英文 RTF ~0.15、延迟 ~1.3 s）调过的。

### `WS /v1/stream/text` —— 句级文本流式输入

Phase 5C 给 voice agent 加的输入流式端点。典型模式：上游 LLM
通过文本通道把 token / 词 / 短语流式推进 LunaVox，LunaVox 不等
完整回复就开始按句输出音频 —— 端到端延迟从"完整 LLM 回复时长
+ 首句 TTFB"降到"首句 LLM 时长 + 首句 TTFB"。

协议：

1. **Init** —— 客户端发送 1 个 JSON 文本帧：
   ```json
   {
     "voice": "base",
     "temperature": 0.7
   }
   ```
   voice / 采样字段与 `SynthRequest` 一致，但没有 `text`。

2. **文本分片** —— 客户端边接 LLM 输出边发送 N 个 JSON 帧：
   ```json
   {"text": "你好。"}
   {"text": "今天天气"}
   {"text": "怎么样？"}
   ```
   服务端把每个分片喂进 `SentenceBuffer`，一旦遇到终止符就立刻
   把完整句子吐给合成器。

3. **音频** —— 每个完整句子的 PCM 通过二进制帧推送（int16 LE，
   引擎采样率）。多句之间天然交错 —— 第 N 句的 chunk 全部到达
   后，第 N+1 句的 chunk 接着到达。

4. **End** —— 客户端发送终止帧：
   ```json
   {"end": true}
   ```
   服务端把缓冲区里没有终止符的残留作为最后一个合成单位 flush
   出去。

5. **Terminal** —— 服务端发送 1 个 JSON 帧后关闭连接：
   ```json
   {
     "done": true,
     "sample_rate": 24000,
     "sentences": 3,
     "stats": {
       "t_total_ms": 1240,
       "audio_duration_ms": 4500,
       "rtf": 0.275,
       "rss_peak_bytes": 1500000000
     }
   }
   ```
   `sentences` 是合成的句子总数；`stats` 是**最后一个句子**的
   时延 / 内存快照（最反映尾包延迟的指标）。

句子边界检测：英文用 `[.!?]` + 空白；中日韩用 `[。！？…．]`
自终止。短于 4 字符的片段（如 "Mr."）会留在缓冲区不立刻 flush，
避免缩写被当成独立句子。

## Stats 信封

所有成功合成响应都会附带一份 `SynthStatsResponse`：

- `t_total_ms` —— 请求进入到完整音频输出的墙钟时间
- `audio_duration_ms` —— 生成音频的总时长
- `rtf` —— 实时率（`t_total_ms / audio_duration_ms`）
- `rss_peak_bytes` —— 合成期间常驻内存峰值

## 后续计划（Phase 5C 完成情况 + 未来）

本次发布交付的 5C 内容：

- ✅ `GET /metrics` Prometheus 指标（pool / 请求 / RTF）
- ✅ `WS /v1/stream/text` 句级文本流式输入（voice agent 模式）
- ✅ `--batch-size auto` 通过 pynvml 探测显存自动定大小

延后到独立 session 的部分：

- 真正的 llama.cpp continuous batching（`llama_wrapper.cpp` 把
  `n_seq_max > 1` 打开 + `TalkerPredictor` 内部按 sequence 切状
  态）。把 `N ×` KV cache 代价合并成 1×，BatchEngine + 服务端
  API 完全不动。预计 2-3 天专注 C++ 工作量，本次为了让 5C 其余
  部分干净落地而推迟。
