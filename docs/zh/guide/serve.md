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
| `--batch-size` | `4` | 并发请求池大小。每个槽位加载一份独立的 Engine —— 按 `N ×` 单 Engine VRAM 预算；低显存部署设为 `1`。 |
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

## Stats 信封

所有成功合成响应都会附带一份 `SynthStatsResponse`：

- `t_total_ms` —— 请求进入到完整音频输出的墙钟时间
- `audio_duration_ms` —— 生成音频的总时长
- `rtf` —— 实时率（`t_total_ms / audio_duration_ms`）
- `rss_peak_bytes` —— 合成期间常驻内存峰值

## 后续计划（Phase 5C）

- 通过 `n_seq_max > 1` 做真正的 llama.cpp continuous batching，把 N×
  KV cache 代价合并成 1×，BatchEngine API 保持不变
- Prometheus `/metrics` 端点（队列深度、逐 engine RTF、VRAM）
- 句级**输入**流式（客户端通过 WS 持续送文本，服务端边收边合成）
- VRAM 感知的 `--batch-size auto`，启动时检测空闲显存自动定大小

本页记录的 HTTP / WebSocket 接口会在这些升级中保持稳定 —— 5C 只调
内部实现，不改 API。
