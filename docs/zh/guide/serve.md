# `lunavox serve` —— HTTP / WebSocket 服务层

`lunavox serve` 启动一个 FastAPI 应用，把进程内 `Engine` 封装成 HTTP +
WebSocket API。底层与 `lunavox synth`、桌面 GUI 完全是同一条代码路径
——不开子进程，不拼接 CLI 字符串，没有第二条合成实现。

## 安装

```bash
pip install "lunavox[serve]"
```

该 extra 会安装 `fastapi`、`uvicorn[standard]` 和 `pydantic>=2`。

## 启动服务

```bash
lunavox serve --host 127.0.0.1 --port 8000
lunavox serve --model base_small --port 8080
lunavox --profile quality serve
```

命令行开关：

| 开关 | 默认 | 说明 |
| :--- | :--- | :--- |
| `--host` | `127.0.0.1` | 监听地址，使用 `0.0.0.0` 监听所有网卡。 |
| `--port` | `8000` | 监听端口。 |
| `--model` | 当前 profile 默认值 | `models/` 下的模型目录名。 |
| `--log-level` | `info` | uvicorn 日志级别（`critical`/`error`/`warning`/`info`/`debug`）。 |

当前生效的 profile、线程数、采样默认值全部来自 `~/.lunavox/config.toml`，
与其他 `lunavox` 命令完全一致。

## 并发模型

Phase 5A 使用一个 `asyncio.Lock` 让并发请求串行化到单个进程内 `Engine`。
多个客户端可以同时连接，但同一时刻只有一次合成在 GPU 上进行 —— 锁
保证 C 引擎状态不被破坏。Phase 5B 会在保持现有 handler 签名不变的前
提下，把这个锁背后的单 Engine 替换成真正的 C++ BatchEngine 实现
continuous batching。

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

WebSocket 流式合成。Phase 5A 只支持 `voice=base`，其他 voice 模式会
直接以 RFC 6455 错误码 `1003` 关闭连接。

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

## 后续计划（Phase 5B）

- C++ `BatchEngine`，支持 `n_seq_max > 1` 和 continuous batching
- `voice=clone` / `custom` / `design` 的流式合成
- Prometheus `/metrics` 端点
- 句级文本流式输入（客户端通过 WS 持续送文本，服务端边收边合成）

本页记录的 HTTP / WebSocket 接口会在 5B 升级中保持稳定 —— 5B 只增加
吞吐和并发能力，不引入新的 API 形状。
