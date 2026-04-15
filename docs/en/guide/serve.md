# `lunavox serve` — HTTP / WebSocket Serving Layer

`lunavox serve` starts a FastAPI application that wraps the in-process
`Engine` with an HTTP + WebSocket API. It's the same code path as
`lunavox synth` and the desktop GUI — there is no subprocess, no CLI
string-building, no second synthesis code path to maintain.

## Installation

```bash
pip install "lunavox[serve]"
```

The extra pulls `fastapi`, `uvicorn[standard]`, and `pydantic>=2`.

## Starting the server

```bash
lunavox serve --host 127.0.0.1 --port 8000
lunavox serve --model base_small --port 8080
lunavox --profile quality serve
```

Flags:

| Flag | Default | Purpose |
| :--- | :--- | :--- |
| `--host` | `127.0.0.1` | Bind address. Use `0.0.0.0` to listen on all interfaces. |
| `--port` | `8000` | Bind port. |
| `--model` | (profile default) | Model directory name under `models/`. |
| `--log-level` | `info` | uvicorn log level (`critical`/`error`/`warning`/`info`/`debug`). |

The active profile, threads, and sampler defaults come from your
`~/.lunavox/config.toml` just like every other `lunavox` command.

## Concurrency model

Phase 5A serialises concurrent requests through a single
`asyncio.Lock` around one in-process `Engine`. Multiple clients may
connect at once, but synthesis happens one-at-a-time on one GPU — the
lock keeps the C engine correct. Phase 5B will swap in a C++
BatchEngine for continuous batching without changing any of the
handler shapes below.

## Endpoints

### `POST /v1/synth`

One-shot synthesis. Accepts all four voice modes. Returns the WAV
bytes in the response body and a compact JSON envelope with stats in
the `X-Lunavox-Stats` header.

```json
{
  "text": "Hello from LunaVox.",
  "voice": "base",
  "temperature": 0.7,
  "top_p": 0.9
}
```

Mode-specific fields:

- `voice=clone` — set `reference` to a `.wav` or precomputed `.json` path
- `voice=custom` — set `speaker` (and optionally `instruct`)
- `voice=design` — set `instruct` (required)

Response:

```
HTTP/1.1 200 OK
Content-Type: audio/wav
X-Lunavox-Stats: {"sample_rate":24000,"n_samples":...,"mode":"base","stats":{...}}

<WAV bytes>
```

cURL example:

```bash
curl -X POST http://127.0.0.1:8000/v1/synth \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from LunaVox.","voice":"base"}' \
  --output out.wav
```

### `WS /v1/stream`

WebSocket sentence-streaming. Phase 5A supports `voice=base` only.
Other voice modes close with RFC 6455 code `1003`.

Protocol:

1. Client sends one JSON text frame matching `SynthRequest` above.
2. Server sends one or more binary frames containing raw
   **int16 little-endian** PCM chunks at the engine's sample rate
   (typically 24 kHz).
3. Server sends one terminal JSON text frame of the form
   ```json
   {"done": true, "sample_rate": 24000, "stats": {"t_total_ms": ..., "rtf": ..., ...}}
   ```
   and closes the connection.

Python client snippet:

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8000/v1/stream") as ws:
        await ws.send(json.dumps({"text": "Hello from LunaVox.", "voice": "base"}))
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

TTFB is driven by the existing C++ decoder pipeline (`first_chunk_frames`
default 8). On the RTX 3090 + Vulkan+DML configuration the first chunk
typically arrives in ~200 ms; subsequent chunks follow at the decoder's
steady-state cadence.

### `GET /health`

Liveness probe. Returns `{"status": "ok" | "loading" | "error", ...}`.

### `GET /v1/models`

Catalog listing — every model in `lunavox.model.config.MODELS` with
an `installed` flag indicating whether it exists under `models/` on
disk.

## Stats envelope

All endpoints that return a successful synthesis include a
`SynthStatsResponse` with:

- `t_total_ms` — wall time from request in to full audio out
- `audio_duration_ms` — produced audio length
- `rtf` — real-time factor (`t_total_ms / audio_duration_ms`)
- `rss_peak_bytes` — peak resident-set memory during synthesis

## What's next (Phase 5B)

- C++ `BatchEngine` with `n_seq_max > 1` and continuous batching
- Streaming for `voice=clone`/`custom`/`design`
- Prometheus `/metrics`
- Sentence-level input streaming (client feeds text over WS, server
  starts synthesising before the full sentence arrives)

The HTTP / WebSocket surface documented here will stay stable across
that transition — 5B adds capacity and throughput, not a new API.
