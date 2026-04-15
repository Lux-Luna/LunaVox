# `lunavox serve` — HTTP / WebSocket Serving Layer

`lunavox serve` starts a FastAPI application that wraps a
concurrent-request `BatchEngine` with an HTTP + WebSocket API. Under
the hood it's the same `Engine` code path used by `lunavox synth`
and the desktop GUI — there is no subprocess, no CLI string-building,
no second synthesis code path to maintain.

Since v2.2.0 (Phase 5B), the server uses a **context pool of N
engines** so multiple clients synthesize in parallel instead of
queuing on one GPU. Streaming (`WS /v1/stream`) also supports every
voice mode now, not just base.

## Installation

```bash
pip install "lunavox[serve]"
```

The extra pulls `fastapi`, `uvicorn[standard]`, and `pydantic>=2`.

## Starting the server

```bash
lunavox serve --host 127.0.0.1 --port 8000
lunavox serve --model base_small --port 8080 --batch-size 4
lunavox --profile quality serve --batch-size 2
```

Flags:

| Flag | Default | Purpose |
| :--- | :--- | :--- |
| `--host` | `127.0.0.1` | Bind address. Use `0.0.0.0` to listen on all interfaces. |
| `--port` | `8000` | Bind port. |
| `--model` | (profile default) | Model directory name under `models/`. |
| `--batch-size` | `4` | Concurrent request pool size. Each slot loads its own engine — plan on `N ×` per-engine VRAM. Set `1` for low-VRAM deployments. |
| `--log-level` | `info` | uvicorn log level (`critical`/`error`/`warning`/`info`/`debug`). |

The active profile, threads, and sampler defaults come from your
`~/.lunavox/config.toml` just like every other `lunavox` command.

## Concurrency model

Phase 5B uses a **context pool of N independent `Engine` instances**
behind the `BatchEngine` class. Incoming requests claim an idle
engine from an `asyncio.Queue`, synthesize on a background thread,
then release the engine back into the pool. Excess concurrent
clients back-pressure on the queue rather than racing for the GPU.

| Config | VRAM footprint | Concurrent requests | Target throughput |
| :--- | :--- | :--- | :--- |
| `--batch-size 1` | 1 × engine | 1 (queued) | baseline |
| `--batch-size 2` | 2 × engine | 2 | ~1.7× baseline |
| `--batch-size 4` (default) | 4 × engine | 4 | ~2.5× baseline |

The trade-off is VRAM — each pool slot carries its own KV caches
and ONNX decoder state, so N=4 on a 0.6B model costs ~800 MB extra
VRAM. On a 24 GB GPU that's negligible; on 8 GB cards consider
`--batch-size 2`. Phase 5C will explore a true multi-sequence
llama.cpp upgrade that collapses the N× cost without changing the
API below.

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

WebSocket sentence-streaming. Since Phase 5B, all four voice modes
are supported (`base`, `clone`, `custom`, `design`) — the handler
calls `BatchEngine.synthesize_stream` which dispatches to the
matching `_streaming` C API symbol.

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

## What's next (Phase 5C)

- True llama.cpp continuous batching via `n_seq_max > 1`, collapsing
  the N× KV cache cost while keeping the same BatchEngine API
- Prometheus `/metrics` endpoint (queue depth, per-engine RTF, VRAM)
- Sentence-level **input** streaming (client feeds text over WS as
  the LLM generates it, server starts synthesising before the full
  sentence arrives)
- VRAM-aware `--batch-size auto` that inspects free VRAM at startup

The HTTP / WebSocket surface documented here is stable across those
upgrades — 5C tunes the internals, not the API.
