# `lunavox serve` — HTTP / WebSocket Serving Layer

`lunavox serve` starts a FastAPI app wrapping a concurrent
`BatchEngine` pool as an HTTP + WebSocket API. Same `Engine` code
path as `lunavox synth` and the GUI — no subprocess, no second
synthesis path. A context pool of N engines lets clients synthesize
in parallel, and streaming supports all four voice modes
(`base` / `clone` / `custom` / `design`).

Install: `pip install lunavox` (see [CLI reference](cli_reference.md)
for extras). FastAPI, uvicorn, pydantic, and prometheus-client are
in the base install.

## Starting the server

```bash
lunavox serve --host 127.0.0.1 --port 8000
lunavox serve --model base_small --port 8080 --batch-size 4
lunavox --profile quality serve --batch-size 2
```

| Flag | Default | Purpose |
| :--- | :--- | :--- |
| `--host` | `127.0.0.1` | Bind address. Use `0.0.0.0` for all interfaces. |
| `--port` | `8000` | Bind port. |
| `--model` | profile default | Model directory under `models/`. |
| `--batch-size` | `4` | Pool size: integer 1–16, or `auto` to probe free VRAM via pynvml. Each slot loads its own engine — budget `N ×` per-engine VRAM. |
| `--log-level` | `info` | uvicorn log level. |

Profile, threads, and sampler defaults come from
`~/.lunavox/config.toml` like every other `lunavox` command.

## Concurrency model

Requests claim an idle engine from an `asyncio.Queue`, synthesize on
a background thread, then release it back to the pool. Excess
clients back-pressure on the queue rather than racing for the GPU.

| Config | VRAM | Concurrent | Throughput |
| :--- | :--- | :--- | :--- |
| `--batch-size 1` | 1× engine | 1 (queued) | baseline |
| `--batch-size 2` | 2× engine | 2 | ~1.7× |
| `--batch-size 4` (default) | 4× engine | 4 | ~2.5× |

Each slot holds its own KV caches and ONNX decoder state, so N=4 on
a 0.6B model costs ~800 MB extra VRAM — negligible on 24 GB, but
consider `--batch-size 2` on 8 GB cards.

## Long-text auto-splitting

Every endpoint routes through a single `AsyncSynthesisPipeline` that
inspects the incoming `text` field. Inputs longer than the pipeline's
**auto-split threshold** (default 240 characters) are chunked at
punctuation boundaries before being dispatched to the engine pool.
The client sees one continuous response — WAV body / PCM stream —
regardless of how many internal segments were synthesized.

Cascade (single implementation, no per-language branches):

1. **Strong terminators** — `.` `!` `?` (needs trailing whitespace),
   and self-terminating `。` `！` `？` `…` `।` `؟` and friends from
   CJK / Indic / Arabic / Burmese / Thai scripts.
2. **Weak terminators** — `,` `;` `:` `、` `，` `；` `：` `،` `؛`
   (clause boundaries, used as fallback when strong-split chunks
   still exceed `max_chars`).
3. **Whitespace** — fallback for long unpunctuated phrases.
4. **Hard cut** at `max_chars` — last-resort slice for things like
   long CJK runs with no punctuation at all (URLs, identifiers).

This means a 2000-character multilingual paragraph is transparently
handled by `POST /v1/synth` without any client-side pre-processing.
Streaming endpoints flatten per-segment chunks into one stream;
exactly one chunk carries `is_last=true` with aggregated stats.

Language coverage targets the 10 languages the model supports today
(CN / EN / JP / KR / RU / DE / FR / IT / ES / PT). New scripts are
added by appending to the punctuation table in
`lunavox.core.text.punctuation`, not by touching the splitter.

## Endpoints

### `POST /v1/synth`

One-shot synthesis. Accepts all four voice modes. Returns WAV bytes
in the body with a compact stats envelope in `X-Lunavox-Stats`.

```json
{
  "text": "Hello from LunaVox.",
  "voice": "base",
  "temperature": 0.7,
  "top_p": 0.9
}
```

Mode-specific fields:

- `voice=clone` — `reference`: path to `.wav` or precomputed `.json`
- `voice=custom` — `speaker` (and optional `instruct`)
- `voice=design` — `instruct` (required)

```
HTTP/1.1 200 OK
Content-Type: audio/wav
X-Lunavox-Stats: {"sample_rate":24000,"n_samples":...,"mode":"base","stats":{...}}

<WAV bytes>
```

```bash
curl -X POST http://127.0.0.1:8000/v1/synth \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from LunaVox.","voice":"base"}' \
  --output out.wav
```

### `WS /v1/stream`

WebSocket sentence-streaming, all four voice modes. Protocol:

1. Client sends one JSON text frame matching `SynthRequest` above.
2. Server sends binary frames of raw **int16 little-endian** PCM at
   the engine sample rate (typically 24 kHz).
3. Server sends one terminal JSON text frame and closes:
   ```json
   {"done": true, "sample_rate": 24000, "stats": {"t_total_ms": ..., "rtf": ..., ...}}
   ```

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
                print("done:", json.loads(msg)["stats"])
                break

asyncio.run(main())
```

TTFB is driven by the C++ decoder pipeline (`first_chunk_frames`
default 8). On RTX 3090 + Vulkan+DML the first chunk typically
arrives in ~200 ms.

### `WS /v1/stream/text` — Sentence-Streaming Input

Input-streaming endpoint for voice agents: an upstream LLM streams
tokens into LunaVox, which emits audio per complete sentence —
dropping end-to-end latency from "full LLM reply + first-sentence
TTFB" to "first-sentence LLM time + first-sentence TTFB".

Protocol:

1. **Init** — one JSON frame with voice / sampler fields (no `text`):
   ```json
   {"voice": "base", "temperature": 0.7}
   ```
2. **Text chunks** — N JSON frames as the LLM produces output:
   ```json
   {"text": "Hello there. "}
   {"text": "How are "}
   {"text": "you today? "}
   ```
   Each chunk feeds a `StreamingSentenceBuffer`; complete sentences
   flush once a terminator + whitespace lands.
3. **Audio** — binary int16-LE PCM frames per sentence, in order.
4. **End** — client signals end-of-stream with `{"end": true}`; any
   leftover fragment flushes as the final unit.
5. **Terminal** — server sends one JSON frame and closes:
   ```json
   {
     "done": true, "sample_rate": 24000, "sentences": 3,
     "stats": {"t_total_ms": 1240, "audio_duration_ms": 4500, "rtf": 0.275, "mem": {"rss_start_bytes": 500000000, "rss_end_bytes": 1400000000, "rss_peak_bytes": 1500000000, "vram_start_bytes": 0, "vram_end_bytes": 0, "vram_peak_bytes": 0, "vram_measured": false}}
   }
   ```
   `stats` is the timing / memory snapshot of the **last** sentence
   (most useful for trailing latency).

Sentence detection uses the same data-driven punctuation tables as
the long-text auto-split cascade above, so new language coverage
lands on this endpoint automatically. Fragments under 4 characters
are held so "Mr." etc. don't flush as standalone sentences.

### `GET /health`

Liveness probe. Returns `{"status": "ok" | "loading" | "error", ...}`.

### `GET /v1/models`

Every entry in `lunavox.model.config.MODELS` with an `installed`
flag indicating whether it exists under `models/` on disk.

### `GET /metrics`

Prometheus scrape endpoint, `text/plain; version=0.0.4`:

| Metric | Type | Labels | Meaning |
| :--- | :--- | :--- | :--- |
| `lunavox_pool_size` | gauge | — | Total engines in pool |
| `lunavox_pool_idle` | gauge | — | Idle engines |
| `lunavox_requests_total` | counter | `voice`, `status` | Requests served |
| `lunavox_request_duration_seconds` | histogram | `voice` | Server-side wall time |
| `lunavox_rtf` | histogram | `voice` | Engine-reported real-time factor |

Pool gauges refresh on every scrape. Histogram buckets are tuned to
a typical RTX 3090 Vulkan run (RTF ~0.15, ~1.3 s for 25 words).

## Stats envelope

Every successful synthesis includes a `SynthStatsResponse`:

- `t_total_ms` — wall time from request in to full audio out
- `audio_duration_ms` — produced audio length
- `rtf` — real-time factor (`t_total_ms / audio_duration_ms`)
- `mem` — nested `MemStatsResponse` with `start` / `end` / `peak` byte counts for both RSS and VRAM. Compute `peak - start` for the synthesis-driven growth; `end - start` is the residual after the run completes.
- `mem.vram_measured` — **authoritative** flag for VRAM availability. When `false`, the `vram_*` fields are undefined — do NOT render them. `vram_peak_bytes > 0` is not a substitute: a zero reading on a CPU-only run is a real measurement. `vram_*` are per-process (summed across visible NVIDIA devices via `nvmlDevice*RunningProcesses`), so a sibling process churning VRAM on the same GPU cannot pollute the reading.
