# Docker Deployment

LunaVox ships a multi-stage `Dockerfile` + `compose.yml` so the
HTTP/WebSocket serving layer deploys in one line — no CMake, no C++
compile on the host.

The image is **CPU-only** (ships `linux_cpu` ONNX Runtime + llama.cpp
libs). CUDA images are on the roadmap — change the builder-stage
`lunavox build libs --platform` target and switch the runtime base to
`nvidia/cuda`.

Install: `pip install lunavox` (see [CLI reference](cli_reference.md)
for extras).

## Prerequisites

- Docker 24+ (`Dockerfile` uses `# syntax=docker/dockerfile:1.7`)
- ~6 GB free disk for the first build
- A LunaVox model already pulled to `./models/` on the host — the
  image doesn't download models itself

## 1. Pull a model on the host

```bash
pip install lunavox
lunavox model pull --model base_small
```

`./models/base_small/` should now contain `*.gguf`, `*.onnx`,
`tokenizer.json`, and `embeddings/`.

## 2. Build the image

```bash
docker build -t lunavox:2.2.2 .
```

First-run cost is ~8–15 min on a modern laptop (mostly CMake + C++
compile). Cached rebuilds: < 1 min for Python-only changes, ~3 min
for C++ changes.

## 3. Run with `docker compose` (recommended)

```bash
docker compose up
```

Brings up the server on `http://localhost:8000` with:
- `./models/` mounted read-only at `/app/models`
- `./ref/` mounted read-only at `/app/ref`
- `./output/` mounted read-write at `/app/output`
- `--batch-size auto` (probes free VRAM; falls back to 4 on CPU)
- A health check against `/health` every 30 s

Override via environment variables:

```bash
LUNAVOX_PORT=9000 docker compose up
LUNAVOX_BATCH_SIZE=2 docker compose up
```

## 4. Run without compose

```bash
# Basic run
docker run --rm \
    -p 8000:8000 \
    -v "$(pwd)/models:/app/models:ro" \
    -v "$(pwd)/ref:/app/ref:ro" \
    -v "$(pwd)/output:/app/output" \
    lunavox:2.2.2

# Pass through any lunavox serve flags
docker run --rm \
    -p 8000:8000 \
    -v "$(pwd)/models:/app/models:ro" \
    lunavox:2.2.2 \
    lunavox serve --host 0.0.0.0 --port 8000 --batch-size 2 --model base_small
```

## Image internals

Two-stage build:

**Stage 1 — builder** (`python:3.11-slim-bookworm`). Installs
`cmake` / `ninja` / `g++` / `libgomp1`, copies the repo to `/src`,
runs `lunavox build libs --platform linux_cpu` then
`lunavox build --clean` — emits `liblunavox.so` + `lunavox-cli` into
`/src/build/`.

**Stage 2 — runtime** (`python:3.11-slim-bookworm`). Installs only
`libgomp1` / `libstdc++6` / `dumb-init`, creates a non-root `lunavox`
user (UID 10001), `pip install lunavox==2.2.2` from PyPI, copies
prebuilt `/src/build/` + `/src/lib/` from stage 1. Writes a
`.lunavox-root` deployment marker so
`lunavox.core.project.resolve_project_root()` trusts `/app` without
needing `CMakeLists.txt` or `src/`. Sets `LUNAVOX_PROJECT_ROOT=/app`
and `LUNAVOX_LIB_PATH=/app/build/liblunavox.so`; `EXPOSE 8000`,
default `CMD` runs `lunavox serve`.

Final image is ~500 MB: ~150 MB base Python, ~200 MB pip deps
(uvicorn, fastapi, pydantic, numpy, prometheus-client, typer, rich,
huggingface-hub), ~150 MB compiled C++ engine + runtime libs.

## Production notes

- **Non-root user.** Runs as UID 10001. Bind-mount directories must
  be readable by that UID or a matching GID.
- **Health checks.** `compose.yml` runs a 30 s probe against
  `GET /health`. Use the same endpoint for Kubernetes liveness /
  readiness probes.
- **Prometheus scraping.** `GET /metrics` on the same port. Point
  Prometheus at `http://<container>:8000/metrics`.
- **Signal handling.** `dumb-init` forwards `SIGTERM` from
  `docker stop` to uvicorn so in-flight requests can finish.
- **Batch size.** See [Serve guide — concurrency model](serve.md#concurrency-model)
  for the VRAM/throughput trade-off. Set `--batch-size 1` on
  memory-constrained deployments.
