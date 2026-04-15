# Docker Deployment

LunaVox ships a multi-stage `Dockerfile` plus a `compose.yml` so the
HTTP/WebSocket serving layer can be deployed in one line, without
users having to install CMake or compile the C++ engine on their
host machine.

> [!NOTE]
> The image built by this Dockerfile is **CPU-only** — it ships with
> the `linux_cpu` ONNX Runtime and llama.cpp runtime libraries. CUDA
> images are on the roadmap but not shipped yet (contributions
> welcome; the change is scoped to swapping the builder-stage
> `lunavox build libs --platform` target and switching the runtime
> base image to `nvidia/cuda`).

## Prerequisites

- Docker 24+ (the `Dockerfile` uses `# syntax=docker/dockerfile:1.7`)
- At least 6 GB free disk for the first build (builder stage
  downloads ONNX Runtime + llama.cpp libs + compiles the C++ engine)
- A LunaVox model already pulled to `./models/` on the host
  (the image doesn't download models itself — they're too large to
  bake in and users typically pick variants at deploy time)

## 1. Pull a model on the host

```bash
pip install lunavox
lunavox model pull --model base_small
```

You should now have `./models/base_small/` containing `*.gguf`,
`*.onnx`, `tokenizer.json`, and the `embeddings/` directory.

## 2. Build the image

```bash
docker build -t lunavox:2.2.0 .
```

First-run cost: ~8–15 minutes on a modern laptop, mostly CMake + C++
compile. Subsequent rebuilds reuse cached layers — expect < 1 min
for Python-only changes, ~3 min for source changes that touch the
C++ build.

## 3. Run with `docker compose` (recommended)

```bash
docker compose up
```

This brings up the server on `http://localhost:8000` with:
- `./models/` mounted read-only at `/app/models`
- `./ref/` mounted read-only at `/app/ref`
- `./output/` mounted read-write at `/app/output`
- `--batch-size auto` (probes free VRAM; falls back to 4 on CPU)
- A health check against `/health` every 30 s

Override the port or batch size via environment variables:

```bash
LUNAVOX_PORT=9000 docker compose up
LUNAVOX_BATCH_SIZE=2 docker compose up
```

## 4. Run without compose

```bash
docker run --rm \
    -p 8000:8000 \
    -v "$(pwd)/models:/app/models:ro" \
    -v "$(pwd)/ref:/app/ref:ro" \
    -v "$(pwd)/output:/app/output" \
    lunavox:2.2.0
```

All the `lunavox serve` flags pass through:

```bash
docker run --rm \
    -p 8000:8000 \
    -v "$(pwd)/models:/app/models:ro" \
    lunavox:2.2.0 \
    lunavox serve --host 0.0.0.0 --port 8000 --batch-size 2 --model base_small
```

## Image internals

The Dockerfile is a two-stage build:

**Stage 1 — builder** (`python:3.11-slim-bookworm`)
- Installs `cmake`, `ninja`, `g++`, `libgomp1`
- Copies the LunaVox repo into `/src`
- Runs `lunavox build libs --platform linux_cpu` to pull ONNX
  Runtime and llama.cpp binaries into `/src/lib/`
- Runs `lunavox build --clean` to compile the C++ engine, emitting
  `liblunavox.so` and `lunavox-cli` into `/src/build/`

**Stage 2 — runtime** (`python:3.11-slim-bookworm`)
- Installs only `libgomp1`, `libstdc++6`, and `dumb-init`
- Creates a non-root `lunavox` user (UID 10001)
- `pip install lunavox[serve]==2.2.0` from PyPI
- Copies the prebuilt `/src/build/` + `/src/lib/` from stage 1
- Creates a `.lunavox-root` deployment marker so
  `lunavox.core.project.resolve_project_root()` trusts `/app` as
  a valid root without needing `CMakeLists.txt` or `src/` in the
  container
- Sets `LUNAVOX_PROJECT_ROOT=/app`, `LUNAVOX_LIB_PATH=/app/build/liblunavox.so`
- `EXPOSE 8000` and default `CMD` runs `lunavox serve`

The final image is in the 500 MB range — about 150 MB for the base
Python image, ~200 MB for pip dependencies (uvicorn, fastapi,
pydantic, numpy, prometheus-client, typer, rich, huggingface-hub),
and ~150 MB for the compiled C++ engine plus ONNX Runtime and
llama.cpp binaries.

## Production notes

- **Non-root user.** The container runs as UID 10001 for safer
  host-volume mounts. If you use bind mounts from a non-root host
  directory, make sure the directory is readable by either that
  UID or a matching GID.
- **Health checks.** `compose.yml` runs a 30 s health probe
  against `GET /health`. Kubernetes users should use the same
  endpoint for liveness and readiness probes.
- **Prometheus scraping.** `GET /metrics` is available on the same
  port. Point Prometheus at `http://<container>:8000/metrics`.
- **Signal handling.** `dumb-init` forwards `SIGTERM` from
  `docker stop` to uvicorn so in-flight synthesis requests get a
  chance to finish before the worker is killed.
- **Batch size trade-off.** `--batch-size auto` defaults to 4 on
  CPU hosts without pynvml. Each slot allocates its own copy of the
  model's KV cache and ONNX decoder state, so `batch_size=4` on a
  0.6B model costs ~800 MB of RAM on top of the ~1.5 GB base
  engine footprint. Set `--batch-size 1` on memory-constrained
  deployments.

## See also

- [Serve guide](serve.md) — full HTTP/WebSocket endpoint reference
- [CLI reference](cli_reference.md) — every flag supported by
  `lunavox serve`
