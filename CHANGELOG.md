# Changelog

All notable changes to LunaVox are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project loosely follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`examples/voice_agent_demo.py`** — end-to-end voice-agent
  demonstration against `lunavox serve`. Fakes an LLM by streaming
  a scripted reply word-by-word over `WS /v1/stream/text`, receives
  PCM chunks, writes them to a WAV, and prints timing stats
  (first-audio TTFB, audio/wall ratio, per-sentence breakdown).
  Verified locally: 5 sentences detected, 508 ms TTFB, 4.86× audio
  faster than wall clock.
- **`examples/README.md`** — index of example scripts with run
  instructions and a snippet showing how to swap the fake LLM for
  a real OpenAI / Ollama / llama.cpp source.
- **Deployment-layout project root** — `lunavox.core.project`
  recognises a `.lunavox-root` marker file as a valid project root,
  so containers and standalone bundles that don't ship the source
  tree can still use the CLI. Backwards-compatible — dev-checkout
  layouts with `CMakeLists.txt` + `src/` continue to work. Two new
  tests lock the behaviour.
- **`Dockerfile` + `compose.yml`** — multi-stage CPU image builds
  the C++ engine inside the builder stage (`lunavox build libs
  --platform linux_cpu` + `lunavox build`) then copies the artifacts
  into a slim `python:3.11-slim-bookworm` runtime that pip-installs
  `lunavox[serve]==2.2.0` from PyPI. Non-root user (UID 10001),
  `dumb-init` as PID 1 for clean `SIGTERM` handling, `/metrics` +
  `/health` exposed on port 8000. `compose.yml` mounts
  `./models/`, `./ref/`, `./output/` and defaults to
  `--batch-size auto`.
- **Bilingual `docs/{en,zh}/guide/docker.md`** — full Docker
  deployment guide including build steps, compose usage,
  standalone `docker run`, image internals breakdown, and
  production notes (healthchecks, Prometheus scraping, batch-size
  trade-offs).

### Changed
- **Test suite GUI fixture** — `tests/conftest.py` adds a
  session-scoped `gui_root` fixture providing a single shared
  `customtkinter.CTk` root to every GUI test. Replaces the
  per-test `ctk.CTk()` construction which occasionally tripped a
  transient `TclError: couldn't read file auto.tcl` on miniconda
  setups by hammering Tcl interpreter bootstrap four times per
  suite run. Strictly less risky — one root per session now.

## [2.2.0] — 2026-04-15

This is a milestone release that bundles the full
architectural maturation of LunaVox: the big 4.x refactor of the
CLI/GUI/Runtime layers plus the three-sub-phase serving roll-out
(5A HTTP + WebSocket, 5B concurrent-request pool, 5C Prometheus
metrics / sentence-streaming input / VRAM-aware auto sizing). All
of this ships in one big release because nothing in between was
ever published to PyPI.

### Added — Serving layer (Phase 5A–5C)

- **HTTP + WebSocket serving** — new `lunavox serve` subcommand
  backed by a FastAPI app under `src/lunavox/serve/`:
  - `POST /v1/synth` — one-shot synthesis, all four voice modes
    (base / clone / custom / design), returns WAV body with
    stats envelope in the `X-Lunavox-Stats` header.
  - `WS /v1/stream` — streaming WebSocket, all four voice modes,
    binary int16 LE PCM chunks + terminal JSON stats frame.
  - `WS /v1/stream/text` — sentence-level **input** streaming for
    voice agents: client streams text chunks, server detects
    sentence boundaries via `SentenceBuffer` and synthesizes each
    complete sentence as soon as it lands. End-to-end TTFB drops
    from "full LLM reply + first sentence TTFB" to "first sentence
    LLM time + first sentence TTFB".
  - `GET /health`, `GET /v1/models` — liveness + catalog.
  - `GET /metrics` — Prometheus exposition with 5 core metrics
    (`lunavox_pool_size`, `lunavox_pool_idle`, `lunavox_requests_total`
    by voice/status, `lunavox_request_duration_seconds` histogram,
    `lunavox_rtf` histogram).
- **`BatchEngine` concurrent-request pool** — owns `N`
  independent `Engine` instances and dispatches work via an
  `asyncio.Queue`. `submit(text, voice, params)` /
  `synthesize_stream(...)` / `close()` form the public API; the
  internal pool implementation is future-proof for a real
  multi-sequence llama.cpp upgrade.
- **`--batch-size auto`** — VRAM-aware pool sizing via pynvml.
  Probes free VRAM, divides by a per-slot footprint estimate
  (1.1 GB for `*_small` models, 3.1 GB for larger ones), reserves
  20% headroom, clamps to `[1, 16]`. `LUNAVOX_VRAM_PER_SLOT_MB`
  overrides the heuristic. Falls back to `4` when pynvml isn't
  available so CPU-only / AMD / Intel hosts still start.
- **Streaming C API family** — 4 new symbols, one per voice mode:
  `lunavox_synthesize_streaming`,
  `lunavox_synthesize_with_voice_file_streaming`,
  `lunavox_synthesize_custom_streaming`,
  `lunavox_synthesize_design_streaming`. All take a
  `LunavoxAudioChunkCallback` fired from the decoder worker
  thread as PCM slices land. The one-shot C API is unchanged.
- **`Engine.synthesize_stream(text, voice, params)`** generator
  yielding `SynthesisChunk` objects; terminal chunk carries full
  `SynthesisStats`. Supports every voice mode.
- **Benchmark harness** — `benchmark/run_serve_benchmark.py`
  fires parallel `POST /v1/synth` calls at a running server and
  reports p50/p95/p99 latency + throughput + speedup vs the
  sequential baseline. Target: `batch_size=4` → ≥ 2.5×.
- **`[serve]` optional extra** — `pip install "lunavox[serve]"`
  pulls `fastapi`, `uvicorn[standard]`, `pydantic>=2`,
  `prometheus-client>=0.20`, and `numpy`.
- **34 new serve tests** across `test_serve_schemas.py`,
  `test_serve_app.py`, `test_serve_metrics.py`,
  `test_serve_sentence_buffer.py`, `test_serve_auto_batch.py`,
  and `test_runtime_batch_engine.py`. All gated behind `[serve]`
  via `pytest.importorskip`.

### Added — Phase 4 API / CLI / GUI refactor

- **`Voice` first-class object** (`lunavox.runtime.Voice`) with
  `.base()`, `.clone_file()`, `.custom()`, `.design()` factories.
  `Engine.synthesize(text, voice, params)` is now the single
  synthesis entry point.
- **`lunavox synth` CLI** — Python in-process synthesis with
  `--voice {base,clone,custom,design}`, `--ref`, `--speaker`,
  `--instruct`, `--temperature`, `--top-p`, `--top-k`. Same code
  path as the GUI and benchmarks.
- **`lunavox gui` CLI + new desktop GUI** under
  `src/lunavox/gui/` — customtkinter app with a sidebar layout
  (Synthesize / Library / Settings), deep-purple Luna theme, and
  directly imports `lunavox.runtime.Engine` (no CLI string
  builder). Gated behind the new `[gui]` optional extra.
- **Profile config** — `~/.lunavox/config.toml` with `[default]` and
  `[profile.<name>]` tables. Layered precedence: CLI flags > env
  vars > profile > default > hardcoded. Select with
  `lunavox --profile NAME …`.
- **Grouped CLI commands**: `lunavox model {pull,convert,list}` and
  `lunavox build [libs]` replace the old flat commands.
- **New runtime package layout** — `engine.py`, `voice.py`,
  `params.py`, `errors.py`, `_capi.py` each own one concern.

### Changed
- **Breaking**: CLI commands reorganized. Old → new mapping:
  `pull-model` → `model pull`, `convert` → `model convert`,
  `download-libs` → `build libs`. `bootstrap` / `doctor` /
  `build` kept their names.
- **Breaking**: `Engine.synthesize_*` family collapsed into a
  single `Engine.synthesize(text, voice, params)`. No compat shim —
  callers migrate to `Voice.*` factories.
- **Breaking**: Module-level `lunavox.runtime.set_log_callback`
  replaced by `Engine.on_log(cb)` instance method. Removes a
  global-state race and ties the lifetime of the log trampoline to
  the engine instance.
- **Breaking**: `customtkinter` and `pygame` moved out of core
  `[project.dependencies]` into the new `[gui]` extra. `pip
  install lunavox` no longer pulls them; `pip install
  "lunavox[gui]"` does.
- `SynthesisMode` shrunk to the four modes Python actually exposes
  (`BASE`, `CLONE_FILE`, `CUSTOM`, `DESIGN`). `CLONE_SAMPLES` and
  `CLONE_EMBEDDING` remain in the C ABI but are no longer bound.
- GUI migrated from top-level `GUI/` to `src/lunavox/gui/` so
  `pip install lunavox[gui]` ships the desktop app.
- `src/lunavox/cli/main.py` shrank from 472 LOC to ~80 LOC. Command
  logic now lives in per-command modules.
- `lunavox serve --batch-size` accepts an integer or the literal
  `auto`; integer values are clamped to `[1, 16]`.
- `BatchEngine` exposes `idle_count` / `busy_count` properties so
  the metrics layer can read pool state without poking at the
  private `_idle` queue.
- `POST /v1/synth` and WS endpoints are instrumented end-to-end for
  the Prometheus metrics.
- `lunavox.serve.EngineHolder` now wraps a `BatchEngine` instead of
  a single `Engine`; the `asyncio.Lock` field is gone.

### Removed
- **Breaking**: `GUI/` directory — everything moved into
  `src/lunavox/gui/`. Legacy `GUI/main_setup.py`,
  `GUI/engine.py::get_command_string`, and the copy-pasted
  `_setup_advanced_fields` helper are gone.
- **Breaking**: `lunavox.runtime.binding` module — callers import
  from the new `lunavox.runtime` package surface (`Engine`,
  `Voice`, `SynthesisParams`, …) directly.

### Fixed
- CI pyright failure after `huggingface_hub` auto-upgraded to 1.x on
  GitHub runners: removed the deprecated `resume_download=True` and
  `local_dir_use_symlinks=False` kwargs from
  `lunavox.model.downloader`. Both are no-ops in the 1.x API.
- `tests/test_convert_hf_to_gguf_registry.py` now imports
  `torch`/`transformers` via `pytest.importorskip(... exc_type=ImportError)`
  so the `[dev]`-only CI install collects the test suite instead
  of crashing on a missing `transformers` import.
- `lunavox.model.pipeline` deferred its `numpy` import so the
  `model/__init__.py` re-export chain no longer leaks a hard numpy
  dependency onto `[dev]` installs.
- Line endings normalized via a new `.gitattributes` so Windows CI
  no longer trips `ruff format --check`.
- `src/lunavox/__init__.py::__version__` was stuck at `2.1.3` while
  `pyproject.toml` advanced; both now read `2.2.0`.

### Infrastructure (folded from Phase 1–3 work)
- `ruff` + `pytest` developer workflow in `pyproject.toml`.
- Python unit test suite under `tests/` (106 passing locally).
- GitHub Actions CI matrix (Windows / Linux / macOS × Python 3.10–3.12).
- `pyright` strict zones for `core/`, `model/config.py`, and the
  new `runtime/errors.py` / `params.py` / `voice.py`.
- MkDocs Material + mkdocstrings documentation site at
  [lux-luna.github.io/LunaVox](https://lux-luna.github.io/LunaVox/).
- Vendored `src/lunavox/model/conversion/hf_export/convert_hf_to_gguf.py`
  trimmed from 11 433 LOC to 2 390 LOC (Qwen2 / Qwen3 only).

## [2.1.6] — 2026-04-15

### Added
- **Streaming synthesis output** (`2fb5887`): the C++ engine overlaps
  talker + predictor + decoder in a threaded pipeline, and the Python
  binding exposes the first-chunk audio via `SynthesisResult.audio`
  without waiting for the full utterance to finish. TTFB instrumentation
  (`t_first_audio_ms`, `first_chunk_frames`) is plumbed through
  `core.stats_schema.StreamStats`.
- **Benchmark harness** (`9526bac`): `benchmark/run_benchmark.py` drives
  100 measurement runs + 5 warmups per backend on a fixed 25-word
  sentence, with per-run JSON, p50/p95/p99 summary, and an NVML VRAM
  sampler. Results written to `benchmark/results/` and aggregated into
  `benchmark/report.md`.

### Changed
- **Cold-start optimization** (`4aea1e9`, `7b7d8fd`): decoder
  `chunk_frames` retuned from 8 to 32 based on RTX 3090 measurements,
  load-time warmup measured separately, `lunavox_last_warmup_ms()`
  exposed via the C API so callers can attribute cold-start cost.
- **One-shot tech-debt cleanup** (`66df1b2`, Phase A–F refactor): removed
  multiple layers of deprecated abstractions, consolidated single-point
  encapsulation rules (`provider_policy.cpp`, `platform_utils.cpp`,
  `core.platform`, `core.ui`, `core.logging`), and locked in the
  "no backward-compat shim" policy documented in `AGENT.md`.

### Fixed
- macOS build path (`f2115ad`).
- GUI audio-capture lifecycle (`68dca54`).
- GUI GBK decoding bug and splitting into per-page components
  (`b898220`, `63ef3f0`).

## [2.1.x] — pre-2026-04

### Added
- Desktop GUI built with CustomTkinter (`bacc163`, `9c2fbc6`): thin
  shell over `lunavox.runtime.Engine` — no subprocess to the CLI, all
  synthesis goes through the ctypes binding.
- Bilingual documentation (`12830c4`, `e56ee88`) under `docs/en/` and
  `docs/zh/` covering CLI reference, install, synthesis pathway, and
  Windows performance benchmarks.
- `lunavox convert` CLI subcommand (`a54a127`, `f09d247`) for local
  HuggingFace → GGUF/ONNX conversion.
- `lunavox bootstrap` one-key setup command (`d085c3e`) combining
  pull-model + download-libs + build + smoke test.
- DirectML / Vulkan execution providers wired into `provider_policy.cpp`,
  gated behind platform detection (`27ad973`).
- VRAM monitor via NVML (`03fa186`) feeding `benchmark/run_benchmark.py`.
- Deferred mmap preload for decoder weights (`3a2e80d`) to shrink
  cold-start memory footprint.
- FP16 embeddings (`40372d1`) — shrinks projected text-embedding and
  codec tables without measurable quality loss.

### Changed
- Memory footprint reduction pass (`4a31183`, `747cd78`) — peak RSS on
  CUDA dropped from ~1.8 GB to ~1.4 GB at the 0.6B scale.
- Quality-correction sweep across custom / design modes (`d1a2f72`).
- Custom-mode TTS latency fix (`166d9dd`) — removed a redundant
  speaker-encoder pass that ran on every synth call.

## [2.0.0] and earlier

Initial public releases. See `git log` for detail — this changelog only
tracks changes starting from the 2.1.x series.

[Unreleased]: https://github.com/Lux-Luna/LunaVox/compare/v2.1.6...HEAD
[2.1.6]: https://github.com/Lux-Luna/LunaVox/releases/tag/v2.1.6
