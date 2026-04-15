# Changelog

All notable changes to LunaVox are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project loosely follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Phase 5C

- **`GET /metrics` Prometheus endpoint** — five core metrics
  (pool size, idle slots, requests counter labelled by voice +
  status, request latency histogram, RTF histogram). New
  `LunavoxMetrics` class in `src/lunavox/serve/metrics.py` owns its
  own `CollectorRegistry` so multiple apps in the same process
  don't collide.
- **`WS /v1/stream/text` sentence-streaming input endpoint** —
  voice-agent pattern where an upstream LLM streams text chunks
  into LunaVox and audio comes back per complete sentence. New
  `SentenceBuffer` + sentence splitter in
  `src/lunavox/serve/sentence_buffer.py`. Three new pydantic
  schemas (`TextStreamInit`, `TextStreamChunk`, `TextStreamEnd`)
  for the protocol frames. End-to-end TTFB drops from "full LLM
  reply + first sentence TTFB" to "first sentence LLM time +
  first sentence TTFB".
- **`--batch-size auto`** — VRAM-aware pool sizing via
  `pynvml.nvmlDeviceGetMemoryInfo`. Probes free VRAM, divides by a
  per-slot estimate (1.1 GB for `*_small` models, 3.1 GB for
  larger ones), reserves 20% headroom, clamps to `[1, 16]`.
  `LUNAVOX_VRAM_PER_SLOT_MB` env var lets ops force the per-slot
  estimate when the heuristic is wrong. Falls back to the literal
  number `4` when pynvml is unavailable (CPU-only / AMD / Intel
  hosts) so the server still starts.
- **`prometheus-client>=0.20`** added to the `[serve]` optional
  extra so `pip install "lunavox[serve]"` brings the metrics
  dependency along.
- **18 new tests**: `test_serve_metrics.py` (4),
  `test_serve_sentence_buffer.py` (7), `test_serve_auto_batch.py`
  (7) — all gated behind `[serve]` via `pytest.importorskip`.

### Changed — Phase 5C
- `lunavox serve --batch-size` now accepts a string instead of an
  int so `auto` is a valid value. Integer values still work
  (`--batch-size 4`, `--batch-size 1`); the CLI passes the literal
  to `auto_batch.resolve_batch_size` which clamps to `[1, 16]`.
- `BatchEngine` exposes `idle_count` / `busy_count` properties so
  the metrics layer can read pool state without poking at the
  private `_idle` queue. `idle + busy == batch_size` holds by
  construction.
- `POST /v1/synth` and `WS /v1/stream` are instrumented end-to-end:
  every request bumps `requests_total{voice=...,status=...}`,
  observes `request_duration_seconds`, and observes `rtf` when the
  engine reported one.
- Bilingual `serve.md` rewritten to document `/metrics`,
  `/v1/stream/text`, and `--batch-size auto`. Phase 5C "what's
  next" section now lists the three shipped items as ✅ and
  identifies true continuous batching as the only deferred piece.

### Added — Phase 5B

- **`BatchEngine` concurrent-request pool** (`lunavox.runtime.BatchEngine`):
  owns `N` independent `Engine` instances loaded from the same model
  directory, dispatches incoming work via an `asyncio.Queue` of idle
  engines. Public API (`submit`, `synthesize_stream`, `close`) is
  designed to stay stable across a future true-batching upgrade.
- **`lunavox serve --batch-size N`** (default 4) — turns on the pool.
  The `asyncio.Lock` from Phase 5A is gone; multiple clients hit the
  pool concurrently and back-pressure on its queue instead of racing.
- **Streaming for every voice mode**:
  - 3 new C API symbols — `lunavox_synthesize_with_voice_file_streaming`,
    `lunavox_synthesize_custom_streaming`,
    `lunavox_synthesize_design_streaming` — mirroring the 5A base
    streaming entry.
  - `Engine.synthesize_stream` drops its base-only gate and dispatches
    the same way `synthesize` does.
  - `WS /v1/stream` accepts `voice=clone|custom|design` in addition
    to `base`.
- **`benchmark/run_serve_benchmark.py`** — concurrent HTTP client
  firing parallel `POST /v1/synth` calls at a running `lunavox serve`
  instance. Reports p50 / p95 / p99 latency, throughput, speedup vs
  sequential baseline. Target: `batch_size=4` → ≥2.5× the
  `concurrency=1` baseline.
- **7 new tests**: `test_runtime_batch_engine.py` (6 construction /
  validation cases) plus a restructured `test_serve_app.py` covering
  the new `EngineHolder` → `BatchEngine` plumbing.

### Changed — Phase 5B
- `lunavox.serve.EngineHolder` now wraps a `BatchEngine` instead of
  a single `Engine`; the `asyncio.Lock` field is gone. `create_app`
  grows a `batch_size` keyword argument (default 4).
- `Engine.synthesize_stream` generalized to all four voice modes via
  a new `_dispatch_stream` helper that mirrors the one-shot
  `_dispatch`.
- Bilingual serve guides rewritten to document the context pool,
  the VRAM trade-off table, streaming for every voice mode, and the
  new Phase 5C roadmap (true llama.cpp multi-sequence batching).

### Added — Phase 5A

- **Phase 5A serving layer** — new `lunavox serve` subcommand backed
  by a FastAPI app under `src/lunavox/serve/`. Endpoints:
  - `POST /v1/synth` — one-shot synthesis supporting every voice mode,
    returns a WAV body and a compact stats header.
  - `WS /v1/stream` — WebSocket sentence-streaming (base voice only
    in 5A), driven by the existing decoder worker pipeline.
  - `GET /health`, `GET /v1/models` — standard liveness / catalog.
- **Streaming C API**: new `lunavox_synthesize_streaming` symbol in
  `lunavox_c_api.h` — takes an `LunavoxAudioChunkCallback` fired from
  the decoder worker thread as each PCM slice becomes available. The
  existing one-shot path is unchanged; callers that want the cumulative
  audio plus chunks get both.
- **Engine streaming generator**: `Engine.synthesize_stream(text,
  voice, params)` yields `SynthesisChunk` objects using a background
  worker + `queue.Queue`. Terminal chunk carries the full
  `SynthesisStats`. Phase 5A restricts streaming to base mode; other
  modes raise `NotImplementedError` until 5B.
- **`[serve]` optional extra**: `pip install "lunavox[serve]"` adds
  FastAPI, uvicorn, pydantic, numpy. Core install stays slim.
- **9 new tests**: `test_serve_schemas.py` (pydantic surface) and
  `test_serve_app.py` (app factory, route registration, engine holder
  locking). All gated behind `[serve]` via `importorskip`.
- **Bilingual serve docs**: `docs/en/guide/serve.md` and
  `docs/zh/guide/serve.md` with endpoint reference, cURL / Python
  client examples, and the Phase 5B roadmap.

## [2.2.0] — 2026-04-15

### Added
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
- **New tests** — `test_runtime_voice.py`, `test_runtime_params.py`,
  `test_cli_config.py`, `test_cli_common.py`-equivalents,
  `test_gui_imports.py`, `test_gui_widgets.py` (72 passing locally
  with the GUI extra installed; CI sees 66 passed + 6 skipped
  because `[dev]` has no customtkinter).

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

### Infrastructure (folded from Phase 1–3 work)
- `ruff` + `pytest` developer workflow in `pyproject.toml`.
- Initial Python unit test suite under `tests/` (45 tests then, 66+ now).
- GitHub Actions CI matrix (Windows / Linux / macOS × Python 3.10–3.12).
- `pyright` strict zones for `core/`, `model/config.py`, and the new
  `runtime/errors.py` / `params.py` / `voice.py`.
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
