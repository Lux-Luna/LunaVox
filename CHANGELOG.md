# Changelog

All notable changes to LunaVox are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project loosely follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `ruff` + `pytest` developer workflow in `pyproject.toml` (`[tool.ruff]`,
  `[tool.ruff.format]`, `[tool.pytest.ini_options]`).
- First Python unit test suite under `tests/` — 45 tests covering
  `core.platform`, `core.project`, `core.stats_schema`, `core.deps`,
  `model.config`, `runtime.binding` ctypes struct layout,
  `build/libs.json` manifest integrity, and CLI `--help` smoke.
- GitHub Actions CI (`.github/workflows/ci.yml`) running ruff + pytest on
  Windows / Linux / macOS for Python 3.10–3.12. C++ build is
  intentionally out of scope so the workflow stays under a minute.
- This `CHANGELOG.md` file.

### Changed
- Codebase formatted with `ruff format`; ~20 Python files reflowed for
  consistent spacing and import ordering. No runtime behavior changes.

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
