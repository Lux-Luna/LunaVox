# AGENTS.md

Practical engineering guide for AI agents working in `lunavox`.

## Purpose

This file is a working guide for agents, not a project report.
Keep it focused on:

- current repo truths an agent must not guess about
- preferred build and test entry points
- stable editing constraints and sync obligations

If a detail is better explained by the code, tests, or conversion docs, link to those instead of duplicating long technical explanations here.

## Fast Facts

- Runtime stack: C++17 + GGML.
- Main user-facing outputs: `qwen3-tts-cli` and shared library `qwen3tts`.
- Public APIs matter: treat CLI behavior, C API behavior, and installed headers as public surface.
- Primary Python entry point: `python manage.py <command>`.
- Preferred build command: `python manage.py build --backend auto`.
- Required runtime model files:
  - `models/qwen3-tts-0.6B-base.gguf` (Main: Talker + Predictor + Vocoder + Text Tokenizer)
- Optional auxiliary model files (required for voice cloning):
  - `models/qwen3-tts-aux-f16.gguf` or `models/qwen3-tts-aux-q8_0.gguf`
- Voice-cloning smoke tests also require `reference/ref-audio.wav`.

## Repository Shape

Only remember the parts agents commonly touch:

- `src/main.cpp`: CLI.
- `src/qwen3_tts.{h,cpp}`: top-level pipeline orchestration and runtime behavior.
- `src/qwen3tts_c_api.{h,cpp}`: C API wrapper.
- `src/text_tokenizer.{h,cpp}`: text tokenization and TTS prompt formatting.
- `src/tts_transformer.{h,cpp}`: talker + code predictor generation core.
- `src/audio_tokenizer_encoder.{h,cpp}`: speaker encoder.
- `src/audio_tokenizer_decoder.{h,cpp}`: vocoder decoder.
- `src/gguf_loader.{h,cpp}`: GGUF loading and backend helpers.
- `tools/build_manager.py`: preferred build helper behind `manage.py build`.
- `tools/setup/setup_pipeline.py`: model setup helper.
- `tools/conversion/`: conversion and inspection scripts.
- `tests/`: smoke and invariant tests.
- `docs/tensor_mapping.md`: deeper tensor/conversion reference.

## Current Build Truths

- Do not assume users must build GGML separately first.
- Top-level `CMakeLists.txt` supports three GGML modes:
  1. `GGML_BUILD_DIR=<configured-build-tree>`
  2. integrated build from vendored `ggml/`
  3. `find_package(ggml CONFIG)` fallback
- On Windows, `tools/build_manager.py` imports an MSVC environment and probes generator health (`NMake` + `Ninja`).
- Default stable order on Windows is:
  - use cached/explicit generator only if health checks pass
  - otherwise fallback to `NMake Makefiles`
  - then fallback to `Ninja`
  - finally let CMake auto-select as a last resort
- If `NMake Makefiles` is selected but `nmake` is unavailable, the build manager no longer proceeds blindly.
- If the existing build tree generator mismatches the selected generator, `tools/build_manager.py` regenerates the build directory automatically.
- `tools/build_manager.py` now sets `Release` for single-config generators by default (`--build-type` to override).
- `tools/build_manager.py` supports `--ggml-build-dir <path>` to reuse an existing configured GGML build tree.
- If `--ggml-build-dir` is omitted, `tools/build_manager.py` auto-detects `ggml/build_<backend>` or `ggml/build-<backend>` when present.
- Minimal Windows build environment for `manage.py build`:
  - Conda env: `python=3.10`, `cmake`, `ninja`, `pip`
  - MSVC Build Tools 2022 with C++ x64 toolset and Windows SDK
- Backend selection in `manage.py build`:
  - `auto` -> `metal` on macOS
  - `auto` -> `cpu` on non-macOS platforms
- Useful commands:
  - `python manage.py build --backend auto`
  - `python manage.py build --backend cpu`
  - `python manage.py build --backend cuda`
  - `python manage.py build --backend metal`
  - `python manage.py build --backend coreml`
  - `python manage.py build --backend cpu --ggml-build-dir ggml/build_cpu`
  - `python manage.py build --backend cpu --build-type RelWithDebInfo`
  - `cmake -S . -B build-cpu`
  - `cmake --build build-cpu -j 4`

## Public Surface To Protect

When changing behavior, check all three:

- CLI in `src/main.cpp`
- C API in `src/qwen3tts_c_api.{h,cpp}`
- install/export surface in `CMakeLists.txt`

Current installed public headers include:

- `src/qwen3_tts.h`
- `src/qwen3tts_c_api.h`
- component headers under `src/`

## Runtime Truths Agents Should Not Guess About

- `Qwen3TTS::load_models()` expects:
  - `qwen3-tts-0.6B-base.gguf` (required)
  - `qwen3-tts-aux-f16.gguf` or `qwen3-tts-aux-q8_0.gguf` (optional)
- Current loading split (restructured):
  - base model GGUF -> text tokenizer + TTS transformer + vocoder
  - aux model GGUF (optional) -> speaker encoder + codec encoder
- Text tokenizer currently formats TTS input as:
  - `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
- `QWEN3_TTS_LOW_MEM` enables lazy decoder loading and component unload/reload behavior.
- CoreML behavior is conditional:
  - only active on Apple builds with `QWEN3_TTS_COREML=ON`
  - runtime env vars:
    - `QWEN3_TTS_USE_COREML`
    - `QWEN3_TTS_COREML_MODEL`
  - implementation may fall back to GGML depending on platform and package availability

## Testing Truths

- Tests are smoke/invariant oriented, not Python-reference parity tests.
- Current CTest entries:
  - `tokenizer_test`
  - `encoder_test`
  - `transformer_test`
  - `decoder_test`
  - `tts_template_chinese_test`
  - `cli_basic_smoke_test`
  - `cli_clone_smoke_test`
- Run tests with the active build directory, for example:
  - `ctest --test-dir build-cpu --output-on-failure`
- Tests require local model assets.
- `encoder_test` and `cli_clone_smoke_test` require `reference/ref-audio.wav`.

## Editing Rules

- Prefer simple control flow and small, direct changes.
- Use `fprintf(stderr, ...)` for logging to match the existing codebase.
- Error-return style is common: return `bool` and store details in `error_msg_`.
- Public interfaces live in namespace `qwen3_tts`.
- Preserve the existing header style; most headers use `#pragma once`.
- Keep backend teardown symmetric with initialization.
- Preserve CPU fallback behavior when a non-CPU backend is selected.

## GGML Execution Guardrails

Most forward paths follow this lifecycle:

1. Build graph.
2. Allocate scheduler graph memory.
3. Set input tensors.
4. Execute graph.
5. Read outputs.
6. Reset scheduler.

When editing forward paths, keep scheduler reset and tensor I/O discipline intact.

Do not casually remove the explicit `ggml_cast(..., GGML_TYPE_F32)` before `ggml_mul_mat` for `ffn_down` in talker or code predictor paths; this is tied to model correctness.

## Language And Token Notes

- Runtime supports manual language override from CLI / params.
- Auto language detection is heuristic, text-based, and currently strongest for:
  - Japanese via kana
  - Korean via Hangul
  - Russian via Cyrillic
  - Chinese via Han characters
  - Latin script fallback to English
- Known language IDs actively used by the runtime:
  - `2050` en
  - `2053` de
  - `2054` es
  - `2055` zh
  - `2058` ja
  - `2061` fr
  - `2064` ko
  - `2069` ru
  - `2070` it
  - `2071` pt
- Common codec special IDs:
  - `codec_pad_id = 2148`
  - `codec_bos_id = 2149`
  - `codec_eos_id = 2150`
  - `codec_think_id = 2154`
  - `codec_nothink_id = 2155`
  - `codec_think_bos_id = 2156`
  - `codec_think_eos_id = 2157`

Treat live code in `src/tts_transformer.h` and GGUF metadata as source of truth if values drift.

## Update Triggers

Update this file in the same change if you modify any of the following:

- target names or install outputs
- required model filenames or model directory layout
- `manage.py build` behavior or `tools/build_manager.py` backend selection
- CTest target names or required test assets
- CLI/C API public behavior
- CoreML activation rules or related env vars
- prompt formatting used by `TextTokenizer::encode_for_tts()`

## What Not To Put Here

Avoid turning this file into:

- a long architecture explainer
- a tensor mapping reference
- a benchmark log
- a changelog
- a copy of `README.md`

For deeper conversion or tensor naming details, use `docs/tensor_mapping.md`.
If this file and the code disagree, trust the code and update this file.
