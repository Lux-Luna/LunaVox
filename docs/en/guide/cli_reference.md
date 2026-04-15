# LunaVox CLI Reference

The `lunavox` CLI is the single entry point for environment setup, model
management, native engine builds, direct in-process synthesis, and the
desktop GUI. Pick `bootstrap` for the one-key path, or run individual
commands as needed.

```powershell
pip install lunavox           # core CLI
pip install "lunavox[gui]"    # + desktop GUI
pip install "lunavox[convert]"  # + source → GGUF conversion toolchain
```

## Command tree

```
lunavox
├── bootstrap            One-key setup: pull → libs → build → smoke test
├── model
│   ├── pull             Pull pre-converted GGUF/ONNX artifacts
│   ├── convert          Convert raw HF weights into LunaVox artifacts
│   └── list             Show the catalog + which models are installed
├── build                Build the C++ engine (cmake wrapper)
│   └── libs             Download ONNX Runtime / llama.cpp binaries
├── synth TEXT           In-process synthesis via the Python Engine
├── serve                HTTP + WebSocket serving layer ([serve] extra)
├── gui                  Launch the desktop GUI (requires [gui] extra)
└── doctor               Environment + dependency health check
```

## 1. `doctor` — System Health Check

Verifies project layout, toolchain, runtime libraries, and which
profile is active. Run this before opening any issue.

```bash
lunavox doctor
```

Checks: project root + `src` / `lib` / `models`; `cmake` on `PATH`;
ONNX Runtime SDK headers; llama.cpp runtime libs; whether the
`[convert]` extra is installed; the currently selected profile.

## 2. `bootstrap` — One-Key Setup

Runs **pull → libs → build → in-process smoke test** in sequence. The
smoke test uses the native Python `Engine` + `Voice.base()` path — no
subprocess, so it exercises exactly what real callers run.

```bash
lunavox bootstrap
lunavox bootstrap --model base_small --platform win_cuda12
lunavox bootstrap --skip-test          # build only, no synthesis check
```

## 3. `model` — Catalog Management

### `lunavox model pull` (recommended)

Pull pre-converted GGUF / ONNX artifacts from the community mirror.

```bash
lunavox model pull
lunavox model pull --model base_small
```

### `lunavox model convert`

Convert from raw `.safetensors` weights locally. Requires the
`[convert]` extra and takes several minutes.

```bash
lunavox model convert --model base_small --force
lunavox model convert --all
```

### `lunavox model list`

Show every catalog entry and whether it is installed locally.

```bash
lunavox model list
```

## 4. `build` — Native Engine

### `lunavox build`

CMake build of the C++ engine and C ABI shared library.

```bash
lunavox build
lunavox build --clean --j 8
lunavox build --toolchain msvc
```

### `lunavox build libs`

Fetch platform-specific ONNX Runtime + llama.cpp binaries.

```bash
lunavox build libs
lunavox build libs --platform win_cuda12
# win_cuda13 / win_vulkan / win_cpu / linux_cuda / mac_arm64
```

## 5. `synth` — In-Process Synthesis

Run the Python `Engine` directly and write a WAV. This is the
canonical smoke test and the same code path used by the GUI.

```bash
# Default speaker
lunavox synth "Hello from LunaVox." -o output/hello.wav

# Clone from a reference
lunavox synth "Okay, fine." \
  --voice clone --ref ref/ref_0.6B.json \
  -o output/cloned.wav

# Catalog speaker with a style instruction
lunavox synth "She said she would be here by noon." \
  --voice custom --speaker Vivian --instruct "Use angry tone." \
  -o output/custom.wav

# Design a voice from a text description
lunavox synth "It's in the top drawer… wait, it's empty?" \
  --voice design --instruct "Speak in an incredulous tone." \
  -o output/designed.wav
```

Tunable flags: `--model`, `--temperature`, `--top-p`, `--top-k`.
Anything not overridden on the command line falls through to the
active profile, then environment variables, then defaults.

## 6. `serve` — HTTP / WebSocket Server

```bash
pip install "lunavox[serve]"
lunavox serve --host 127.0.0.1 --port 8000 --batch-size 4
```

Starts a FastAPI app with `POST /v1/synth`, `WS /v1/stream`, `GET
/health`, and `GET /v1/models`. Under the hood a `BatchEngine` pool
of `N` independent engines handles concurrent requests — the
`--batch-size` flag sets the pool size (default 4; drop to 1 for
low-VRAM deployments). Streaming supports every voice mode
(`base` / `clone` / `custom` / `design`).

Full endpoint reference and protocol details: **[Serve guide](serve.md)**.

## 7. `gui` — Desktop App

```bash
lunavox gui
```

Requires `pip install "lunavox[gui]"`. The GUI is a three-view
sidebar layout (Synthesize / Library / Settings) that shares the
same `Engine` API as `lunavox synth`.

## 8. Model ID Reference

| Model ID | Full Name | Notes |
| :--- | :--- | :--- |
| `base_small` | Qwen3-TTS 0.6B Base | Fast, balanced, low-resource friendly |
| `custom_small` | Qwen3-TTS 0.6B Custom | Built-in speaker IDs |
| `base` | Qwen3-TTS 1.7B Base | High fidelity; GPU recommended |
| `custom` | Qwen3-TTS 1.7B Custom | Large speaker-customised model |
| `design` | Qwen3-TTS 1.7B Design | Prompt-to-Voice |

## 9. Profiles and Config

LunaVox reads `~/.lunavox/config.toml` on every invocation. The file
has a `[default]` table plus any number of `[profile.<name>]`
overrides. Precedence, highest wins:

1. CLI flags (`--temperature 0.9`, `--model base`)
2. Environment variables (`LUNAVOX_MODEL`, `LUNAVOX_BACKEND`, …)
3. The `[profile.NAME]` table selected with `--profile NAME`
4. The `[default]` table
5. Hardcoded defaults

Example `config.toml`:

```toml
[default]
model = "base_small"
backend = "auto"
n_threads = 4

[profile.quality]
backend = "cuda"
temperature = 0.7
top_p = 0.9

[profile.fast]
backend = "vulkan+dml"
temperature = 0.8
```

```bash
lunavox --profile quality synth "High fidelity please." -o out.wav
```

## 10. Global Flags

Apply to every `lunavox` subcommand:

- `--profile <NAME>` — pick a `[profile.<NAME>]` table from `config.toml`
- `--project-root <PATH>` — explicit project root (development)
- `--yes` — auto-confirm all prompts (CI)
- `--no-install` — disable automatic Python module fixing
- `--verbose` — raw output for builds and downloads

## See also

- [Model profile & runtime contract](../technical/model_profile.md)
- [Usage tutorial (`lunavox synth` modes)](usage_tutorial.md)
- [Serve layer (`lunavox serve`)](serve.md)
- [Runtime API](../api/runtime.md)
