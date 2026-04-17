# LunaVox CLI Reference

`lunavox` is the single entry point for environment setup, model
management, native engine builds, in-process synthesis, and the
desktop GUI. Use `bootstrap` for the one-key path, or run individual
commands as needed.

```powershell
pip install lunavox             # core CLI + GUI + HTTP/WebSocket server
pip install "lunavox[convert]"  # + source → GGUF conversion toolchain (optional)
```

## Command tree

```
lunavox
├── bootstrap            One-key setup: pull → libs → build → smoke test
├── model {pull|convert|list}
├── build [libs]         C++ engine build, or fetch runtime libs
├── synth TEXT           In-process synthesis via the Python Engine
├── serve                HTTP + WebSocket serving layer
├── gui                  Launch the desktop GUI
└── doctor               Environment + dependency health check
```

## 1. `doctor` — System Health Check

Verifies project layout (`src` / `lib` / `models`), `cmake` on
`PATH`, ONNX Runtime SDK headers, llama.cpp runtime libs, the
`[convert]` extra, and the active profile. Run before opening any
issue.

```bash
lunavox doctor
```

## 2. `bootstrap` — One-Key Setup

Runs **pull → libs → build → in-process smoke test** in sequence.
The smoke test uses the native Python `Engine` + `Voice.base()` —
exactly what real callers run.

```bash
lunavox bootstrap
lunavox bootstrap --model base_small --platform win_cuda12
lunavox bootstrap --skip-test          # build only, no synthesis check
```

## 3. `model` — Catalog Management

```bash
lunavox model pull                                    # pull default
lunavox model pull --model base_small                 # pull specific
lunavox model convert --model base_small --force      # needs [convert] extra
lunavox model convert --all
lunavox model list                                    # show installed state
```

`pull` fetches pre-converted GGUF / ONNX artifacts from the community
mirror. `convert` builds them locally from raw `.safetensors` weights
and requires the `[convert]` extra (takes several minutes).

## 4. `build` — Native Engine

```bash
lunavox build                               # cmake build
lunavox build --clean --j 8
lunavox build --toolchain msvc
lunavox build libs                          # fetch default runtime libs
lunavox build libs --platform win_cuda12
# also: win_cuda13 / win_vulkan / win_cpu / linux_cuda / mac_arm64
```

## 5. `synth` — In-Process Synthesis

Runs the Python `Engine` directly and writes a WAV — the same code
path used by the GUI.

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
Anything not set on the command line falls through to the active
profile, then environment variables, then defaults.

## 6. `serve` — HTTP / WebSocket Server

```bash
lunavox serve --host 127.0.0.1 --port 8000 --batch-size 4
```

FastAPI app with `POST /v1/synth`, `WS /v1/stream`,
`WS /v1/stream/text`, `GET /health`, `GET /v1/models`, `GET /metrics`.
A `BatchEngine` pool of N engines handles concurrent requests. See
**[Serve guide](serve.md)** for full endpoint / protocol reference.

## 7. `gui` — Desktop App

```bash
lunavox gui
```

Three-view sidebar layout (Synthesize / Library / Settings) sharing
the same `Engine` API as `lunavox synth`.

## 8. Model ID Reference

| Model ID | Full Name | Notes |
| :--- | :--- | :--- |
| `base_small` | Qwen3-TTS 0.6B Base | Fast, balanced, low-resource friendly |
| `custom_small` | Qwen3-TTS 0.6B Custom | Built-in speaker IDs |
| `base` | Qwen3-TTS 1.7B Base | High fidelity; GPU recommended |
| `custom` | Qwen3-TTS 1.7B Custom | Large speaker-customised model |
| `design` | Qwen3-TTS 1.7B Design | Prompt-to-Voice |

## 9. Profiles and Config

LunaVox reads `~/.lunavox/config.toml` on every invocation —
a `[default]` table plus any number of `[profile.<name>]` overrides.
Precedence (highest wins): CLI flags → env vars (`LUNAVOX_MODEL`,
`LUNAVOX_BACKEND`, …) → `[profile.NAME]` (selected via
`--profile NAME`) → `[default]` → hardcoded defaults.

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

Apply to every subcommand:

- `--profile <NAME>` — pick a `[profile.<NAME>]` table
- `--project-root <PATH>` — explicit project root (development)
- `--yes` — auto-confirm all prompts (CI)
- `--no-install` — disable automatic Python module fixing
- `--verbose` — raw output for builds and downloads

## See also

- [Model profile & runtime contract](../technical/model_profile.md) · [Usage tutorial](usage_tutorial.md) · [Serve guide](serve.md) · [Runtime API](../api/runtime.md)
