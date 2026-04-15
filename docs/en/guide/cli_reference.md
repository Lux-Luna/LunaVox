# LunaVox CLI Reference

The `lunavox` CLI is the single entry point for environment setup, model management, and the C++ build. Commands below are listed in execution order — pick `bootstrap` for the one-key path, or run the steps individually.

```powershell
pip install lunavox
```

## 1. `doctor` — System Health Check

Verifies project layout, toolchain, and runtime libraries. Run this before opening any issue.

```bash
lunavox doctor
```

Checks: project root + `src` / `lib` / `models`; `cmake` on `PATH`; ONNX Runtime SDK headers; llama.cpp runtime libs; whether the `[convert]` extra is installed.

## 2. `bootstrap` — One-Key Setup

Runs **pull-model → download-libs → build → interactive test** in sequence.

```bash
lunavox bootstrap
lunavox bootstrap --model base_small --platform win_cuda12
```

## 3. Model Management

### `pull-model` (recommended)

Pull pre-converted GGUF / ONNX artifacts from the official mirror.

```bash
lunavox pull-model
lunavox pull-model --model base_small
```

### `convert`

Convert from raw `.safetensors` weights locally. Requires the `[convert]` extra and takes several minutes.

```bash
lunavox convert --model base_small --force
```

## 4. Manual Build

### `download-libs`

Fetch the platform-specific ONNX Runtime + llama.cpp binaries.

```bash
lunavox download-libs
lunavox download-libs --platform win_cuda12   # win_cuda13 / win_vulkan / win_cpu / linux_cuda / mac_arm64
```

### `build`

CMake build of `lunavox-cli` (and the C ABI shared library).

```bash
lunavox build
lunavox build --clean --j 8
```

## 5. Model ID Reference

| Model ID | Full Name | Notes |
| :--- | :--- | :--- |
| `base_small` | Qwen3-TTS 0.6B Base | Fast, balanced, low-resource friendly |
| `custom_small` | Qwen3-TTS 0.6B Custom | Built-in speaker IDs |
| `base` | Qwen3-TTS 1.7B Base | High fidelity; GPU recommended |
| `custom` | Qwen3-TTS 1.7B Custom | Large speaker-customised model |
| `design` | Qwen3-TTS 1.7B Design | Prompt-to-Voice |

## 6. Global Flags

Apply to every `lunavox` subcommand:

- `--project-root <PATH>` — explicit project root (development).
- `--yes` — auto-confirm all prompts (CI).
- `--no-install` — disable automatic Python module fixing.
- `--verbose` — raw output for builds and downloads.

## See also

- [Model profile & runtime contract](../technical/model_profile.md)
- [Usage tutorial (`lunavox-cli` modes)](usage_tutorial.md)
