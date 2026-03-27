# 🌌 LunaVox CLI Comprehensive Reference Manual

This document provides detailed documentation for the LunaVox Unified Command Line Interface (CLI). Ordered by execution flow, you can choose between one-key guided setup or manual step-by-step configuration.

---

## 🛠️ 1. Environment Preparation & Diagnosis

Before performing any model work, run this tool to ensure your system environment (CMake, C++, Python dependencies) is ready.
```powershell
# Install core inference tooling
pip install lunavox
```

### `doctor` - System Health Check
Check project structure, toolchain paths, and runtime library integrity.

**Example Command:**
```bash
lunavox doctor
```

**Checks Include:**
- Presence of project root and key subdirectories (`src`, `lib`, `models`).
- `cmake` availability in system PATH.
- Integrity of ONNX Runtime SDK headers and Llama.cpp runtime libraries.
- Whether the `[convert]` optional package is installed.

---

## 🚀 2. Core Guided Setup (The One-Key Solution)

Recommended for first-time users or those wanting to start inference quickly.

### `bootstrap` - One-Key Setup
A highly automated composite command that executes:
1.  **Pull Model**: Download the selected model from HuggingFace.
2.  **Download Libs**: Detect system and download appropriate ONNX/Llama runtime libraries.
3.  **Build**: Configure and compile the C++ inference engine.
4.  **Interactive Test**: Start an interactive test to hear synthesis results immediately.

**Usage Example:**
```bash
# Start interactive setup
lunavox bootstrap

# Or specify parameters
lunavox bootstrap --model base_small --platform win_cuda12
```

---

## 📦 3. Model Management

You can choose to download pre-converted models (recommended) or convert from raw weights locally.

### `pull-model` - Pull Pre-converted Models (Recommended)
Sync optimized runtime formats (GGUF/ONNX) directly from the official repository.

**Usage Example:**
```bash
# Start interactive selection
lunavox pull-model

# Download specific model
lunavox pull-model --model base_small
```

### `convert` - Local Model Conversion
Use this if you have raw Qwen3-TTS weights (`.safetensors`) or need custom conversion parameters.

**Usage Example:**
```bash
# Local conversion
lunavox convert --model base_small --force
```
*Note: Local conversion may take several minutes and requires the Python conversion environment.*

---

## ⚙️ 4. Manual Build Workflow

If you prefer not to use `bootstrap`, follow these steps manually.

### `download-libs` - Download Runtime Libraries
Download platform-specific binary cores (ONNX Runtime / Llama.cpp).

**Usage Example:**
```bash
# Smart download (interactive)
lunavox download-libs

# Specified platform download
lunavox download-libs --platform win_cuda12
```

### `build` - Compile C++ Inference Engine
Local build based on CMake to generate the `qwen3-tts-cli` executable.

**Usage Example:**
```bash
# Minimal build
lunavox build

# Clean and parallel build
lunavox build --clean --j 8
```

---

## 📜 Appendix: Model ID Reference Table

| Model ID | Full Name | Inference Capability |
| :--- | :--- | :--- |
| `base_small` | Qwen3-TTS 0.6B Base | Fast & Balanced, good for low-resource devices |
| `custom_small` | Qwen3-TTS 0.6B Custom | Supports fixed speaker ID switching |
| `base` | Qwen3-TTS 1.7B Base | High fidelity, GPU recommended |
| `custom` | Qwen3-TTS 1.7B Custom | Large speaker-customized model |
| `design` | Qwen3-TTS 1.7B Design | Prompt-to-Voice (Design voice using text) |

---

## 🌍 Global Parameters

Applicable to **all** `lunavox` commands:

- `--project-root <PATH>`: Manually specify root directory (often used in dev).
- `--yes`: Auto-confirm all risky operations and downloads (required for CI).
- `--no-install`: Disable automatic Python module detection/fixing.
- `--verbose`: Show detailed raw output for builds and downloads.

---

## 📜 More Information

See also:
- **[Runtime Design Constraints (Runtime Specs)](../technical/runtime_specs.md)**
