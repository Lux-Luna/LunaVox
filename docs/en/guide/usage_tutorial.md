# LunaVox CLI Usage Guide

This tutorial details the command format, core parameters, and constraints for the LunaVox C++ inference engine (`lunavox-cli.exe`).

> [!IMPORTANT]
> LunaVox CLI uses a "Command + Options" structure. Different synthesis features are toggled via the `--mode` parameter. If `--mode` is omitted, the system automatically selects based on the `model_type` in `model_profile.json`.

---

## 1. Base Mode
Simple synthesis, only applicable to `base` type models.

```bash
./build/lunavox-cli `
  -m models/base_small `
  -t "What do you mean that I'm not real?" `
  -o out.wav
```

> [!WARNING]
> **Base models do not support `--instruct`**. If you provide instruction text in base mode, the engine will error and terminate.

---

## 2. Voice Cloning / Reference
Mimic a specific voice by providing reference audio or features. **This feature only works for `base` type models.**

### 2.1 Using Reference Audio
```bash
./build/lunavox-cli `
  -m models/base_small `
  -r ref/ref.wav `
  -t "Hello world." `
  -o out.wav
```

### 2.2 Using Reference Features (JSON)
```bash
./build/lunavox-cli `
  --mode clone `
  -m models/base_small `
  -r ref/ref_0.6B.json `
  -t "Hello world." `
  -o out.wav
```

---

## 3. Custom Voice
Use specific expert speaker IDs. **This feature only works for `custom` type models.**

```bash
./build/lunavox-cli `
  --mode custom `
  -m models/custom `
  --speaker Vivian `
  --instruct "Use angry tone." `
  -t "She said she would be here by noon." `
  -o out.wav
```

- **--speaker**: Required. Specify speaker (e.g., `Vivian`, `Aiden`, `Ryan`).
- **--instruct**: Optional. Adjust emotion or tone of the preset character.

---

## 4. Voice Design
Dynamically design a new voice based on text description. **This feature only works for `design` type models.**

```bash
./build/lunavox-cli `
  --mode design `
  -m models/design `
  --instruct "A warm female voice, speaking gently with a hint of a smile." `
  -t "Hello, it's nice to meet you!" `
  -o out.wav
```

- **--instruct**: Required. Provide detailed physical features of the voice.

---

## 📜 Core Constraints & Compatibility Matrix

To ensure best quality and stability, follow these mappings:

| Model Type (`model_type`) | Supported Modes (`--mode`) | Support `--instruct`? | Support `--reference`? |
| :--- | :--- | :---: | :---: |
| **Base** | `base` (default), `clone` | ❌ Disabled | ✅ Supported |
| **Custom** | `custom` | ✅ Supported (Tuning) | ❌ Disabled |
| **Design** | `design` | ✅ Required (Definition) | ❌ Disabled |

### Common Errors
- **Base + Instruct**: Throws `Error: --instruct is forbidden in base mode`.
- **Custom/Design + Reference**: Throws `Error: mode 'clone' is incompatible with model_type ...`.

---

## 5. Performance & Threading

- **Thread Control**: Use `-j` or `--threads` (default 4) to control CPU resources.
- **Stats**: Add `--stats-json <path>` to get detailed time, RTF, and memory reports.
- **Build**: Use `python manage.py build --portable` on Windows to bundle all runtime DLLs.
