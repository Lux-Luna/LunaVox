# `lunavox-cli` Usage Tutorial

`lunavox-cli` follows a **command + options** layout. The `--mode` flag picks the synthesis path; if omitted, the engine routes by `model_profile.model_type` (see [model_profile.md](../technical/model_profile.md)).

## 1. Base Mode

Plain text-to-speech for `base` models.

```bash
./build/lunavox-cli `
  -m models/base_small `
  -t "What do you mean that I'm not real?" `
  -o out.wav
```

> [!WARNING]
> Base models reject `--instruct`. Passing it is a hard error, not a warning.

## 2. Voice Cloning (Base only)

### From reference audio

```bash
./build/lunavox-cli `
  -m models/base_small `
  -r ref/ref.wav `
  -t "Hello world." `
  -o out.wav
```

### From pre-computed reference JSON

```bash
./build/lunavox-cli `
  --mode clone `
  -m models/base_small `
  -r ref/ref_0.6B.json `
  -t "Hello world." `
  -o out.wav
```

The JSON path skips the speaker / codec encoders entirely — see [synthesis_pathway.md](../technical/synthesis_pathway.md).

## 3. Custom Voice (Custom models only)

```bash
./build/lunavox-cli `
  --mode custom `
  -m models/custom `
  --speaker Vivian `
  --instruct "Use angry tone." `
  -t "She said she would be here by noon." `
  -o out.wav
```

- `--speaker` — required; e.g. `Vivian`, `Aiden`, `Ryan`.
- `--instruct` — optional; tunes emotion / tone.

## 4. Voice Design (Design models only)

```bash
./build/lunavox-cli `
  --mode design `
  -m models/design `
  --instruct "A warm female voice, speaking gently with a hint of a smile." `
  -t "Hello, it's nice to meet you!" `
  -o out.wav
```

- `--instruct` — required; describe the target voice.

## Compatibility Matrix

| `model_type` | Allowed `--mode` | `--instruct` | `--reference` |
| :--- | :--- | :---: | :---: |
| Base | `base` (default), `clone` | ❌ forbidden | ✅ supported |
| Custom | `custom` | ✅ tuning | ❌ forbidden |
| Design | `design` | ✅ required | ❌ forbidden |

Hard errors:

- Base + `--instruct` → `Error: --instruct is forbidden in base mode`
- Custom / Design + `--reference` → `Error: mode 'clone' is incompatible with model_type ...`

## Performance Knobs

- `-j` / `--threads` (default 4) — CPU thread count.
- `--stats-json <path>` — dump structured timing / RTF / memory ([stats_schema.md](../technical/stats_schema.md)).
- For Windows portable builds bundling all DLLs, use `lunavox build --clean` after `lunavox download-libs --platform win_*`.
