# Synthesis Usage Tutorial

Since v2.2.0 LunaVox exposes two equivalent front doors:

- **`lunavox synth`** — the Python CLI driving `lunavox.runtime.Engine`
  directly. Recommended for scripts, CI, and day-to-day use.
- **`./build/lunavox-cli`** — the standalone C++ exe, useful for
  deep profiling or for environments without Python.

Both use the same underlying engine. Examples in this tutorial lead
with `lunavox synth`; the C++ exe invocations follow for reference.

## 1. Base — Default Voice

The model's built-in speaker. No reference audio, no instruction.

```bash
lunavox synth "What do you mean that I'm not real?" -o out.wav
```

```bash
# C++ equivalent
./build/lunavox-cli -m models/base_small \
  -t "What do you mean that I'm not real?" \
  -o out.wav
```

> [!WARNING]
> Base models reject `--instruct` in the C++ exe. Passing it is a hard
> error, not a warning. The Python CLI simply ignores the flag when the
> chosen voice doesn't consume it.

## 2. Voice Cloning

### From a reference audio clip

```bash
lunavox synth "Hello world." \
  --voice clone --ref ref/ref.wav \
  -o out.wav
```

### From a pre-computed reference JSON (fast path)

```bash
lunavox synth "Hello world." \
  --voice clone --ref ref/ref_0.6B.json \
  -o out.wav
```

The JSON path skips the speaker / codec encoders entirely — see
[synthesis_pathway.md](../technical/synthesis_pathway.md).

```bash
# C++ equivalent
./build/lunavox-cli -m models/base_small \
  -r ref/ref_0.6B.json \
  -t "Hello world." -o out.wav
```

## 3. Custom Voice — Catalog Speaker

Pick a named speaker from the `custom` model catalog. `--instruct` is
an optional tone modifier.

```bash
lunavox synth "She said she would be here by noon." \
  --voice custom --speaker Vivian --instruct "Use angry tone." \
  -o out.wav
```

- `--speaker` — required; e.g. `Vivian`, `Aiden`, `Ryan`.
- `--instruct` — optional; tunes emotion / tone.

```bash
# C++ equivalent
./build/lunavox-cli --mode custom -m models/custom \
  --speaker Vivian --instruct "Use angry tone." \
  -t "She said she would be here by noon." \
  -o out.wav
```

## 4. Voice Design — Describe the Voice

```bash
lunavox synth "Hello, it's nice to meet you!" \
  --voice design --instruct "A warm female voice, speaking gently with a hint of a smile." \
  -o out.wav
```

- `--instruct` — required; describe the target voice.

```bash
# C++ equivalent
./build/lunavox-cli --mode design -m models/design \
  --instruct "A warm female voice, speaking gently with a hint of a smile." \
  -t "Hello, it's nice to meet you!" \
  -o out.wav
```

## Python API

The same four modes expose cleanly through the runtime API — this is
what `lunavox synth` and the GUI both drive internally.

```python
from lunavox.runtime import Engine, SynthesisParams, Voice

with Engine("models/base_small") as engine:
    params = SynthesisParams(temperature=0.7, top_p=0.9)

    # Pick whichever mode fits the request
    voice = Voice.clone_file("ref/ref_0.6B.json")

    result = engine.synthesize("Hello world.", voice=voice, params=params)
    print(f"RTF {result.stats.rtf:.3f}, duration {result.stats.audio_duration_ms} ms")
```

See the [Runtime API reference](../api/runtime.md) for the full surface.

## Compatibility Matrix

| `model_type` | Allowed `--voice` | `--instruct` | `--ref` |
| :--- | :--- | :---: | :---: |
| Base | `base` (default), `clone` | ❌ forbidden | ✅ supported (`clone`) |
| Custom | `custom` | ✅ tuning | ❌ forbidden |
| Design | `design` | ✅ required | ❌ forbidden |

Hard errors on the C++ exe:

- Base + `--instruct` → `Error: --instruct is forbidden in base mode`
- Custom / Design + `--reference` → `Error: mode '…' is incompatible with model_type …`

## Performance Knobs

- `--model` (Python CLI) / `-m` (C++ exe) — pick which model directory to load.
- `--temperature` / `--top-p` / `--top-k` — sampler tuning (Python CLI).
- `-j` / `--threads` (default 4) — CPU thread count.
- `lunavox doctor` — verify your profile and runtime libraries.
- `--stats-json <path>` (C++ exe) — dump structured timing / RTF / memory
  ([stats_schema.md](../technical/stats_schema.md)).

## Profiles

For repeatable runs, define profiles in `~/.lunavox/config.toml` and
select them with `--profile`:

```toml
[default]
model = "base_small"

[profile.quality]
backend = "cuda"
temperature = 0.7
top_p = 0.9
```

```bash
lunavox --profile quality synth "High fidelity please." -o out.wav
```

See [CLI reference § Profiles](cli_reference.md) for the full precedence chain.
