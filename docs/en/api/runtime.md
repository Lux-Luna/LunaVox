# Runtime Engine

The runtime binding is the main **embedding surface** of LunaVox. GUIs,
scripts, and notebooks call into `lunavox.runtime.Engine` directly via
ctypes — no subprocess, no stdout parsing. The C handle is managed with
RAII semantics so a `with Engine(...) as eng:` block is safe from leaks.

Since v2.2.0 the public API is a single `synthesize(text, voice, params)`
entry point. Every voice mode is a `Voice.<factory>()` call; adding a
new mode is a one-line extension instead of a new method.

## Example

```python
from pathlib import Path

from lunavox.runtime import Engine, SynthesisParams, Voice

with Engine(Path("models/base_small")) as eng:
    params = SynthesisParams(temperature=0.6, top_p=1.0, top_k=50)

    # Default speaker
    result = eng.synthesize("Hello from LunaVox.", voice=Voice.base(), params=params)

    # Clone from a reference file (.wav or pre-computed .json)
    cloned = eng.synthesize(
        "Hello from a cloned voice.",
        voice=Voice.clone_file("ref/ref_0.6B.json"),
        params=params,
    )

    # Catalog speaker with an optional style instruction
    custom = eng.synthesize(
        "Use angry tone.",
        voice=Voice.custom("Vivian", instruct="Use angry tone."),
        params=params,
    )

    # Design a new voice from a text description
    designed = eng.synthesize(
        "It's in the top drawer… wait, it's empty?",
        voice=Voice.design(instruct="Speak in an incredulous tone."),
        params=params,
    )

    print(f"RTF: {result.stats.rtf:.3f}")
    print(f"Peak RSS delta: {result.stats.mem.rss_peak_delta_bytes / 1024**2:.1f} MB")
    if result.stats.mem.vram_measured:
        print(f"Peak VRAM delta: {result.stats.mem.vram_peak_delta_bytes / 1024**2:.1f} MB")
    # result.audio is a numpy.float32 array in [-1, 1], mono @ sample_rate
```

## Engine

::: lunavox.runtime.engine.Engine
    options:
      show_root_heading: true
      members:
        - __init__
        - __enter__
        - __exit__
        - close
        - is_loaded
        - sample_rate
        - last_load_ms
        - last_warmup_ms
        - synthesize
        - on_log

## Voice

::: lunavox.runtime.voice.Voice
    options:
      show_root_heading: true
      members:
        - base
        - clone_file
        - custom
        - design

## Synthesis params

::: lunavox.runtime.params.SynthesisParams
    options:
      show_root_heading: true

::: lunavox.runtime.params.default_params
    options:
      show_root_heading: true

## Synthesis result

::: lunavox.runtime.params.SynthesisResult
    options:
      show_root_heading: true

::: lunavox.runtime.params.SynthesisStats
    options:
      show_root_heading: true

::: lunavox.runtime.params.SynthesisMode
    options:
      show_root_heading: true

## Error hierarchy

::: lunavox.runtime.errors.LunavoxLibraryError
    options:
      show_root_heading: true

::: lunavox.runtime.errors.LunavoxSynthesisError
    options:
      show_root_heading: true

## Library loader

::: lunavox.runtime._capi.library_path
    options:
      show_root_heading: true
