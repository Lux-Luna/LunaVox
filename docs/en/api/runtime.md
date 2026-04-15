# Runtime Engine

The runtime binding is the main **embedding surface** of LunaVox. GUIs,
scripts, and notebooks call into `lunavox.runtime.Engine` directly via
ctypes — no subprocess, no stdout parsing. The C handle is managed with
RAII semantics so a `with Engine(...) as eng:` block is safe from leaks.

## Example

```python
from pathlib import Path

from lunavox.runtime import Engine, SynthesisParams

with Engine(Path("models/base_small")) as eng:
    params = SynthesisParams(temperature=0.6, top_p=1.0, top_k=50)
    result = eng.synthesize_with_voice_file(
        text="Hello from LunaVox.",
        reference_path="ref/ref_0.6B.json",
        params=params,
    )
    print(f"RTF: {result.stats.rtf:.3f}")
    print(f"Peak RSS: {result.stats.rss_peak_bytes / 1024**2:.1f} MB")
    # result.audio is a numpy.float32 array in [-1, 1], mono @ sample_rate
```

## Engine

::: lunavox.runtime.binding.Engine
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
        - synthesize_with_voice_file
        - synthesize_custom
        - synthesize_design

## Synthesis params

::: lunavox.runtime.binding.SynthesisParams
    options:
      show_root_heading: true

::: lunavox.runtime.binding.default_params
    options:
      show_root_heading: true

## Synthesis result

::: lunavox.runtime.binding.SynthesisResult
    options:
      show_root_heading: true

::: lunavox.runtime.binding.SynthesisStats
    options:
      show_root_heading: true

::: lunavox.runtime.binding.SynthesisMode
    options:
      show_root_heading: true

## Error hierarchy

::: lunavox.runtime.binding.LunavoxLibraryError
    options:
      show_root_heading: true

::: lunavox.runtime.binding.LunavoxSynthesisError
    options:
      show_root_heading: true

## Log callback

::: lunavox.runtime.binding.set_log_callback
    options:
      show_root_heading: true

::: lunavox.runtime.binding.library_path
    options:
      show_root_heading: true
