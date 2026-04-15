# Model Catalog

The model catalog is the **single source of truth** for every Qwen3-TTS
variant LunaVox ships with. Every other module — the downloader, the
conversion pipeline, the CLI prompts — reads `MODELS` from this module.
Adding or renaming a model means touching this file and nothing else.

## The MODELS registry

```python
from lunavox.model import MODELS, all_models, get_model

# Dict keyed by internal short name
print(list(MODELS.keys()))
# ['base_small', 'custom_small', 'base', 'custom', 'design']

# Ordered list preserving registry order
for spec in all_models():
    print(f"{spec.name:15s}  {spec.size}  mode={spec.mode}  repo={spec.repo_id}")

# Direct lookup raises ValueError on unknown
spec = get_model("base_small")
```

## ModelSpec

::: lunavox.model.config.ModelSpec
    options:
      show_root_heading: true

::: lunavox.model.config.MODELS
    options:
      show_root_heading: true

::: lunavox.model.config.all_models
    options:
      show_root_heading: true

::: lunavox.model.config.model_keys
    options:
      show_root_heading: true

::: lunavox.model.config.get_model
    options:
      show_root_heading: true

::: lunavox.model.config.get_snapshot
    options:
      show_root_heading: true

## ModelConfig and project-rooted view

::: lunavox.model.config.ModelConfig
    options:
      show_root_heading: true

::: lunavox.model.config.Models
    options:
      show_root_heading: true
