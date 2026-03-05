# LunaVox Model Converter

This module provides tools to convert PyTorch model checkpoints to the ONNX format
used by LunaVox TTS.

## Supported Output Formats

- **fp16**: FP16 weights + FP32 ONNX skeleton (production format)

## Usage

```python
from converter import convert

# Convert to FP16 format (default)
convert(
    ckpt_path="path/to/s1bert.ckpt",
    pth_path="path/to/s2G.pth",
    output_dir="output/model",
    format="fp16"
)
```

## CLI Usage

```bash
python -m converter --ckpt path/to/s1bert.ckpt --pth path/to/s2G.pth --output output/model --format fp16
```
