# v2ProPlus VITS ONNX Template

**🔴 Manual ONNX Export Required**

This directory should contain the ONNX template for v2ProPlus VITS models with speaker vector (SV) support.

## Key Differences from v2Pro

v2ProPlus uses a **larger generator** configuration:

1. **Configuration**:
   - `gin_channels`: 1024 (same as v2Pro)
   - `upsample_initial_channel`: **768** (vs 512 in v2/v2Pro)
   - `upsample_kernel_sizes`: **[20, 16, 8, 2, 2]** (vs [16, 16, 8, 2, 2])

2. **SV Processing**: Identical to v2Pro
   - Linear(20480 → 1024)
   - Add to `ge` + PReLU
   - Linear(1024 → 512)

3. **Additional Inputs**: Same as v2Pro - must include `sv_emb` input

## Export Instructions

Follow the same export process as v2Pro but ensure the model configuration matches v2ProPlus:

```python
# Load v2ProPlus model
checkpoint = torch.load("path/to/v2ProPlus_model.pth")
config = checkpoint['config']

# Verify configuration
assert config['model']['upsample_initial_channel'] == 768
assert config['model']['upsample_kernel_sizes'][0] == 20
assert config['model']['gin_channels'] == 1024

# Export with correct configuration
# ... (same process as v2Pro)
```

## Validation

```python
import onnx
import json

model = onnx.load("vits_fp32.onnx")

# Check inputs include sv_emb
inputs = [i.name for i in model.graph.input]
assert 'sv_emb' in inputs

# Verify model size (should be larger than v2Pro)
# Check initializers for larger channel dimensions
```

## Notes

- v2ProPlus provides higher fidelity and more stable long-form synthesis
- Requires more VRAM/compute than v2Pro due to larger generator
- SV extraction and injection logic is identical to v2Pro
- Weight key list is the same as v2Pro (both have sv_emb, ge_to512, prelu)

