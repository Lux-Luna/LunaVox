# 🌌 LunaVox Runtime Technical Specifications (Strict Profile Contract)

This document defines the strict constraints for the LunaVox inference engine when loading and running models.

## 1. Model Configuration (`model_profile.json`)
`model_profile.json` is a mandatory runtime contract. Missing required fields will result in fatal errors.

### Core Required Fields:
- **`talker_n_ctx`**: Inference context capacity limit (default `2048`).
- **`talker_n_ctx_train`**: Original weight training context limit.
- **`predictor_n_ctx`**: Predictor context capacity (default `256`).
- **`codec_num_codebooks`**: Fixed to `16` for current version.
- **`predictor_vocab_size`**: Predictor vocabulary size.

## 2. Mode Switching Logic
The `--mode` parameter is optional. If omitted, routing is determined by `model_profile.model_type`:
- **`base`**: Automatically switches to standard synthesis. Smart switches to `clone` if `--reference` is provided.
- **`custom`**: Forces `custom` mode routing.
- **`design`**: Forces `design` mode routing.

## 3. Strict Error Control
- **Illegal Combination**: `base` model + `--instruct` or **0.6B** model + `--instruct` will throw a Hard Error and terminate.
- **Weights**: The Talker runtime currently only supports `qwen3_tts_talker.q5_k.gguf` as a valid inference artifact.

## 4. Default Sampling Strategy (Quality-Driven)
Deterministic sampling strategies for optimized quality:
- **Temperature**: `0.6`
- **Predictor Temperature**: `0.6`
- **Max Generation Length**: `max_new_tokens <= 400`
- **Random Seed**: `42`
- **Predictor Seed**: `45`

## 5. Quality Gate

Verify CLI availability:
```bash
./build/qwen3-tts-cli.exe --help
```

- **Timeout Requirement**: Single verification inference should be under 20 seconds.
- **Validation**: Manual subjective hearing check must be performed before any quantization performance comparison.
