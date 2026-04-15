# Model Profile & Runtime Contract

`model_profile.json` is the contract every loader honours. It lives in three places that **must** be edited in the same commit:

| Site | Type / file | How it reads |
| --- | --- | --- |
| **C++ engine** (authoritative) | `lunavox::ModelProfile` in `src/model_profile.h` | `src/lunavox_engine.cpp::load_model_profile` |
| **Python runtime** (display only) | `lunavox.model.ModelProfile` in `src/lunavox/model/profile.py` | `ModelProfile.load(model_dir)` |
| **Disk artifact** | `models/<name>/model_profile.json` | Written by `lunavox convert` |

The C++ side is strict (`is_valid` aborts `Engine::load_models` on any mismatch). The Python side is permissive — display metadata only.

## 1. Identity

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `version` | int | 1 | Schema version. Readers refuse unknown majors. |
| `model_type` | string | `"base"` | One of `base`, `custom`, `design`. Gates which `--mode` values the model accepts. |
| `model_size` | string | `"unknown"` | Display label, e.g. `"0.6b"`, `"1.7b"`. |
| `instruct_support` | bool | false | Whether the model accepts `--instruct` (custom / design only). |

## 2. Runtime Limits

| Field | Notes |
| --- | --- |
| `talker_n_ctx` | Talker LLM context cap. Must be ≤ `talker_n_ctx_train`. |
| `talker_n_ctx_train` | Training context length of the underlying weights. |
| `predictor_n_ctx` | Q1..Q15 predictor context cap. |
| `codec_num_codebooks` | **Must be 16** for the current runtime. |
| `codec_id_start` / `codec_id_end` | Codec token ID range (start inclusive, end exclusive). |
| `predictor_vocab_size` | Must equal `(codec_id_end - codec_id_start) * (codec_num_codebooks - 1)`. |

## 3. Special Tokens

| Group | Fields |
| --- | --- |
| Codec framing | `codec_pad_id`, `codec_bos_id`, `codec_eos_id` |
| Think gating | `codec_think_id`, `codec_nothink_id`, `codec_think_bos_id`, `codec_think_eos_id` |
| Text framing | `tts_pad_id`, `tts_bos_id`, `tts_eos_id` |

## 4. Generation Defaults

Pulled from upstream `generation_config.json`. Each field is overridable on the CLI.

| Field | Default | CLI override |
| --- | --- | --- |
| `default_max_new_tokens` | 400 | `--max-tokens` |
| `default_temperature` | 0.6 | `--temperature` |
| `default_top_p` | 1.0 | `--top-p` |
| `default_top_k` | 50 | `--top-k` |
| `default_repetition_penalty` | 1.05 | `--repetition-penalty` |
| `default_predictor_do_sample` | true | `--predictor-greedy` |
| `default_predictor_temperature` | 0.6 | `--predictor-temperature` |
| `default_predictor_top_p` | 1.0 | `--predictor-top-p` |
| `default_predictor_top_k` | 50 | `--predictor-top-k` |
| `default_seed` | 42 | `--seed` |
| `default_predictor_seed` | 45 | `--predictor-seed` |

## 5. Language / Speaker Maps

| Field | Shape |
| --- | --- |
| `language_map` | `{lowercase_name: language_id}` |
| `speaker_map` | `{lowercase_name: speaker_id}` — populated only for `custom` models |
| `speaker_dialect_map` | `{lowercase_name: lowercase_dialect_tag}` (optional) |

Keys are pre-lowercased by the writer; `ModelProfile::resolve_*` lowercases lookups, so callers do not normalize.

## 6. Mode Routing & Hard Errors

`--mode` is optional. Routing follows `model_type`:

- `base` → `base` (auto-switches to `clone` when `--reference` is given)
- `custom` → `custom`
- `design` → `design`

Hard errors (loader aborts):

- `base` model + `--instruct` → `--instruct is forbidden in base mode`
- `custom` / `design` + `--reference` → `mode 'clone' is incompatible with model_type ...`
- Any 0.6B model + `--instruct` (no model in this size class supports it)
- Talker weight other than `qwen3_tts_talker.q5_k.gguf`

## 7. Quality Gate

```bash
./build/lunavox-cli --help            # CLI smoke test
./build/lunavox-cli -m models/base_small -t "hello" -o out.wav
```

Single inference should finish in under 20 s on the smallest model. Always do a manual listening check before any quantization comparison — automated metrics never replace ears.
