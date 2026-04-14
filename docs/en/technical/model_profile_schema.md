# Model Profile Schema

`model_profile.json` is the contract between three sites:

| Site | Type / file | How it reads |
| --- | --- | --- |
| **C++ engine** | `lunavox::ModelProfile` in `src/model_profile.h` | `src/lunavox_engine.cpp::load_model_profile` via the JSON extractors in `src/json_utils.h` |
| **Python runtime** | `lunavox.model.ModelProfile` in `src/lunavox/model/profile.py` | `ModelProfile.load(model_dir)` used by GUI and scripts |
| **Disk artifact** | `models/<name>/model_profile.json` | Produced by `lunavox convert` (see `src/lunavox/model/pipeline.py::_write_model_profile`) |

Every field added to the JSON must land in **both** the C++ struct and
the Python dataclass in the same commit. The C++ side is authoritative
for inference — the Python dataclass is a permissive subset used for
display metadata. If the two disagree, the C++ one wins.

## Top-level fields

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `version` | int | 1 | Schema version. Increment on breaking changes. Readers must refuse unknown majors. |
| `model_type` | string | `"base"` | One of `base`, `custom`, `design`. Gates which synthesize modes the model supports. |
| `model_size` | string | `"unknown"` | Human label like `"0.6b"`, `"1.7b"`. Purely for display. |
| `instruct_support` | bool | false | Whether the model accepts an `instruct` prompt in custom/design modes. |

## Runtime limits

| Field | Type | Notes |
| --- | --- | --- |
| `talker_n_ctx` | int | Runtime cap for the talker LLM context window. Must be ≤ `talker_n_ctx_train`. |
| `talker_n_ctx_train` | int | The model's training context length. |
| `predictor_n_ctx` | int | Runtime cap for the Q1..Q15 predictor context. |
| `codec_num_codebooks` | int | Number of codebooks. **Must be 16** for the current runtime. |
| `codec_id_start` | int | Lower bound (inclusive) of codec token IDs. |
| `codec_id_end` | int | Upper bound (exclusive) of codec token IDs. |
| `predictor_vocab_size` | int | Must be divisible by `codec_num_codebooks - 1`; quotient must equal `codec_id_end - codec_id_start`. |

## Special tokens

| Field | Type | Purpose |
| --- | --- | --- |
| `codec_pad_id` / `codec_bos_id` / `codec_eos_id` | int | Codec pad / BOS / EOS |
| `codec_think_id` / `codec_nothink_id` | int | Think / nothink gating |
| `codec_think_bos_id` / `codec_think_eos_id` | int | Think-mode framing |
| `tts_pad_id` / `tts_bos_id` / `tts_eos_id` | int | Text pad / BOS / EOS |

## Generation defaults

Pulled from upstream `generation_config.json`. Each one can be
overridden on the CLI via the matching `--*` flag.

| Field | Type | Default | Overrides |
| --- | --- | --- | --- |
| `default_max_new_tokens` | int | 400 | `--max-tokens` |
| `default_temperature` | float | 0.6 | `--temperature` |
| `default_top_p` | float | 1.0 | `--top-p` |
| `default_top_k` | int | 50 | `--top-k` |
| `default_repetition_penalty` | float | 1.05 | `--repetition-penalty` |
| `default_predictor_do_sample` | bool | true | `--predictor-greedy` |
| `default_predictor_temperature` | float | 0.6 | `--predictor-temperature` |
| `default_predictor_top_p` | float | 1.0 | `--predictor-top-p` |
| `default_predictor_top_k` | int | 50 | `--predictor-top-k` |
| `default_seed` | int | 42 | `--seed` |
| `default_predictor_seed` | int | 45 | `--predictor-seed` |

## Language / speaker maps

| Field | Type | Shape |
| --- | --- | --- |
| `language_map` | object | `{lowercase_name: language_id}` |
| `speaker_map` | object | `{lowercase_name: speaker_id}` — populated only for `custom` models |
| `speaker_dialect_map` | object | `{lowercase_name: lowercase_dialect_tag}` — optional |

Keys are pre-lowercased by the writer; `lunavox::ModelProfile::resolve_*`
methods lowercase lookups before matching, so callers don't need to
normalize.

## Validation

The C++ loader calls `lunavox::ModelProfile::is_valid(&reason)` after
parsing. If it returns false, `Engine::load_models` aborts with the
reason string. Keep the check strict — silently accepting a malformed
profile would only surface later as garbled audio.

The Python `ModelProfile.from_dict` is intentionally permissive (missing
fields fall back to defaults) because Python consumers only care about
display metadata. If you need strict validation in Python, read the
C++ side: it is the authoritative loader.
