# Stats Schema

Three producers emit the same structured stats:

| Producer | Surface | Path |
| --- | --- | --- |
| `lunavox-cli --stats-json report.json` | JSON file | `src/main.cpp` |
| `LunavoxAudio` struct returned by the C API | In-memory | `src/lunavox_c_api.h` |
| `lunavox.runtime.SynthesisStats` dataclass | Python | `src/lunavox/runtime/binding.py` |

The shared schema is pinned in `src/lunavox/core/stats_schema.py` as
`TimingMs` / `MemoryBytes` / `RunStats` / `StatsJSON` TypedDicts.
Adding a field means editing that module **and** `src/main.cpp` +
`src/lunavox_c_api.h` + `src/lunavox/runtime/binding.py` in the same
commit.

## Top-level `StatsJSON`

```jsonc
{
  "t_load_ms":   1714,     // Wall time spent inside Engine::load_models
  "t_warmup_ms":  565,     // Warmup portion of t_load_ms (decoder first-run)
  "runs": [ ... RunStats ... ]
}
```

## `RunStats`

```jsonc
{
  "run_id":              1,
  "sample_rate":         24000,
  "n_samples":           71040,
  "audio_duration_s":    2.96,
  "rtf":                 0.175,
  "effective_language_id": -1,
  "timing_ms":           { ... TimingMs ... },
  "mem":                 { ... MemoryBytes ... }
}
```

`rtf = timing_ms.total / 1000 / audio_duration_s`. Lower is faster;
`< 1.0` is faster than realtime.

## `TimingMs` (milliseconds)

| Field | Always populated | Description |
| --- | :---: | --- |
| `tokenize` | ✓ | Text → token IDs |
| `encode` | ✓ | Speaker encoder (0 when using pre-computed embedding JSON) |
| `generate` | ✓ | LLM sequence generation (talker + predictor + sampling) |
| `decode` | ✓ | ONNX decoder session + post-processing |
| `total` | ✓ | Sum of the above + overhead |
| `llama_prefill` |   | Detailed breakdown; requires `LUNAVOX_TIMING` build flag |
| `llama_decode_loop` |   | Same |
| `talker_post` / `predictor_sample` / `talker_decode` |   | Same |
| `decoder_tensor_prep` / `decoder_ort_run` / `decoder_tensor_extract` / `decoder_state_trim` |   | Same |
| `pcm_gather` |   | Same |

## `MemoryBytes` (bytes)

| Field | Description |
| --- | --- |
| `rss_start` / `rss_end` | Process RSS at synth entry / exit |
| `rss_peak` | High-water RSS during the synth |
| `phys_start` / `phys_end` / `phys_peak` | macOS `phys_footprint` (equal to RSS on Windows/Linux) |

## C API fields on `LunavoxAudio`

The C API exposes a **flat** subset of the above directly on the audio
struct for convenience (no extra round-trip through `lunavox_get_stats`):

```c
typedef struct LunavoxAudio {
    const float* samples;
    int32_t  n_samples;
    int32_t  sample_rate;
    int64_t  t_tokenize_ms;
    int64_t  t_encode_ms;
    int64_t  t_generate_ms;
    int64_t  t_decode_ms;
    int64_t  t_total_ms;
    int64_t  audio_duration_ms;
    float    rtf;
    uint64_t rss_peak_bytes;
    uint64_t rss_end_bytes;
} LunavoxAudio;
```

The Python binding (`lunavox.runtime.SynthesisStats`) mirrors this
subset. The finer `llama_*` / `decoder_*` sub-timings only appear in
the `--stats-json` file output.

## Consumers

- `tests/bench_baseline.py` — reads the CLI `--stats-json` file to
  build the regression matrix. Uses `stats_schema.parse_stats_json` for
  the structural load.
- `GUI/engine.py::format_metrics` — projects `SynthesisStats` into the
  string dict the report panel displays.
- Any future notebook / benchmark should import from
  `lunavox.core.stats_schema` instead of reaching into free-form dicts.
