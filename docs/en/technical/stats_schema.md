# Stats Schema

Three producers emit the same structured stats:

| Producer | Surface | Source |
| --- | --- | --- |
| `lunavox-cli --stats-json report.json` | JSON file | `src/main.cpp` |
| `LunavoxAudio` from the C API | In-memory struct | `src/lunavox_c_api.h` |
| `lunavox.runtime.SynthesisStats` | Python dataclass | `src/lunavox/runtime/binding.py` |

The shared shape is pinned in `src/lunavox/core/stats_schema.py`. Adding a field means editing that module **and** `src/main.cpp` + `src/lunavox_c_api.h` + `src/lunavox/runtime/binding.py` in the same commit.

## Top-level `StatsJSON`

```jsonc
{
  "t_load_ms":    1714,    // wall time inside Engine::load_models
  "t_warmup_ms":   565,    // warmup portion of t_load_ms (decoder first-run)
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
  "timing_ms": { ... TimingMs ... },
  "stream":    { ... StreamStats ... },
  "mem":       { ... MemoryBytes ... }
}
```

`rtf = timing_ms.total / 1000 / audio_duration_s`. Lower is faster; `< 1.0` is faster than realtime.

## `TimingMs` (milliseconds)

| Field | Always populated | Description |
| --- | :---: | --- |
| `tokenize` | ✓ | Text → token IDs |
| `encode` | ✓ | Speaker encoder (0 when using a pre-computed embedding JSON) |
| `generate` | ✓ | LLM sequence generation (talker + predictor + sampling) |
| `decode` | ✓ | ONNX decoder session + post-processing |
| `total` | ✓ | Sum of the above + overhead |
| `first_audio` | ✓ | Wall time to first decoded PCM chunk (streaming pipeline) |
| `llama_prefill` / `llama_decode_loop` / `talker_post` / `predictor_sample` / `talker_decode` |   | Detailed sub-timings; require the `LUNAVOX_TIMING` build flag |
| `decoder_tensor_prep` / `decoder_ort_run` / `decoder_tensor_extract` / `decoder_state_trim` / `pcm_gather` |   | Same |

## `StreamStats`

| Field | Description |
| --- | --- |
| `first_chunk_frames` | Frames in the first decoded chunk (TTFB tuning knob) |
| `t_first_audio_ms` | Same value the C API exposes as `audio.first_audio_ms` |

## `MemoryBytes` (bytes)

| Field | Description |
| --- | --- |
| `rss_start` / `rss_end` | Process RSS at synth entry / exit |
| `rss_peak` | High-water RSS during the synth |
| `phys_start` / `phys_end` / `phys_peak` | macOS `phys_footprint` (equal to RSS on Windows / Linux) |

## C API Subset

`LunavoxAudio` exposes a flat subset of the above directly on the audio struct, so the C / Python binding does not need an extra round-trip:

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

The Python binding mirrors this subset as `SynthesisStats`. The finer `llama_*` / `decoder_*` sub-timings only appear in the JSON file output.

## Consumers

- `benchmark/run_benchmark.py` — reads `--stats-json` to compute the 100-run latency / TTFB / RTF / memory distribution that powers `benchmark/report.md`.
- `lunavox.gui.widgets.stats_card::StatsCard.update_stats` — renders `SynthesisStats` directly into the GUI's stats card (no legacy dict projection).
- New tooling should import from `lunavox.core.stats_schema` instead of poking at free-form dicts.
