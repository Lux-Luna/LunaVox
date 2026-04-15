# Stats Schema

The stats schema is the **contract between producer and consumer** for
the `--stats-json` output of `lunavox-cli`. Three producers emit the
same shape:

1. `src/main.cpp` writes the JSON file when the user passes
   `--stats-json report.json`.
2. `src/lunavox_c_api.cpp::to_c_audio` fills `LunavoxAudio` with the
   same fields so the Python ctypes binding gets them as struct members.
3. `src/lunavox/runtime/binding.py::SynthesisStats` is the Python
   dataclass the GUI and any embedding script consumes.

Consumers like `benchmark/run_benchmark.py` and the GUI can
`from lunavox.core.stats_schema import StatsJSON` instead of reaching
into free-form dicts.

## TypedDicts (structural types)

::: lunavox.core.stats_schema.TimingMs
    options:
      show_root_heading: true

::: lunavox.core.stats_schema.StreamStats
    options:
      show_root_heading: true

::: lunavox.core.stats_schema.MemoryBytes
    options:
      show_root_heading: true

::: lunavox.core.stats_schema.RunStats
    options:
      show_root_heading: true

::: lunavox.core.stats_schema.StatsJSON
    options:
      show_root_heading: true

## Parser for downstream consumers

::: lunavox.core.stats_schema.ParsedStats
    options:
      show_root_heading: true

::: lunavox.core.stats_schema.parse_stats_json
    options:
      show_root_heading: true
