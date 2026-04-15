# LunaVox 0.6B Benchmark

A statistically meaningful (100-run) latency + memory benchmark for the
`models/base_small` model, driven by `build/lunavox-cli.exe` and
`ref/ref_0.6B.json`. Covers three backend configurations in a single
report: **Full CPU**, **CUDA 13**, and **Universal GPU (Vulkan + DirectML)**.

## What the script measures

- **Latency**: per-run `total_ms` → mean / median / p50 / p95 / p99 / min / max / stddev
- **TTFB**: `stream.t_first_audio_ms` with the same distribution stats
- **RTF**: real-time factor (synth_wall / audio_duration) → mean / median / p95, plus realtime multiplier
- **Stage breakdown**: mean per-run tokenize / encode / generate / decode
- **Memory**: peak RSS and physical memory from per-run `mem` stats
- **VRAM**: external pynvml sampling at 100 ms intervals (delta above baseline)
- **Throughput**: chars/sec derived from fixed 25-word English text
- **Load/warmup**: one-shot `t_load_ms` and in-load decoder `t_warmup_ms`

Fixed constants (do not override via CLI — keeping every run comparable):

| Knob | Value |
| :--- | :--- |
| Text | `"LunaVox is a lightweight on-device text-to-speech engine built on Qwen3, optimized for low-latency streaming synthesis on consumer GPUs."` |
| Warm-up runs | 5 |
| Repeat runs | 100 |
| Model | `models/base_small` |
| Voice reference | `ref/ref_0.6B.json` |

Warm-up runs execute inside the CLI's `--warmup` loop before the `--repeat`
loop and are **not** written to the stats JSON `runs` array, so all 100
recorded samples are post-warmup.

## How to run

Each backend requires its own build of `lunavox-cli.exe`. Switch conda envs
and rebuild between runs:

```bash
# 1. Universal GPU (current default Vulkan+DirectML build)
lunavox build
python benchmark/run_benchmark.py --backend vulkan

# 2. CUDA 13
conda activate cuda13
lunavox build
python benchmark/run_benchmark.py --backend cuda

# 3. Full CPU
conda activate <cpu-env>
lunavox build
python benchmark/run_benchmark.py --backend cpu

# 4. Merge into benchmark/report.md
python benchmark/run_benchmark.py --aggregate
```

The script records `git rev-parse HEAD` into each per-backend
`meta_<backend>.json`. `--aggregate` warns if the three runs used different
commits (which would make the comparison invalid).

## Outputs

```
benchmark/
  run_benchmark.py
  report.md                        # written by --aggregate
  results/
    stats_vulkan.json              # raw --stats-json from lunavox-cli
    summary_vulkan.json            # aggregated metrics
    meta_vulkan.json               # git / env / host / CLI mtime
    stats_cuda.json, summary_cuda.json, meta_cuda.json
    stats_cpu.json,  summary_cpu.json,  meta_cpu.json
```

## Dependencies

- Python 3.10+
- `pynvml` (optional) — without it, VRAM columns fall back to `n/a` with a
  `vram_skipped_reason`. The CPU backend run is still valid without NVML.

## Troubleshooting

- **`CLI binary not found`**: run `lunavox build` in the active env first.
- **100 runs but `len(runs) < 100`**: a synth call failed mid-loop. Re-run.
- **VRAM peak = 0 for CUDA/Vulkan**: pynvml likely couldn't open the device
  (driver/permission); check `summary_<backend>.json → memory.vram_skipped_reason`.
- **Numbers differ a lot between re-runs**: the benchmark is sensitive to
  background GPU/CPU load; close other workloads and retry.
