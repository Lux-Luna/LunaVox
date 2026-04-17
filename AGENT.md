# AGENT.md — LunaVox project brain

High-performance C++ inference engine for **Qwen3-TTS** plus a unified
Python CLI / GUI / HTTP layer for pulling, converting, building, and
serving models. Cross-platform (Windows / Linux / macOS), pluggable
backends (CPU / CUDA / DML / Vulkan / CoreML / Metal), four Qwen3 modes
(Base / Clone / Custom / Design). Active development — layout / APIs /
CLI / C ABI / Python / docs all fair game. Out of scope: training,
non-Qwen3 architectures, backwards-compat shims.

## 1. Source of truth

| Topic | Location |
| --- | --- |
| C++ engine + C ABI | `src/*.cpp` `src/*.h` (root — **not** in the Python package); ABI: `src/lunavox_c_api.*` |
| Single-home C++ modules | `src/platform_utils.*` (all `#ifdef`), `src/provider_policy.*` (all EP selection) |
| CMake | `CMakeLists.txt` (`project(lunavox)`, `lunavox_*` targets) |
| Python package | `src/lunavox/` (via `package-dir = { "" = "src" }`) |
| CLI + runtime | `cli/main.py` + `runtime/{engine,voice,params,errors,_capi}.py` |
| Build + model | `build/*.py` (+ `libs.json`); `model/{config.py,downloader.py,pipeline.py,conversion/}` |
| GUI / HTTP-WS | `gui/` (customtkinter + pygame); `serve/` (FastAPI) |
| Core utilities | `core/{ui,logging,project,platform,deps}.py` |
| Docs + metadata + runtime dirs | `docs/{en,zh}/`; `pyproject.toml` (2.2.2); `models/ ref/ build/ logs/latest.log` |

## 2. Task routing

- C++ inference / RTF → `lunavox_engine.cpp`, `talker_predictor_llama.cpp`, `audio_decoder.cpp`
- Backend EP / platform `#ifdef` → `provider_policy.cpp` / `platform_utils.*` only (no leaks); dyn-lib via `platform::dynlib_*`
- C ABI change → `lunavox_c_api.*`, then sync `runtime/_capi.py::_bind_symbols`
- CLI / build / model catalog → `cli/`, `build/` (+ `libs.json`), `model/config.py`; Python platform branch → `core/platform.py` (build factory excepted)
- GUI / HTTP-WS → thin shells over `Engine`; handlers only assemble voice/params. New mode → `Voice.<mode>()` factory + one branch in `Engine._dispatch`

## 3. Commands

```bash
pip install -e ".[convert,dev]"     # dev install (plain `pip install -e .` for core CLI + GUI + serve)
lunavox bootstrap                   # pull → libs → build → smoke
lunavox model {pull,convert,list}; lunavox build [libs] [--clean]
lunavox {synth "text" -o out.wav, serve --batch-size auto, gui, doctor}
lunavox --profile quality synth …   # profile from ~/.lunavox/config.toml
cmake -S . -B build -G Ninja && cmake --build build -j
```

Shell on Windows is bash (git bash / MSYS) — forward slashes,
`/dev/null`, never `NUL`.

## 4. Testing & done definition

No mandatory CI / test suite. Minimum bar:

| Change | Verification |
| --- | --- |
| C++ engine | `cmake --build build -j` + `benchmark/run_benchmark.py` within prior RTF/TTFB |
| Backend policy | ≥ one GPU EP + CPU EP green |
| Model conversion | `validate_onnx_models.py` + artifact loads under `lunavox-cli` |
| Python / CLI | `pytest -x` + `lunavox doctor` + touched subcommand |
| Build drivers / GUI / Docs | `lunavox build --clean` / manual launch, clean `logs/latest.log` / links + code runnable |

Can't verify the target platform? Say so explicitly — don't pretend.

## 5. Code style

**C++** — C++17 (`CMAKE_CXX_EXTENSIONS OFF`); namespace `lunavox::`
everywhere (C API internals included); logging via `logger.h`, never
bare `std::cerr` / `fprintf(stderr)` (exceptions: `main.cpp::--help`,
`cli/stats_reporter` success). C API stays `extern "C"`, no STL across
ABI. Windows does not enable OpenMP.

**Python** — `from __future__ import annotations`; `Optional` + `Path`
hints; CLI is **typer**; Rich comes from `lunavox.core.ui.console`
(**never** `Console()` at module top); commands read state via
`_state(ctx)`; logging through `core.logging`; paths via `pathlib.Path`
+ `core.project.resolve_project_root`; heavy deps lazy via
`core.deps.ensure_dependency_group` (no top-level `import torch`).
User-facing strings default English; Chinese only in `docs/zh/` and
`代办.txt`.

**GUI** — thin shell over `runtime.Engine` + `Voice`; model/build ops
reuse CLI internals (e.g. `cli.synth_cmd._run_synth`). **No
`subprocess`** — no `Popen(['lunavox', …])` or CLI string-building.
Background work: `threading.Thread(daemon=True)` + `tk.after(0, …)`
onto the main thread. Translations: `i18n.Translator`; tokens:
`theme.py`.

**General** — default to **no comments**; add one only when WHY is
non-obvious. No "fix X bug" / "added for Y" notes (commit-message
material). No backwards-compat shims — delete old paths.

## 6. Safety

**Never** modify contents of `lib/ build/ logs/ models/ ref/`; the
`LICENSE` files; `pyproject.toml` `license` / `authors` / `version`
(release task only); `代办.txt` (user's); `hf_export/` (no real uploads).

**Confirm first**: C ABI signature changes, `libs.json` bumps, `lunavox
model pull` source switches, deleting whole `src/*.cpp` or
`src/lunavox/` modules. **Escalate**: multi-GB downloads, HF uploads,
version bumps or release tags, unexplained RTF regression, `代办.txt`
conflicts, cross-platform change the host cannot verify.

**Free**: rename / move / refactor / delete stale abstractions, CLI
flag renames, module or doc restructuring, C++ internals (non-ABI).

## 7. Commits, PRs, self-evolution

Default branch `main`, merges from `dev`. Short imperative commits
(English or Chinese). Never `--amend` a pushed commit or pass
`--no-verify` / `--no-gpg-sign`. Don't `git push` or open PRs unless
asked. Multi-file C++ + Python + docs changes: list the plan in chat
first. Update this file when an agent repeats a mistake, a new
subsystem lands, or a change flow stabilises — not for one-offs or
tutorials that belong in `README.md` / `docs/`.

---

Related: [README](README.md) · [CLI](docs/en/guide/cli_reference.md) · [usage](docs/en/guide/usage_tutorial.md) · [serve](docs/en/guide/serve.md) · [docker](docs/en/guide/docker.md) · [technical](docs/en/technical/) · [中文镜像](docs/zh/).
