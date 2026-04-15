# AGENT.md — LunaVox 项目大脑

> **项目状态（2026-04）：活跃开发期。**
> 架构、目录、API、CLI、C++ 接口、Python 包布局、模型配置、文档结构均可直接演进。
> **不保留向后兼容路径**：遇到过时代码、过时命名、过时抽象，直接删改重写，不需要 shim、不需要 deprecated 标记、不需要迁移提示。
> 除非用户显式要求“保留旧行为”或“最小改动”，否则默认走“正确重构”而不是“兼容补丁”。

---

## 1. Mission

LunaVox 是一个面向 **Qwen3-TTS** 的高性能 C++ 推理引擎，外加一个统一的 Python CLI / GUI 用于模型拉取、转换、构建和测试。

核心目标：
- C++ 运行时轻量、低延迟、跨平台（Windows / Linux / macOS，含 Apple Silicon）
- 硬件后端可插拔：CPU / CUDA / DML / Vulkan / CoreML / Metal
- 覆盖 Qwen3-TTS 的四种模式：Base 合成、Voice Cloning、Custom Voice、Voice Design
- 工具链闭环：`lunavox` CLI 一键完成 pull → convert → download-libs → build → 测试

**不在当前范围**：
- 训练 / 微调（仅做推理与格式转换）
- 兼容非 Qwen3-TTS 的其他 TTS 架构
- 保留旧版本 CLI/接口/目录命名

## 2. Current priorities

参考根目录 `代办.txt`（这是工作清单，不是文档）：
- 已完成项占绝大多数，剩余活跃方向：
  - 流式 / 预加载延迟优化
  - 内存优化
  - 参考 json 训练、专家声音训练（研究性）
  - 可能的改名动作
- 当 `代办.txt` 与本文件冲突时，以用户当前对话为准，其次以 `代办.txt` 为准。

## 3. Source of truth

| 主题 | 真源位置 |
| --- | --- |
| C++ 引擎实现 | `src/*.cpp` `src/*.h`（根 `src/`，**不是** Python 包） |
| C API（对外稳定面） | `src/lunavox_c_api.{h,cpp}` |
| C++ 构建系统 | `CMakeLists.txt`（`project(lunavox)`，所有目标 `lunavox_*`） |
| 平台抽象（#ifdef 单一家） | `src/platform_utils.{h,cpp}` |
| 音频 I/O (WAV + resample) | `src/audio_io.{h,cpp}` |
| Python 包（CLI / 构建 / 模型工具链） | `src/lunavox/` |
| CLI 入口 | `src/lunavox/cli/main.py`（`lunavox = lunavox.cli.main:run`） |
| ctypes 运行时绑定（GUI / 脚本直调 C API） | `src/lunavox/runtime/binding.py` |
| 构建驱动（跨平台） | `src/lunavox/build/{base,windows,linux,macos}.py` + `main.py` + factory in `__init__.py` |
| 预编译库清单 | `src/lunavox/build/libs.json` |
| 模型目录（单一真源） | `src/lunavox/model/config.py`（`MODELS` + `ModelSpec`） |
| 模型下载 / 转换流水线 | `src/lunavox/model/{downloader,pipeline}.py` + `model/conversion/` |
| 依赖策略（按需安装 convert 组） | `src/lunavox/core/deps.py` |
| Rich Console 单例 | `src/lunavox/core/ui.py` |
| Session 日志 | `src/lunavox/core/logging.py`（`session_start` + `append`） |
| Python 平台抽象 | `src/lunavox/core/platform.py` |
| GUI | `GUI/main.py`、`GUI/components/`、`GUI/engine.py` — 通过 `lunavox.runtime` + `lunavox.*` 直接调用，不 subprocess |
| 用户文档（英文） | `docs/en/{guide,install,technical,benchmark}/` |
| 用户文档（中文） | `docs/zh/` |
| 运行时规范 / 合成通路 | `docs/en/technical/{runtime_specs,synthesis_pathway}.md` |
| CLI 参考 | `docs/en/guide/cli_reference.md` |
| Python 包元数据 / 版本 | `pyproject.toml`（当前 2.1.6） |
| 发布 CLI-only 源码分支 | GitHub `cli-only` 分支（不在本仓库内） |
| 模型权重（本地缓存） | `models/{base,base_small,custom,custom_small,design}/` |
| 参考音频 / 特征 | `ref/ref.wav`、`ref/ref_0.6B.json`、`ref/ref_1.7B.json` |
| 构建产物 | `build/`（含 `lunavox-cli[.exe]`） |
| 运行日志 | `logs/latest.log`（CLI 会话日志） |

> C++ 源码在仓库根 `src/` 下直接平铺，而 Python 包在 `src/lunavox/` 下。修改时注意不要混淆两者——**根 `src/*.cpp` 不属于 `lunavox` Python 包**，`pyproject.toml` 的 `package-dir = { "" = "src" }` 只暴露 `lunavox/` 子目录。

## 4. Repository map

```
lunavox/
├── CMakeLists.txt          # C++ 构建入口（project(lunavox) + lunavox_* 目标）
├── pyproject.toml          # Python 包定义（name=lunavox, ver=2.1.6）
├── README.md               # 用户视角说明（英文）
├── 代办.txt                # 滚动 TODO（非正式，持续演进）
├── src/                    # C++ 引擎源码 + Python 包
│   ├── main.cpp                  # lunavox-cli 可执行入口
│   ├── lunavox_engine.{h,cpp}    # 顶层合成流水线（lunavox::Engine）
│   ├── lunavox_c_api.{h,cpp}     # C ABI facade (lunavox.dll / liblunavox)
│   ├── platform_utils.{h,cpp}    # 唯一允许 #ifdef _WIN32/__APPLE__ 的文件
│   ├── audio_io.{h,cpp}          # WAV load/save + windowed-sinc resample
│   ├── audio_decoder.{h,cpp}     # ONNX decoder session（原 onnx_audio_runtime）
│   ├── provider_policy.{h,cpp}   # EP 选择（唯一入口，其他文件禁止 AppendExecutionProvider*）
│   ├── talker_predictor_llama.*  # LLM 序列预测
│   ├── llama_wrapper.*           # llama.cpp 动态加载封装
│   ├── text_tokenizer.*          # 文本前处理 / 语言检测
│   ├── nvml_monitor.*            # GPU 监控
│   ├── assets_manager.*          # 模型资产定位
│   ├── model_profile.h           # 模型规格描述 (lunavox::ModelProfile)
│   ├── string_utils.h / timing_utils.h / format_utils.h
│   ├── json_utils.h / logger.*
│   └── lunavox/                  # Python 包
│       ├── cli/main.py           # typer app
│       ├── runtime/binding.py    # ctypes 绑定 → liblunavox
│       ├── build/                # base/windows/linux/macos + factory
│       ├── core/                 # ui, logging, project, platform, deps
│       └── model/                # config(MODELS), downloader, pipeline + conversion/
├── GUI/                    # customtkinter 桌面 GUI（薄壳，通过 lunavox.runtime 直调 C API）
│   ├── main.py  main_setup.py  engine.py  i18n.py
│   └── components/{header,report,setup_page}.py
├── lib/                    # 运行时库（ONNX Runtime / llama.cpp），由 download-libs 填充
│   ├── onnx/               # include/ + lib/
│   └── llama/              # 预编译二进制
├── models/                 # 本地模型目录（pull-model 填充）
├── ref/                    # 克隆参考素材
├── docs/{en,zh}/           # 面向用户的文档（非 agent 运行手册）
├── build/                  # CMake 构建产物
└── logs/latest.log         # CLI 会话日志（CLI 自管）
```

### 任务路由（常见改动 → 首选入口）

- **改 C++ 推理路径 / 合成质量 / RTF**：`src/lunavox_engine.cpp`、`src/talker_predictor_llama.cpp`、`src/audio_decoder.cpp`
- **改后端策略（EP 选择 / DML / Vulkan / CUDA）**：`src/provider_policy.cpp`（唯一入口）
- **改平台差异 (#ifdef / mmap / LoadLibrary / argv / process memory)**：`src/platform_utils.{h,cpp}` —— 其他 C++ 文件禁止出现 `#ifdef _WIN32/__APPLE__/__linux__`
- **改文本前处理 / 语言检测**：`src/text_tokenizer.cpp`
- **改音频 I/O (WAV 读写、resample)**：`src/audio_io.{h,cpp}`
- **改 CLI 命令 / 参数**：`src/lunavox/cli/main.py`
- **改构建流程（编译、拷 DLL、平台差异）**：`src/lunavox/build/`（base + windows/linux/macos + `get_builder_class` 工厂）
- **改运行时库下载源**：`src/lunavox/build/libs.json` + `lib_downloader.py`
- **改模型目录清单**：`src/lunavox/model/config.py`（`MODELS` 字典 + `ModelSpec`）—— `downloader.py` 和 CLI 都从这里读
- **改模型下载 / 转换 / 量化**：`src/lunavox/model/`，特别是 `conversion/`
- **改 Python 平台差异（`sys.platform`）**：`src/lunavox/core/platform.py`（唯一入口；build factory 是允许的例外）
- **改 Console / 日志**：`src/lunavox/core/ui.py`（Rich 单例）+ `src/lunavox/core/logging.py`（`session_start` / `append`）
- **改 GUI**：`GUI/`（薄壳，通过 `lunavox.runtime.Engine` + `lunavox.*` 直调；禁止 subprocess）
- **改 ctypes 绑定**：`src/lunavox/runtime/binding.py`（随 C API 变更同步）
- **改 C ABI**：`src/lunavox_c_api.*`——稳定对外面，改动必须同步 `src/lunavox/runtime/binding.py`、`GUI/engine.py` 和任何外部绑定
- **改依赖自动安装策略**：`src/lunavox/core/deps.py`

## 5. Commands

> 所有命令在仓库根运行。Windows 用户注意：shell 是 bash（git bash / MSYS），请用正斜杠路径与 `/dev/null`，不要用 `NUL`。

### Python 工具链
```bash
# 开发安装（推荐，可编辑模式 + convert + dev）
pip install -e ".[convert,dev]"

# 或：只装核心 CLI
pip install -e .

# CLI 一键闭环（拉模型 + 下运行库 + 构建 + 交互测试）
lunavox bootstrap

# 单步
lunavox pull-model           # 从 HF 拉预转换模型
lunavox download-libs        # 下 ONNX Runtime + llama.cpp 到 lib/
lunavox build --clean        # 调用 CMake 构建 C++ 引擎
lunavox convert <model>      # 本地权重 → onnx/gguf/embedding
lunavox doctor               # 环境自检
```

### C++ 构建（直接 CMake，不经 CLI）
```bash
cmake -S . -B build -G Ninja
cmake --build build -j
# 产物：build/lunavox-cli[.exe]
```

### 推理冒烟测试
```bash
# Voice Cloning（最小闭环）
./build/lunavox-cli.exe \
  -m models/base_small \
  -r ref/ref_0.6B.json \
  -t "Hello from LunaVox." \
  -o output/smoke.wav \
  --stats-json output/smoke.json
```

### 测试 / 检查
- **没有 pytest 套件**（`[dev]` 装了 pytest 但仓库未托管单元测试）。现有自动化只有 `tests/bench_baseline.py`——对所有模型 × 模式跑一轮 CLI 合成并落 `tests/results/baseline_<ts>/`，是 C++ 引擎改动的回归基准。结果目录已 gitignore，不要提交。
- 改动 Python 工具链时：
  - 跑 `python -c "import lunavox; from lunavox.cli.main import app"` 做 import 级冒烟
  - 跑 `lunavox doctor` 做环境级冒烟
  - 对 `conversion/` 改动，跑 `src/lunavox/model/conversion/validate_onnx_models.py`
- C++ 改动：`cmake --build build -j` 必须过；随后跑 `python tests/bench_baseline.py` 并与前一次 `tests/results/baseline_*/summary.json` 对比 RTF / 延迟，或用下面的冒烟命令对比 `--stats-json` 的 RTF。

### GUI
```bash
python GUI/main.py
```

## 6. Code style rules（LunaVox 特有）

**C++（根 `src/`）**
- C++17，`CMAKE_CXX_EXTENSIONS OFF`。不要引入 C++20 特性。
- **命名空间 `lunavox::`**。所有顶层类型进这个命名空间，没有例外（包括 C API 内部实现）。
- 所有对外暴露的运行时错误都走 `logger.h`，不要裸 `std::cerr` / `fprintf(stderr,...)`。例外：`main.cpp` 的 `--help` 可以直接 stdout；`cli/stats_reporter` 的成功消息可以 stdout。
- ONNX Runtime EP 选择集中在 `provider_policy.cpp`，其他位置禁止直接 `SessionOptions::AppendExecutionProvider_*`。
- C API 文件（`lunavox_c_api.*`）必须保持 `extern "C"`、无 STL 类型穿越 ABI。改动 C API 时必须同步 `src/lunavox/runtime/binding.py`。
- **所有 `#ifdef _WIN32/__APPLE__/__linux__`** 只允许出现在 `src/platform_utils.cpp`。其他文件需要平台分支时，往 `platform_utils.{h,cpp}` 加接口。
- 动态库加载必须走 `platform::dynlib_open/close/symbol`。Windows 下使用 `LOAD_WITH_ALTERED_SEARCH_PATH`（已封装），不要直接调 `LoadLibraryA`。
- Windows 上不启用 OpenMP（见 `CMakeLists.txt`），不要在代码里假设 `_OPENMP` 已定义。
- 日志输出必须走项目 `logger`，以保证被写入 `logs/latest.log`。

**Python 包（`src/lunavox/`）**
- `from __future__ import annotations`；类型注解用 `Optional`、`Path`。
- CLI 用 **typer**（不是 argparse / click），富文本用 **rich** 但只从 `lunavox.core.ui.console` 导入——**禁止**在模块顶部 `Console()`。
- 所有 CLI 命令必须通过 `_state(ctx)` 获取 `RuntimeState`，不要直接读 `ctx.obj`。
- 日志必须走 `lunavox.core.logging` (`session_start`/`append`)，不要 `open(log_file, "a")`。
- 依赖按需安装走 `core.deps.ensure_dependency_group`，不要在模块顶部 `import torch`（会污染纯 CLI 路径）。
- 路径一律 `pathlib.Path`，不混用字符串路径。路径解析走 `lunavox.core.project.resolve_project_root`。
- 用户可见字符串默认英文（README / CLI help / 日志）；中文仅限 `docs/zh/` 与 `代办.txt`。
- 平台差异走 `lunavox.core.platform` (`shared_lib_name`, `executable_suffix`, `is_*`)。`build/__init__.py` 的 `get_builder_class`/`get_resolver_class` 工厂是允许的例外。
- 模型目录的唯一真源是 `lunavox.model.config.MODELS` + `ModelSpec`。新增模型只改 `config.py`。

**GUI（`GUI/`）**
- customtkinter + pygame，i18n 走 `GUI/i18n.py`。
- UI 组件按页面拆入 `components/`，不要把逻辑塞进 `main.py`。
- **薄壳**：GUI 不自己做业务。合成走 `lunavox.runtime.Engine`（ctypes 直调 C API），build/pull-model/download-libs 走 `lunavox.build.main.run_build` / `lunavox.model.ModelDownloader` / `lunavox.build.lib_downloader.download_platform_libs`。
- **禁止 `subprocess`**：没有任何 `subprocess.Popen(['lunavox', ...])` 或调 `lunavox-cli`。后台长任务用 `threading.Thread(daemon=True)` + `tk.after(0, ...)` 回调主线程。

**通用**
- 默认 **不写注释**。只有 WHY 不明显（硬件约束、平台差异、兼容性假设、非直觉算法）才加一行。
- 不写"修复 X bug"、"为 Y 加"这类与历史相关的注释——那是 commit message 的事。
- **不加向后兼容 shim**。项目活跃演进，直接删旧路径。

## 7. Testing & Done definition

没有强制 CI / 强制测试套件。对"完成"的最低标准：

| 改动类型 | 最小验证 |
| --- | --- |
| C++ 引擎 | `cmake --build build -j` 通过 + `python tests/bench_baseline.py` 对上一次 baseline RTF/延迟未劣化（或手动 Voice Cloning 冒烟 + `--stats-json` 对比） |
| 后端策略（`provider_policy`） | 至少在一个 GPU EP 和 CPU EP 下分别跑通 |
| 模型转换 (`conversion/`) | `validate_onnx_models.py` 通过，产物能被 `lunavox-cli` 加载 |
| CLI | `lunavox doctor` 通过 + 改动的子命令走一次 |
| 构建驱动 | 至少在当前宿主平台跑完 `lunavox build --clean` |
| GUI | 手动启动 + 目标交互路径 + 检查 `logs/latest.log` 无异常栈 |
| 文档 | 链接可达，代码块可复制粘贴运行 |

**如果无法在当前环境验证（例如改了 macOS CoreML 路径但宿主是 Windows），必须在 PR / 回复里显式声明"未在目标平台验证"。不要假装跑过。**

## 8. Safety boundaries

**禁止**：
- 修改 `lib/onnx/` 或 `lib/llama/` 下的内容——那是 `download-libs` 的输出，应通过 `libs.json` + `lib_downloader.py` 控制。
- 修改 `build/` 或 `logs/` 下的产物（`logs/latest.log` 由 CLI 自己重写）。
- 修改 `models/`、`ref/` 下的二进制 / 权重文件——只能通过 `pull-model` / `convert` 生成。
- 修改 `LICENSE`、`models/LICENSE`、`pyproject.toml` 里的 `license` 字段、`authors` 字段，除非用户显式要求。
- 升版 `pyproject.toml` 里的 `version`——除非发布任务里明确说要升。
- 改动 HuggingFace 上传 / 模型分发相关脚本时，不要触发实际上传（`hf_export/`）。
- 修改 `代办.txt`——那是用户自己的滚动清单，不要主动整理它，除非用户让你加/划掉某项。

**需要先确认**：
- C ABI (`lunavox_c_api.*`) 的签名变更（虽然不保留兼容，但 GUI / CLI / 外部绑定都依赖它，改动面大）
- `libs.json` 中运行时库的版本切换（会影响所有平台用户的下一次 `download-libs`）
- `pull-model` 指向的 HF 仓库名 / 路径
- 删除 `src/*.cpp` 或 `src/lunavox/` 下整块文件

**可以自由做**：
- 重命名、移动、重构、删除过时抽象
- 改 CLI 子命令名 / 参数
- 重构 Python 模块布局
- 改 C++ 内部接口（非 C API）
- 改文档结构

## 9. PR / commit workflow

- 默认分支：`main`。历史合并来自 `dev`（见最近 commit）。
- Commit message 风格看 `git log` —— 简短祈使句，英文或中文都接受，不强制 conventional commits。
- **不要** `--amend` 已推送的 commit。失败就新开一个 commit。
- **不要** 加 `--no-verify`、`--no-gpg-sign`。
- agent 默认不直接 `git push`、不直接开 PR——除非用户明确说"push"或"提 PR"。
- 涉及多文件大改时，先在对话里给出改动计划让用户确认，再动手；不需要写到文件里。

## 10. When to create a plan

大多数任务不需要独立的 plan 文件。但这些情况应先在对话里列出改动清单再动：
- 同时涉及 C++ 引擎 + Python CLI + 文档
- 重命名公开符号（CLI 命令、C API、Python 包模块）
- 修改构建驱动跨越多个平台
- 涉及模型转换流水线的格式变更

不要把 plan 写进文件（`PLANS.md` / `TODO.agent.md` 等）——这个仓库只用 `代办.txt`，并且那是用户自己的。

## 11. When to escalate to a human

遇到以下情况停手并问用户：
- 需要真的访问网络执行 `pull-model` / `download-libs`（可能下载数 GB）
- 需要实际触发 HuggingFace 上传
- 需要 bump `pyproject.toml` 版本或打 release tag
- C++ 改动导致 RTF 明显劣化且根因不明
- `代办.txt` 和用户当前指令出现冲突
- 需要删除 `models/`、`lib/`、`build/`、`logs/` 下任何已存在的内容
- 需要改动 `LICENSE` 或 `models/LICENSE`
- 跨平台改动但当前宿主无法验证目标平台

## 12. Related docs

- [README.md](README.md) — 用户视角总览
- [docs/en/guide/cli_reference.md](docs/en/guide/cli_reference.md) — CLI 参数手册
- [docs/en/guide/usage_tutorial.md](docs/en/guide/usage_tutorial.md) — 推理三模式用例
- [docs/en/technical/runtime_specs.md](docs/en/technical/runtime_specs.md) — 运行时规格
- [docs/en/technical/synthesis_pathway.md](docs/en/technical/synthesis_pathway.md) — 合成通路细节
- [docs/en/technical/model_profile_schema.md](docs/en/technical/model_profile_schema.md) — `ModelProfile` 字段契约（C++ ↔ Python ↔ JSON）
- [docs/en/technical/stats_schema.md](docs/en/technical/stats_schema.md) — `--stats-json` / `LunavoxAudio` 字段契约
- [docs/en/install/cuda12_windows.md](docs/en/install/cuda12_windows.md) / [cuda13_windows.md](docs/en/install/cuda13_windows.md) — CUDA 依赖
- [docs/en/benchmark/windows_performance.md](docs/en/benchmark/windows_performance.md) — Windows 性能基准
- [docs/zh/](docs/zh/) — 中文镜像文档

---

## 自我演进规则

本文件应随项目演进更新。触发条件：
1. agent 连续两次在同一处犯同类错误 → 加一条具体规则
2. 出现新的任务入口 / 构建命令 / 子系统 → 更新第 3–5 节
3. 某类文件被反复误改 → 加入第 8 节 "禁止"
4. 某类改动流程稳定下来 → 加入第 7 节或第 9 节

不做的事：
- 不把偶发问题写成永久规则
- 不把个人偏好伪装成硬性标准
- 不无限增长——超过当前长度 1.5 倍时考虑把章节下沉到 `docs/`
- 不写与 `README.md` / `docs/` 重复的用户教程
