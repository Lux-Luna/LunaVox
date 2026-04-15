# LunaVox CLI 指令汇总

`lunavox` 是 LunaVox 的单一入口，覆盖环境配置、模型管理、C++ 引擎构建、
进程内直接合成以及桌面 GUI。想要一键上手直接用 `bootstrap`，否则按需
运行各条子命令即可。

```powershell
pip install lunavox            # 核心 CLI
pip install "lunavox[gui]"     # + 桌面 GUI
pip install "lunavox[convert]" # + 原始权重 → GGUF 转换工具链
```

## 命令树

```
lunavox
├── bootstrap            一键安装：拉模型 → 下运行库 → 构建 → 冒烟合成
├── model
│   ├── pull             拉取预转换好的 GGUF/ONNX 产物
│   ├── convert          把原始 HF 权重转成 LunaVox 产物
│   └── list             查看目录 + 哪些模型已安装
├── build                构建 C++ 引擎（cmake 封装）
│   └── libs             下载 ONNX Runtime / llama.cpp 二进制
├── synth 文本           走 Python Engine 的进程内合成
├── gui                  启动桌面 GUI（需要 [gui] extra）
└── doctor               环境 + 依赖健康检查
```

## 1. `doctor` —— 系统健康检查

检查项目结构、工具链、运行库及当前 profile。任何问题反馈前先跑一下。

```bash
lunavox doctor
```

检查：项目根目录下 `src` / `lib` / `models`、`cmake` 是否在 `PATH`、
ONNX Runtime SDK 头文件、llama.cpp 运行库、`[convert]` extra 是否装齐、
当前生效的 profile。

## 2. `bootstrap` —— 一键安装

依次执行 **pull → libs → build → 进程内冒烟合成**。冒烟合成走的是 Python
侧原生 `Engine` + `Voice.base()`，不经过子进程，校验的正是真实调用
路径。

```bash
lunavox bootstrap
lunavox bootstrap --model base_small --platform win_cuda12
lunavox bootstrap --skip-test          # 只构建，不跑合成冒烟
```

## 3. `model` —— 目录管理

### `lunavox model pull`（推荐）

从社区镜像拉取已转换的 GGUF / ONNX 产物。

```bash
lunavox model pull
lunavox model pull --model base_small
```

### `lunavox model convert`

本地把原始 `.safetensors` 权重转成 LunaVox 产物。需要 `[convert]` extra，
过程需要几分钟。

```bash
lunavox model convert --model base_small --force
lunavox model convert --all
```

### `lunavox model list`

展示目录里所有条目以及本地是否已安装。

```bash
lunavox model list
```

## 4. `build` —— 本地引擎

### `lunavox build`

CMake 构建 C++ 引擎和 C ABI 共享库。

```bash
lunavox build
lunavox build --clean --j 8
lunavox build --toolchain msvc
```

### `lunavox build libs`

拉取特定平台的 ONNX Runtime + llama.cpp 二进制。

```bash
lunavox build libs
lunavox build libs --platform win_cuda12
# win_cuda13 / win_vulkan / win_cpu / linux_cuda / mac_arm64
```

## 5. `synth` —— 进程内合成

直接调用 Python `Engine` 并写出 WAV。既是冒烟测试的官方入口，也是
桌面 GUI 使用的同一条代码路径。

```bash
# 默认音色
lunavox synth "你好，来自 LunaVox。" -o output/hello.wav

# 参考音频克隆
lunavox synth "Okay, fine." \
  --voice clone --ref ref/ref_0.6B.json \
  -o output/cloned.wav

# 使用内置发音人并附带风格指令
lunavox synth "她说她中午就到。" \
  --voice custom --speaker Vivian --instruct "Use angry tone." \
  -o output/custom.wav

# 用文字描述设计音色
lunavox synth "就在最上面的抽屉里，不对，怎么是空的？" \
  --voice design --instruct "Speak in an incredulous tone." \
  -o output/designed.wav
```

可覆盖参数：`--model`、`--temperature`、`--top-p`、`--top-k`。命令行
没显式指定的参数会依次回落到当前 profile → 环境变量 → 默认值。

## 6. `gui` —— 桌面应用

```bash
lunavox gui
```

需先执行 `pip install "lunavox[gui]"`。新版 GUI 是左侧栏 + 三视图布局
（合成 / 素材库 / 设置），底层调用与 `lunavox synth` 完全相同的
`Engine` API。

## 7. 模型 ID 对照表

| 模型 ID | 完整名称 | 备注 |
| :--- | :--- | :--- |
| `base_small` | Qwen3-TTS 0.6B Base | 快速、均衡、适合低端设备 |
| `custom_small` | Qwen3-TTS 0.6B Custom | 内置发音人 ID |
| `base` | Qwen3-TTS 1.7B Base | 高保真，建议 GPU |
| `custom` | Qwen3-TTS 1.7B Custom | 大模型发音人定制 |
| `design` | Qwen3-TTS 1.7B Design | 文字描述生成音色 |

## 8. Profile 与配置文件

LunaVox 在每次执行时都会读取 `~/.lunavox/config.toml`。文件结构是一个
`[default]` 表 + 任意多个 `[profile.<name>]` 覆盖。优先级从高到低：

1. 命令行开关（`--temperature 0.9`、`--model base`）
2. 环境变量（`LUNAVOX_MODEL`、`LUNAVOX_BACKEND` 等）
3. 通过 `--profile NAME` 选中的 `[profile.NAME]` 表
4. `[default]` 表
5. 硬编码默认值

示例 `config.toml`：

```toml
[default]
model = "base_small"
backend = "auto"
n_threads = 4

[profile.quality]
backend = "cuda"
temperature = 0.7
top_p = 0.9

[profile.fast]
backend = "vulkan+dml"
temperature = 0.8
```

```bash
lunavox --profile quality synth "请用高保真合成。" -o out.wav
```

## 9. 全局开关

对每一条 `lunavox` 子命令都生效：

- `--profile <NAME>` —— 从 `config.toml` 选择 `[profile.<NAME>]`
- `--project-root <PATH>` —— 显式指定项目根（开发时）
- `--yes` —— 自动确认所有提示（CI）
- `--no-install` —— 禁止自动修复 Python 依赖
- `--verbose` —— 构建 / 下载原始输出

## 相关文档

- [模型 Profile 与运行时契约](../technical/model_profile.md)
- [使用教程（`lunavox synth` 各模式）](usage_tutorial.md)
- [Runtime API](../api/runtime.md)
