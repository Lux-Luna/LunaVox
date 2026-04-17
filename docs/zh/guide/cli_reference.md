# LunaVox CLI 指令汇总

`lunavox` 是 LunaVox 的单一入口，覆盖环境配置、模型管理、C++ 引擎构建、
进程内合成以及桌面 GUI。想要一键上手用 `bootstrap`，否则按需运行各条
子命令即可。

```powershell
pip install lunavox             # 核心 CLI + GUI + HTTP/WebSocket 服务层
pip install "lunavox[convert]"  # + 原始权重 → GGUF 转换工具链（可选）
```

## 命令树

```
lunavox
├── bootstrap            一键安装：拉模型 → 下运行库 → 构建 → 冒烟合成
├── model {pull|convert|list}
├── build [libs]         C++ 引擎构建，或拉取运行库
├── synth 文本           走 Python Engine 的进程内合成
├── serve                HTTP + WebSocket 服务层
├── gui                  启动桌面 GUI
└── doctor               环境 + 依赖健康检查
```

## 1. `doctor` —— 系统健康检查

检查项目结构（`src` / `lib` / `models`）、`cmake` 是否在 `PATH`、
ONNX Runtime SDK 头文件、llama.cpp 运行库、`[convert]` extra 以及
当前 profile。反馈问题前先跑一下。

```bash
lunavox doctor
```

## 2. `bootstrap` —— 一键安装

依次执行 **pull → libs → build → 进程内冒烟合成**。冒烟合成走 Python
原生 `Engine` + `Voice.base()` —— 就是真实调用路径。

```bash
lunavox bootstrap
lunavox bootstrap --model base_small --platform win_cuda12
lunavox bootstrap --skip-test          # 只构建，不跑合成冒烟
```

## 3. `model` —— 目录管理

```bash
lunavox model pull                                    # 拉默认模型
lunavox model pull --model base_small                 # 指定模型
lunavox model convert --model base_small --force      # 需要 [convert] extra
lunavox model convert --all
lunavox model list                                    # 查看已安装
```

`pull` 从社区镜像拉取已转换的 GGUF / ONNX 产物。`convert` 本地把原始
`.safetensors` 权重转成 LunaVox 产物，需要 `[convert]` extra，耗时
几分钟。

## 4. `build` —— 本地引擎

```bash
lunavox build                               # cmake 构建
lunavox build --clean --j 8
lunavox build --toolchain msvc
lunavox build libs                          # 拉默认运行库
lunavox build libs --platform win_cuda12
# 其他: win_cuda13 / win_vulkan / win_cpu / linux_cuda / mac_arm64
```

## 5. `synth` —— 进程内合成

直接调用 Python `Engine` 并写出 WAV —— 与桌面 GUI 完全相同的代码路径。

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
没显式指定的参数依次回落到当前 profile → 环境变量 → 默认值。

## 6. `serve` —— HTTP / WebSocket 服务器

```bash
lunavox serve --host 127.0.0.1 --port 8000 --batch-size 4
```

FastAPI 应用，提供 `POST /v1/synth`、`WS /v1/stream`、
`WS /v1/stream/text`、`GET /health`、`GET /v1/models`、`GET /metrics`。
`BatchEngine` 构成的 N 槽并发请求池。完整接口与协议细节请参阅
**[服务层指南](serve.md)**。

## 7. `gui` —— 桌面应用

```bash
lunavox gui
```

左侧栏 + 三视图布局（合成 / 素材库 / 设置），底层调用与 `lunavox synth`
完全相同的 `Engine` API。

## 8. 模型 ID 对照表

| 模型 ID | 完整名称 | 备注 |
| :--- | :--- | :--- |
| `base_small` | Qwen3-TTS 0.6B Base | 快速、均衡、适合低端设备 |
| `custom_small` | Qwen3-TTS 0.6B Custom | 内置发音人 ID |
| `base` | Qwen3-TTS 1.7B Base | 高保真，建议 GPU |
| `custom` | Qwen3-TTS 1.7B Custom | 大模型发音人定制 |
| `design` | Qwen3-TTS 1.7B Design | 文字描述生成音色 |

## 9. Profile 与配置文件

LunaVox 每次执行都会读取 `~/.lunavox/config.toml` —— 一个
`[default]` 表 + 任意多个 `[profile.<name>]` 覆盖。优先级从高到低：
命令行开关 → 环境变量（`LUNAVOX_MODEL`、`LUNAVOX_BACKEND` 等）→
`[profile.NAME]`（`--profile NAME` 选中）→ `[default]` → 硬编码默认值。

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

## 10. 全局开关

对每一条子命令都生效：

- `--profile <NAME>` —— 选择 `[profile.<NAME>]`
- `--project-root <PATH>` —— 显式指定项目根（开发时）
- `--yes` —— 自动确认所有提示（CI）
- `--no-install` —— 禁止自动修复 Python 依赖
- `--verbose` —— 构建 / 下载原始输出

## 相关文档

- [模型 Profile 与运行时契约](../technical/model_profile.md) · [使用教程](usage_tutorial.md) · [服务层指南](serve.md) · [Runtime API](../api/runtime.md)
