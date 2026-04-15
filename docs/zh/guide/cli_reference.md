# LunaVox CLI 指令手册

`lunavox` CLI 是环境配置、模型管理和 C++ 构建的统一入口。下列命令按执行顺序排列——首次安装可直接走 `bootstrap`，需要细粒度控制就分步执行。

```powershell
pip install lunavox
```

## 1. `doctor` —— 系统体检

检查项目结构、工具链和运行时库。提交 issue 之前先跑一遍。

```bash
lunavox doctor
```

检查项：项目根目录与 `src` / `lib` / `models`、`PATH` 中的 `cmake`、ONNX Runtime SDK 头文件、llama.cpp 运行库、是否装了 `[convert]` 选装包。

## 2. `bootstrap` —— 一键设置

按顺序执行 **pull-model → download-libs → build → 交互式测试**。

```bash
lunavox bootstrap
lunavox bootstrap --model base_small --platform win_cuda12
```

## 3. 模型管理

### `pull-model`（推荐）

从官方镜像拉取已转换的 GGUF / ONNX 工件。

```bash
lunavox pull-model
lunavox pull-model --model base_small
```

### `convert`

从原始 `.safetensors` 权重本地转换。需要装 `[convert]` 选装包，耗时数分钟。

```bash
lunavox convert --model base_small --force
```

## 4. 手动构建

### `download-libs`

下载平台对应的 ONNX Runtime + llama.cpp 二进制。

```bash
lunavox download-libs
lunavox download-libs --platform win_cuda12   # win_cuda13 / win_vulkan / win_cpu / linux_cuda / mac_arm64
```

### `build`

基于 CMake 构建 `lunavox-cli`（以及 C ABI 共享库）。

```bash
lunavox build
lunavox build --clean --j 8
```

## 5. 模型 ID 参考

| 模型 ID | 全称 | 备注 |
| :--- | :--- | :--- |
| `base_small` | Qwen3-TTS 0.6B Base | 极速均衡，低资源设备友好 |
| `custom_small` | Qwen3-TTS 0.6B Custom | 内置发音人 ID |
| `base` | Qwen3-TTS 1.7B Base | 高保真，建议 GPU |
| `custom` | Qwen3-TTS 1.7B Custom | 大尺寸发音人定制 |
| `design` | Qwen3-TTS 1.7B Design | 文字描述设计音色 |

## 6. 全局参数

所有 `lunavox` 子命令通用：

- `--project-root <PATH>` —— 显式指定项目根（开发用）。
- `--yes` —— 自动确认全部提示（CI 必备）。
- `--no-install` —— 禁用自动修复 Python 模块。
- `--verbose` —— 显示构建与下载的原始输出。

## 相关文档

- [模型配置与运行时契约](../technical/model_profile.md)
- [使用教程 (`lunavox-cli` 各模式)](usage_tutorial.md)
