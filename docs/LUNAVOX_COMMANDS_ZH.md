# LunaVox CLI 指令汇总手册

本文档汇总了 LunaVox 统一命令行界面 (CLI) 的所有可用指令及其参数说明。

## 全局参数 (Global Options)

这些参数适用于所有 `lunavox` 命令，且应置于子命令之前。

- `--project-root <PATH>`: 手动指定 LunaVox 项目根目录。
- `--yes`: 自动确认所有安装或下载提示（非交互模式必需）。
- `--no-install`: 禁用缺失 Python 依赖项的自动安装功能。
- `--verbose`: 启用详细日志输出，方便排查问题。

---

## 指令详情

### 1. `convert` - 模型本地转换
从原始模型权重（SafeTensors）转换为 LunaVox 运行所需的优化格式（ONNX/GGUF）。

**用法:**
```powershell
lunavox convert [OPTIONS]
```

**参数:**
- `--model <ID>`: 指定要转换的模型变体 ID（如 `base_small`, `base`, `design` 等）。
- `--models-dir <PATH>`: 覆盖默认的模型输出目录。
- `--force`: 强制重新转换，即使目标文件已存在。
- `--all`: 转换所有预定义的模型变体。
- 若不带参数运行，将进入**交互式模型选择**界面。

---

### 2. `pull-model` - 拉取预转换模型
直接从 HuggingFace 仓库下载已经转换好的分发包（GGUF/ONNX），无需本地进行耗时的转换过程。

**用法:**
```powershell
lunavox pull-model [OPTIONS]
```

**参数:**
- `--model <ID>`: 指定要下载的模型变体。
- 若不带参数运行，将进入**交互式拉取选择**界面。

---

### 3. `build` - 构建推理引擎
编译 LunaVox C++ 推理引擎。

**用法:**
```powershell
lunavox build [OPTIONS]
```

**参数:**
- `--clean`: 在构建前先清理构建目录。
- `--j <NUM>`: 并行构建任务数量（默认为 4）。
- `--toolchain <STR>`: 强制指定使用的编译器工具链（默认为 `auto`）。

---

### 4. `bootstrap` - 一键引导设置
引导式交互流程，依次执行：拉取模型 -> 下载二进制依赖 -> 构建引擎 -> 推理测试。适合首次安装。

**用法:**
```powershell
lunavox bootstrap [OPTIONS]
```

**参数:**
- `--model <ID>`: 指定使用的模型变体。
- `--platform <KEY>`: 指定目标平台（如 `win_cuda`）。
- `--force`: 强制重新转换模型。
- `--clean`: 清理构建目录。
- `--j <NUM>`: 并行构建任务数。
- `--toolchain <STR>`: 指定工具链。

---

### 5. `download-libs` - 下载库依赖
下载对应平台的二进制依赖项，如 ONNX Runtime 或 Llama.cpp。

**用法:**
```powershell
lunavox download-libs [OPTIONS]
```

**参数:**
- `--platform`, `-p <KEY>`: 指定目标平台（如 `win_cuda`, `win_vulkan`, `linux_cpu` 等）。
- `--lib <NAME>`: 仅下载特定库（`onnx` 或 `llama`）。
- `--backend <NAME>`: 为特定库指定后端驱动。
- 若不指定平台，将进入**交互式平台选择**界面。

---

### 6. `doctor` - 环境诊断
检查当前开发环境和依赖项状态，并尝试修复常见问题。

**用法:**
```powershell
lunavox doctor
```

**功能:**
- 检查项目目录结构。
- 探测 `cmake` 工具。
- 验证运行时库（Llama/ONNX）是否缺失。
- 检查 Python 转换依赖包是否安装完整。

---

## 模型变体 ID 参考表

在执行 `convert`, `pull-model` 或 `bootstrap` 时，可使用以下 `--model` ID：

| 模型 ID | 说明 |
| :--- | :--- |
| `base_small` | Qwen3-TTS 0.6B 基础版 |
| `custom_small` | Qwen3-TTS 0.6B 定制音色版 |
| `base` | Qwen3-TTS 1.7B 基础版 |
| `custom` | Qwen3-TTS 1.7B 定制音色版 |
| `design` | Qwen3-TTS 1.7B 音色设计版 |
