# 🌌 LunaVox CLI 完整指令手册

本文档提供 LunaVox 统一命令行界面 (CLI) 的详细说明。按照执行流程排序，您可以根据需求选择一键全自动引导或手动分步配置。

---

## 🛠️ 1. 环境准备与诊断 (Preparation)

在进行任何模型工作之前，建议运行此工具以确保当前系统环境（CMake, C++, Python 依赖）已就绪。
```powershell
# 安装核心推理工具
pip install lunavox
```

### `doctor` - 系统体检
检查项目结构、工具链路径及运行时库的完整性。

**命令示例:**
```bash
lunavox doctor
```

**检查项包括:**
- 项目根目录及其关键子目录（`src`, `lib`, `models`）是否存在。
- `cmake` 是否在系统 PATH 中。
- ONNX Runtime SDK 头文件与 Llama.cpp 运行时库是否缺失。
- `[convert]` 选装包是否已通过 pip 安装。

---

## 🚀 2. 核心全自动引导 (The One-Key Solution)

对于首次安装 LunaVox 或想要快速开始推理的用户，建议使用此命令。

### `bootstrap` - 一键引导设置
这是一个高度自动化的复合指令，它会顺序执行以下任务：
1. **Pull Model**: 从 HuggingFace 下载选定的模型。
2. **Download Libs**: 根据您的系统自动检测并下载 ONNX/Llama 运行库。
3. **Build**: 自动配置并编译 C++ 进入推理引擎。
4. **Interactive Test**: 完成后开启交互式测试，让您立刻听到声音。

**用法示例:**
```bash
# 进入全自动互动引导
lunavox bootstrap

# 或指定特定参数
lunavox bootstrap --model base_small --platform win_cuda12
```

---

## 📦 3. 模型获取方式 (Model Management)

您可以选择直接下载转换好的模型（推荐），或者从原始权重进行本地转换。

### `pull-model` - 拉取预转换模型 (推荐)
直接从官方仓库同步经过 LunaVox 深度优化的运行格式（GGUF/ONNX）。

**用法示例:**
```bash
# 开启交互式选择下载
lunavox pull-model

# 下载指定模型
lunavox pull-model --model base_small
```

### `convert` - 模型本地转换
如果您已经拥有 Qwen3-TTS 的原始权重（`.safetensors`），或者需要自定义转换参数，请使用此命令。

**用法示例:**
```bash
# 本地转换
lunavox convert --model base_small --force
```
*注：本地转换可能需要数分钟时间，并且需要额外部署 Python 转换环境。*

---

## ⚙️ 4. 手动构建流程 (Manual Setup)

如果您不想使用 `bootstrap`，可以通过以下步骤手动完成环境搭建。

### `download-libs` - 下载运行库
下载特定平台的二进制核心（ONNX Runtime / Llama.cpp）。

**用法示例:**
```bash
# 智能下载（推荐进入选择界面）
lunavox download-libs

# 手动指定平台下载
lunavox download-libs --platform win_cuda12
```

### `build` - 编译 C++ 推理引擎
基于 CMake 进行本地编译，生成最终的 `lunavox-cli` 执行程序。

**用法示例:**
```bash
# 极简构建
lunavox build

# 高效清理并加速构建
lunavox build --clean --j 8
```

---

## 📝 附录：模型 ID 参考表

| 模型 ID | 模型全称 | 推理能力 |
| :--- | :--- | :--- |
| `base_small` | Qwen3-TTS 0.6B Base | 极速均衡，适合低资源设备 |
| `custom_small` | Qwen3-TTS 0.6B Custom | 支持固定发音人 ID 切换 |
| `base` | Qwen3-TTS 1.7B Base | 高保真度，建议使用 GPU 加速 |
| `custom` | Qwen3-TTS 1.7B Custom | 大尺寸发音人定制模型 |
| `design` | Qwen3-TTS 1.7B Design | 支持纯文字描述设计声音 (Prompt-to-Voice) |

---

## 🌍 全局参数

这些参数适用于上述**所有** `lunavox` 命令：

- `--project-root <PATH>`: 手动指定根目录（常用于开发环境）。
- `--yes`: 自动确认所有风险操作和下载提示（非交互式/CI 执行必备）。
- `--no-install`: 强制不检测/自动修复 Python 模块缺失。
- `--verbose`: 显示构建和网络下载的详细原始输出。

---

## 📜 更多信息

有关运行时详细配置及设计准则，请参阅:
- **[运行时设计契约与约束 (Runtime Specs)](../technical/runtime_specs.md)**
