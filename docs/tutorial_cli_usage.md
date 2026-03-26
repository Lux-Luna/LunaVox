# LunaVox CLI 指令使用指南

本教程详细说明 LunaVox C++ 推理引擎（`qwen3-tts-cli.exe`）的实际指令格式、核心参数用法及合成模式说明。

> [!IMPORTANT]
> LunaVox CLI 采用主命令加可选参数的结构，不同的合成功能通过 `--mode` 参数进行切换，而非子命令形式。

---

## 1. 基础模式 (Base Mode) - 默认
最简单的合成方式，使用模型自带的默认参数。若不指定 `--mode`，系统默认为 `base`。

```bash
./build/qwen3-tts-cli \
  -m models/base_small \
  -t "What do you mean that I'm not real?" \
  -o out.wav
```

- **核心参数**:
  - `-m, --model`: 模型目录路径。
  - `-t, --text`: 目标合成文本。
  - `-o, --output`: 输出 WAV 文件路径。

---

## 2. 声音克隆 (Voice Cloning)
通过提供参考特征来模仿特定音色。需指定 `--mode clone`。

### 2.1 使用参考音频 (Reference Audio)
提供原始 WAV 文件，系统会实时提取其声学特征和 Speaker Embedding。
```bash
./build/qwen3-tts-cli `
  -m models/base_small `
  -r ref/ref.wav `
  -t "Hello world." `
  -o out.wav
```

### 2.2 使用参考特征 (Reference JSON)
直接读取预计算的特征 JSON（包含 `spk_emb` 和 `codes`），可节省推理开销并确保特征一致性。
```bash
./build/qwen3-tts-cli \
  --mode clone \
  -m models/base_small \
  -r ref/ref_0.6B.json \
  -t "Hello world." \
  -o out.wav
```

- **关键参数**:
  - `-r, --reference`: 参考音频文件 (.wav) 或特征文件 (.json)。

---

## 3. 定制化声音 (CustomVoice / Speaker ID)
使用系统内置的特定发音人 ID。需指定 `--mode custom`。

```bash
./build/qwen3-tts-cli \
  --mode custom \
  -m models/custom_small \
  --speaker Vivian \
  -t "Welcome to the show." \
  -o out.wav
```

- **--speaker**: 必需参数。指定发音人（如 `Vivian`, `Aiden`, `Ryan`, `Serena` 等）。
- **--instruct**: 可选参数。用于附加情绪或语气指令。

---

## 4. 声音设计 (VoiceDesign)
根据文字描述动态设计全新音色。需指定 `--mode design`。

```bash
./build/qwen3-tts-cli \
  --mode design \
  -m models/base_small \
  --instruct "A warm female voice" \
  -t "Hello!" \
  -o out.wav
```

- **--instruct**: 必需参数。提供详尽的音色描述。

---

## 重点补充说明

### 1. `custom` 与 `design` 的区别
| 特性 | Custom (Speaker ID) | Design (Voice Design) |
| :--- | :--- | :--- |
| **Speaker ID** | **必需** (`--speaker`) | **不需要** |
| **Instruct 作用** | 调节预设人物的**情感/风格** | **从零定义**全新的声音物理特征 |
| **典型指令** | `"Speak gently"`, `"Excited"` | `"Deep male voice with accent"`, `"Young girl"` |

### 2. 非必要参数与自动检测
- **--ref-text (参考文本)**: 
  在 `clone` 模式下为可选。若提供，系统会尝试对齐参考音频；若不提供，系统将使用无对齐克隆逻辑（x-vector 模式）。
- **--language / -l**: 
  LunaVox 默认开启**自动语言检测**。
  - 默认情况下程序会自动识别中、英、日、韩等主流语言。
  - 使用 `-l none` 可以显式跳过特定语言预设或强制使用未指定的行为。
- **--ref-audio**:
  注意在本项目 CLI 中，参数名为 `-r` 或 `--reference`，图片中提到的 `--ref-audio` 实际上已合并至该选项中。

### 3. 系统优化提示
- **线程控制**: 使用 `-j` 或 `--threads`（默认 4）来控制推理时的 CPU 资源占用。
- **性能统计**: 运行命令时添加 `--stats-json <path>` 可获取详细的耗时、RTF 以及内存占用报告。

---

## 5. 环境依赖与便携构建 (Portable Build)

如果您在具有 NVIDIA GPU 的 Windows 环境中运行，但发现系统报错 `Failed to load shared library (CUDA)`，通常是因为程序无法定位到系统或 Conda 环境中的 CUDA/cuDNN 运行库。

### 5.1 便携式构建 (Recommended for Distributing)
通过在构建时添加 `--portable` 参数，LunaVox 构建管理器会自动从当前的 Conda 环境或系统 `PATH` 中提取必要的 CUDA 和 cuDNN 核心 DLL 文件（如 `cudart64_*.dll`, `cudnn*.dll` 等），并将其直接包裹 (Bundle) 到 `build/` 目录下。

```bash
# 执行便携式构建
python manage.py build --clean --portable
```

**优点：**
- **开箱即用**: 构建后的 `build/` 文件夹可以拷贝到任何安装了对应驱动的机器上直接运行，无需安装 Conda 或 CUDA Toolkit 环境。
- **独立性**: 解决了因环境变量冲突导致的 CUDA 无法激活问题。

**说明：**
- 打包 CUDA 依赖会显著增加 `build/` 目录的体积（通常增加 500MB~1GB）。
- 非 Windows 平台（如 macOS）由于 CoreML 是系统内置框架，无需也暂不支持 `--portable` 选项。
