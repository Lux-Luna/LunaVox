# LunaVox CLI 指令使用指南

本教程详细说明 LunaVox C++ 推理引擎（`lunavox-cli.exe`）的实际指令格式、核心参数用法及参数约束说明。

> [!IMPORTANT]
> LunaVox CLI 采用主命令加可选参数的结构，不同的合成功能通过 `--mode` 参数进行切换。如果未指定 `--mode`，系统将根据 `model_profile.json` 中的 `model_type` 自动选择。

---

## 1. 基础模式 (Base Mode)
最简单的合成方式，仅适用于 `base` 类型的模型。

```bash
./build/lunavox-cli `
  -m models/base_small `
  -t "What do you mean that I'm not real?" `
  -o out.wav
```

> [!WARNING]
> **Base 模型不支持 `--instruct`**。如果您在 base 模式下提供指令文本，引擎将报错并终止运行。

---

## 2. 声音克隆 (Voice Cloning / Reference)
通过提供参考音频或特征来模仿特定音色。**此功能仅对 `base` 类型模型有效**。

### 2.1 使用参考音频 (Reference Audio)
```bash
./build/lunavox-cli `
  -m models/base_small `
  -r ref/ref.wav `
  -t "Hello world." `
  -o out.wav
```

### 2.2 使用参考特征 (Reference JSON)
```bash
./build/lunavox-cli `
  --mode clone `
  -m models/base_small `
  -r ref/ref_0.6B.json `
  -t "Hello world." `
  -o out.wav
```

---

## 3. 定制化声音 (Custom Voice)
使用系统内置的特定专家发音人 ID。**此功能仅对 `custom` 类型模型有效**。

```bash
./build/lunavox-cli `
  --mode custom `
  -m models/custom `
  --speaker Vivian `
  --instruct "Use angry tone." `
  -t "She said she would be here by noon." `
  -o out.wav
```

- **--speaker**: 必需参数。指定发音人（如 `Vivian`, `Aiden`, `Ryan` 等）。
- **--instruct**: 可选参数。用于调节预设人物的情感或语气。

---

## 4. 声音设计 (Voice Design)
根据文字描述动态设计全新音色。**此功能仅对 `design` 类型模型有效**。

```bash
./build/lunavox-cli `
  --mode design `
  -m models/design `
  --instruct "A warm female voice, speaking gently with a hint of a smile." `
  -t "Hello, it's nice to meet you!" `
  -o out.wav
```

- **--instruct**: 必需参数。提供详尽的音色物理特征描述。

---

## 📜 核心约束与兼容性矩阵

为了确保最佳合成质量及运行时稳定性，请遵循以下模式与模型类型的对应关系：

| 模型类型 (`model_type`) | 支持的模式 (`--mode`) | 支持 `--instruct` ? | 支持 `--reference` ? |
| :--- | :--- | :---: | :---: |
| **Base** | `base` (默认), `clone` | ❌ 禁用 | ✅ 支持 |
| **Custom** | `custom` | ✅ 支持 (调优) | ❌ 禁用 |
| **Design** | `design` | ✅ 必需 (定义) | ❌ 禁用 |

### 常见错误说明
- **Base + Instruct**: 运行时将抛出 `Error: --instruct is forbidden in base mode`。
- **Custom/Design + Reference**: 运行时将抛出 `Error: mode 'clone' is incompatible with model_type ...`。

---

## 5. 环境依赖与性能统计

- **线程控制**: 使用 `-j` 或 `--threads`（默认 4）来控制 CPU 资源占用。
- **性能统计**: 运行命令时添加 `--stats-json <path>` 可获取详细的耗时、RTF 以及内存占用报告。
- **便携式构建**: Windows 端使用 `python manage.py build --portable` 可打包所有运行库 DLL。
