# `lunavox-cli` 使用教程

`lunavox-cli` 采用 **主命令 + 选项** 结构。`--mode` 决定合成路径；省略时引擎按 `model_profile.model_type` 自动路由（参见 [model_profile.md](../technical/model_profile.md)）。

## 1. Base 模式

普通文本合成，仅适用于 `base` 模型。

```bash
./build/lunavox-cli `
  -m models/base_small `
  -t "What do you mean that I'm not real?" `
  -o out.wav
```

> [!WARNING]
> Base 模型不接受 `--instruct`，传入会直接抛硬错误。

## 2. 声音克隆（仅 Base 模型）

### 使用参考音频

```bash
./build/lunavox-cli `
  -m models/base_small `
  -r ref/ref.wav `
  -t "Hello world." `
  -o out.wav
```

### 使用预计算 JSON

```bash
./build/lunavox-cli `
  --mode clone `
  -m models/base_small `
  -r ref/ref_0.6B.json `
  -t "Hello world." `
  -o out.wav
```

JSON 路径完全跳过 speaker / codec encoder，详见 [synthesis_pathway.md](../technical/synthesis_pathway.md)。

## 3. 定制化声音（仅 Custom 模型）

```bash
./build/lunavox-cli `
  --mode custom `
  -m models/custom `
  --speaker Vivian `
  --instruct "Use angry tone." `
  -t "She said she would be here by noon." `
  -o out.wav
```

- `--speaker` —— 必需，如 `Vivian`、`Aiden`、`Ryan`。
- `--instruct` —— 可选，用于调节情感 / 语气。

## 4. 声音设计（仅 Design 模型）

```bash
./build/lunavox-cli `
  --mode design `
  -m models/design `
  --instruct "A warm female voice, speaking gently with a hint of a smile." `
  -t "Hello, it's nice to meet you!" `
  -o out.wav
```

- `--instruct` —— 必需，描述目标音色。

## 兼容性矩阵

| `model_type` | 允许 `--mode` | `--instruct` | `--reference` |
| :--- | :--- | :---: | :---: |
| Base | `base`（默认）、`clone` | ❌ 禁用 | ✅ 支持 |
| Custom | `custom` | ✅ 调优 | ❌ 禁用 |
| Design | `design` | ✅ 必需 | ❌ 禁用 |

硬错误：

- Base + `--instruct` → `Error: --instruct is forbidden in base mode`
- Custom / Design + `--reference` → `Error: mode 'clone' is incompatible with model_type ...`

## 性能旋钮

- `-j` / `--threads`（默认 4）—— CPU 线程数。
- `--stats-json <path>` —— 输出结构化耗时 / RTF / 内存（参见 [stats_schema.md](../technical/stats_schema.md)）。
- Windows 绿色打包：先 `lunavox download-libs --platform win_*`，再 `lunavox build --clean`，DLL 会被一并拷贝到 `build/`。
