# 合成使用教程

自 v2.2.0 起，LunaVox 提供两条等价的入口：

- **`lunavox synth`** —— Python CLI，直接驱动 `lunavox.runtime.Engine`。
  推荐脚本、CI、日常使用。
- **`./build/lunavox-cli`** —— 独立 C++ 可执行程序，适合深度性能分析
  或者没有 Python 的环境。

两者底层共享同一个引擎。下文示例以 `lunavox synth` 为主，C++ 调用作为
参考并列出。

## 1. Base —— 默认音色

模型内建的默认发音人，不需要参考音频，不需要描述指令。

```bash
lunavox synth "What do you mean that I'm not real?" -o out.wav
```

```bash
# C++ 版等效命令
./build/lunavox-cli -m models/base_small \
  -t "What do you mean that I'm not real?" \
  -o out.wav
```

> [!WARNING]
> Base 模型在 C++ 版中拒绝 `--instruct`，传入会直接报错而不是警告。
> Python CLI 下如果当前 voice 模式不需要该选项，会自动忽略。

## 2. 声音克隆

### 直接用参考音频

```bash
lunavox synth "你好世界。" \
  --voice clone --ref ref/ref.wav \
  -o out.wav
```

### 使用预计算的参考特征 JSON（快速通道）

```bash
lunavox synth "你好世界。" \
  --voice clone --ref ref/ref_0.6B.json \
  -o out.wav
```

JSON 路径完全跳过 speaker / codec encoder，详情见
[synthesis_pathway.md](../technical/synthesis_pathway.md)。

```bash
# C++ 版等效命令
./build/lunavox-cli -m models/base_small `
  -r ref/ref_0.6B.json `
  -t "你好世界。" -o out.wav
```

## 3. 内置发音人（Custom）

从 `custom` 模型目录里选一个发音人，`--instruct` 可选，用来微调情绪
或语气。

```bash
lunavox synth "她说她中午就到。" \
  --voice custom --speaker Vivian --instruct "Use angry tone." \
  -o out.wav
```

- `--speaker` —— 必填，例如 `Vivian`、`Aiden`、`Ryan`。
- `--instruct` —— 可选，调节情绪/语气。

```bash
# C++ 版等效命令
./build/lunavox-cli --mode custom -m models/custom \
  --speaker Vivian --instruct "Use angry tone." \
  -t "她说她中午就到。" \
  -o out.wav
```

## 4. Voice Design —— 文字描述音色

```bash
lunavox synth "你好，很高兴认识你！" \
  --voice design --instruct "A warm female voice, speaking gently with a hint of a smile." \
  -o out.wav
```

- `--instruct` —— 必填，用自然语言描述目标音色。

```bash
# C++ 版等效命令
./build/lunavox-cli --mode design -m models/design \
  --instruct "A warm female voice, speaking gently with a hint of a smile." \
  -t "你好，很高兴认识你！" \
  -o out.wav
```

## Python API

同样四种模式都能通过 Runtime API 调用 —— `lunavox synth` 和 GUI 走的就是
下面这条代码路径。

```python
from lunavox.runtime import Engine, SynthesisParams, Voice

with Engine("models/base_small") as engine:
    params = SynthesisParams(temperature=0.7, top_p=0.9)

    # 按需求选择模式
    voice = Voice.clone_file("ref/ref_0.6B.json")

    result = engine.synthesize("你好世界。", voice=voice, params=params)
    print(f"RTF {result.stats.rtf:.3f}, 时长 {result.stats.audio_duration_ms} ms")
```

完整接口请参阅 [Runtime API 参考](../api/runtime.md)。

## 模式兼容矩阵

| `model_type` | 允许的 `--voice` | `--instruct` | `--ref` |
| :--- | :--- | :---: | :---: |
| Base | `base`（默认）、`clone` | ❌ 禁止 | ✅ 支持（`clone`） |
| Custom | `custom` | ✅ 微调 | ❌ 禁止 |
| Design | `design` | ✅ 必填 | ❌ 禁止 |

C++ 版硬错误：

- Base + `--instruct` → `Error: --instruct is forbidden in base mode`
- Custom / Design + `--reference` → `Error: mode '…' is incompatible with model_type …`

## 性能相关开关

- `--model` (Python CLI) / `-m` (C++ 版) —— 选择要加载的模型目录。
- `--temperature` / `--top-p` / `--top-k` —— 采样器调参（Python CLI）。
- `-j` / `--threads`（默认 4）—— CPU 线程数。
- `lunavox doctor` —— 检查 profile 和运行库状态。
- `--stats-json <path>`（C++ 版） —— 导出结构化时延 / RTF / 内存
  （参见 [stats_schema.md](../technical/stats_schema.md)）。

## Profile

如果需要可复现的运行，把参数固化到 `~/.lunavox/config.toml`，然后用
`--profile` 指定：

```toml
[default]
model = "base_small"

[profile.quality]
backend = "cuda"
temperature = 0.7
top_p = 0.9
```

```bash
lunavox --profile quality synth "请用高保真合成。" -o out.wav
```

完整的优先级规则见 [CLI 参考 § Profile](cli_reference.md)。
