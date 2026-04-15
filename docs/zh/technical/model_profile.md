# 模型配置与运行时契约

`model_profile.json` 是各加载器都必须遵守的契约，它在三处需要 **同一次提交内** 同步修改：

| 位置 | 类型 / 文件 | 加载方式 |
| --- | --- | --- |
| **C++ 引擎**（权威） | `lunavox::ModelProfile` 于 `src/model_profile.h` | `src/lunavox_engine.cpp::load_model_profile` |
| **Python 运行时**（仅展示用） | `lunavox.model.ModelProfile` 于 `src/lunavox/model/profile.py` | `ModelProfile.load(model_dir)` |
| **磁盘工件** | `models/<name>/model_profile.json` | 由 `lunavox convert` 写入 |

C++ 侧严格校验（`is_valid` 任一不符即让 `Engine::load_models` 中止）。Python 侧宽松——只用于元数据展示。

## 1. 标识

| 字段 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- |
| `version` | int | 1 | Schema 版本，未知主版本会被拒绝。 |
| `model_type` | string | `"base"` | `base` / `custom` / `design` 之一，决定支持的 `--mode`。 |
| `model_size` | string | `"unknown"` | 展示标签，例如 `"0.6b"`、`"1.7b"`。 |
| `instruct_support` | bool | false | 是否接受 `--instruct`（仅 custom / design）。 |

## 2. 运行时上限

| 字段 | 说明 |
| --- | --- |
| `talker_n_ctx` | Talker LLM 上下文上限，必须 ≤ `talker_n_ctx_train`。 |
| `talker_n_ctx_train` | 权重训练时的上下文长度。 |
| `predictor_n_ctx` | Q1..Q15 预测器上下文上限。 |
| `codec_num_codebooks` | 当前运行时 **必须为 16**。 |
| `codec_id_start` / `codec_id_end` | codec token ID 范围（左闭右开）。 |
| `predictor_vocab_size` | 必须等于 `(codec_id_end - codec_id_start) * (codec_num_codebooks - 1)`。 |

## 3. 特殊 token

| 分组 | 字段 |
| --- | --- |
| codec 帧 | `codec_pad_id`、`codec_bos_id`、`codec_eos_id` |
| think 门控 | `codec_think_id`、`codec_nothink_id`、`codec_think_bos_id`、`codec_think_eos_id` |
| 文本帧 | `tts_pad_id`、`tts_bos_id`、`tts_eos_id` |

## 4. 生成默认值

字段来自上游 `generation_config.json`，每一项都可在 CLI 覆盖。

| 字段 | 默认 | CLI 覆盖 |
| --- | --- | --- |
| `default_max_new_tokens` | 400 | `--max-tokens` |
| `default_temperature` | 0.6 | `--temperature` |
| `default_top_p` | 1.0 | `--top-p` |
| `default_top_k` | 50 | `--top-k` |
| `default_repetition_penalty` | 1.05 | `--repetition-penalty` |
| `default_predictor_do_sample` | true | `--predictor-greedy` |
| `default_predictor_temperature` | 0.6 | `--predictor-temperature` |
| `default_predictor_top_p` | 1.0 | `--predictor-top-p` |
| `default_predictor_top_k` | 50 | `--predictor-top-k` |
| `default_seed` | 42 | `--seed` |
| `default_predictor_seed` | 45 | `--predictor-seed` |

## 5. 语言 / 发音人映射

| 字段 | 结构 |
| --- | --- |
| `language_map` | `{lowercase_name: language_id}` |
| `speaker_map` | `{lowercase_name: speaker_id}` —— 仅 `custom` 模型有 |
| `speaker_dialect_map` | `{lowercase_name: lowercase_dialect_tag}`（可选） |

写入时 key 已被小写化；`ModelProfile::resolve_*` 查询前也会小写化，调用方无需额外处理。

## 6. 模式路由与硬错误

`--mode` 可省略，路由按 `model_type` 决定：

- `base` → `base`（提供 `--reference` 时自动切换为 `clone`）
- `custom` → `custom`
- `design` → `design`

下列情况直接抛硬错误（加载即中止）：

- `base` 模型 + `--instruct` → `--instruct is forbidden in base mode`
- `custom` / `design` + `--reference` → `mode 'clone' is incompatible with model_type ...`
- 任意 0.6B 模型 + `--instruct`（该尺寸没有支持 instruct 的模型）
- Talker 权重不是 `qwen3_tts_talker.q5_k.gguf`

## 7. 质量校验

```bash
./build/lunavox-cli --help            # CLI 烟雾测试
./build/lunavox-cli -m models/base_small -t "hello" -o out.wav
```

最小模型上单次推理应 20 s 内完成。任何量化对比之前都要先做人工听感检查——自动指标不替代耳朵。
