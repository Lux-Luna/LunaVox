# 🌌 LunaVox 运行时技术规范 (Strict Profile Contract)

本文档定义了 LunaVox 推理引擎在加载和运行模型时的严格约束协议。

## 1. 模型配置文件 (`model_profile.json`)
`model_profile.json` 是一个强制性的运行时契约。如果遗漏必要字段，引擎将抛出致命错误。

### 核心必填项：
- **`talker_n_ctx`**: 推理上下文容量上限（默认 `2048`）。
- **`talker_n_ctx_train`**: 原始权重的训练上下文上下文限制。
- **`predictor_n_ctx`**: 预测器上下文容量（默认 `256`）。
- **`codec_num_codebooks`**: 当前版本固定为 `16`。
- **`predictor_vocab_size`**: 预测器词表大小。

## 2. 模式切换逻辑
`--mode` 参数现在是可选的。如果省略，运行时将根据 `model_profile.model_type` 自动路由：
- **`base`**: 自动切换到标准合成模式。若提供 `--reference`，则智能切换至 `clone`。
- **`custom`**: 强制使用 `custom` 模式路由。
- **`design`**: 强制使用 `design` 模式路由。

## 3. 严格错误控制
- **非法组合**: `base` 模型 + `--instruct` 或 **0.6B** 模型 + `--instruct` 将直接抛出硬错误（Hard Error），不再无声忽略。
- **权重工件**: 目前 Talker 运行时仅支持 `qwen3_tts_talker.q5_k.gguf` 作为合法推理工件。

## 4. 默认采样策略 (基于质量驱动)
引擎执行的一组具有确定性的质量优化采样策略：
- **温度 (Temperature)**: `0.6`
- **预测器温度 (Predictor Temp)**: `0.6`
- **最大生成长度**: `max_new_tokens <= 400`
- **随机种子 (Seed)**: `42`
- **预测器种子**: `45`

## 5. 质量校验门禁 (Quality Gate)

执行以下命令快速验证 CLI 可用性：
```bash
./build/lunavox-cli.exe --help
```

- **超时要求**: 单次人工验证推理建议控制在 20 秒以内。
- **验证流程**: 在进行任何量化性能对比前，必须先进行人工主观听感检查。
