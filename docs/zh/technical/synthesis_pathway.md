# 合成链路：什么时候需要 encoder？

`codec_encoder.fp16.onnx` 与 `speaker_encoder.fp16.onnx` **只有** 在引擎需要从原始 WAV 实时提取 speaker / codec 特征时才会被加载。其他链路都属于"冷推理"——输入是文本、指令、内置发音人 ID 或预计算的 JSON。

## 不需要 encoder 的链路

只动 `talker` (GGUF)、`predictor` (GGUF)、`decoder` (ONNX)。

| 模型类型 | 链路 | 说明 |
| --- | --- | --- |
| Base (0.6B / 1.7B) | Standard TTS | 仅文本，`speaker_embedding` / `ref_codes` 为 null。 |
| Base (0.6B / 1.7B) | Clone from JSON | `--reference foo.json`，特征已缓存。 |
| Custom (0.6B / 1.7B) | Standard TTS | 默认音色。 |
| Custom (0.6B / 1.7B) | Custom voice | `--speaker Vivian`，从 `embeddings/` 加载固化向量。 |
| Design (1.7B) | Standard TTS | 仅文本。 |
| Design (1.7B) | Voice design | 仅 `--instruct`。 |

## 唯一仍需 encoder 的链路

| 模型类型 | 链路 | 原因 |
| --- | --- | --- |
| Base (0.6B / 1.7B) | Clone from WAV | 从 `--reference foo.wav` 实时提取 codes + speaker embedding。 |

`custom` 和 `design` 模型 **完全不支持 clone**（loader 直接拒绝 `--reference`）。

## 清理建议

如果你只用冷推理链路，可以把任一 `models/<name>/` 目录下的 `speaker_encoder.fp16.onnx` 和 `codec_encoder.fp16.onnx` 删掉，每个模型省 ~130–140 MB。运行时容忍它们缺失。
