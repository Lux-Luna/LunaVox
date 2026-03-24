# LunaVox 合成链路编码器需求分析报告

本报告针对 LunaVox 项目中 22 种潜在合成链路对 `codec_encoder.fp16.onnx` 和 `speaker_encoder.fp16.onnx` 的依赖性进行了详细评估。

## 1. 核心结论

在 22 条链路中，一共有 **15 条链路不需要** 加载或运行 `codec_encoder` 和 `speaker_encoder`。

这些链路统称为 **“冷推理 (Cold Inference)”** 链路，因为它们不需要从原始音频提取特征（Embedding 和 Codes），而是直接输入文本、指令、预定义音色 ID 或 预计算的 JSON 数据。

## 2. 无需编码器的链路列表

以下链路完全脱离了编码器依赖，仅使用 `talker` (GGUF)、`predictor` (GGUF) 和 `decoder` (ONNX)：

### 2.1 Base 类模型 (4 条链路)
*   **Base (1.7B)**: `Standard TTS` (标准合成模式)
*   **Base (1.7B)**: `Clone (JSON)` (使用预提取的 JSON 数据进行复刻)
*   **Base (0.6B)**: `Standard TTS` (对应 `base_small`)
*   **Base (0.6B)**: `Clone (JSON)` (对应 `base_small`)

### 2.2 Custom 类模型 (8 条链路)
*   **Custom (1.7B)**: `Standard TTS`
*   **Custom (1.7B)**: `Clone (JSON)`
*   **Custom (1.7B)**: `Custom Voice` (使用内置 Expert 音色名，如 Vivian)
*   **Custom (1.7B)**: `Clone (JSON) + Instruct` (带指令的 JSON 复刻)
*   **Custom (0.6B)**: `Standard TTS` (对应 `custom_small`)
*   **Custom (0.6B)**: `Clone (JSON)` (对应 `custom_small`)
*   **Custom (0.6B)**: `Custom Voice` (对应 `custom_small`)
*   **Custom (0.6B)**: `Clone (JSON) + Instruct` (对应 `custom_small`)

### 2.3 Design 类模型 (3 条链路)
*   **Design (1.7B)**: `Standard TTS`
*   **Design (1.7B)**: `Clone (JSON)`
*   **Design (1.7B)**: `Voice Design` (仅使用 `--instruct` 文本描述)

---

## 3. 为什么这些链路不需要编码器？

通过分析 `src/qwen3_tts.cpp` 代码逻辑，可以发现：

1.  **Standard TTS / Design**: `speaker_embedding` 和 `ref_codes` 被显式设为 `nullptr`，Talker 模型直接从文本或提示词开始预测。
2.  **Custom Voice**: 系统直接从硬盘加载 `embeddings/codec_embedding_0.npy` 等预计算好的矩阵（对应 Vivian 等内置专家音色），通过索引获取 Embedding，无需现场分析音频。
3.  **Clone (JSON)**: 所有的音色特征（`spk_emb` 和 `codes`）已经以 Base64 或数组形式存储在 JSON 文件中。系统直接解析 JSON 数据，跳过了录音分析阶段。

## 4. 剩余 7 条需要编码器的链路

作为对比，以下链路**必须**使用编码器：
*   所有基于 **WAV 参考音频** 的克隆操作：
    *   Base (1.7B/0.6B) 的 `Clone (WAV)` (2条)
    *   Custom (1.7B/0.6B) 的 `Clone (WAV)` 和 `Clone (WAV) + Instruct` (4条)
    *   Design (1.7B) 的 `Clone (WAV)` (1条)

## 5. 性能与资源建议 (Tips)

> [!TIP]
> 如果您在嵌入式或内存敏感的环境下运行，且只使用上述 15 条链路，您可以尝试禁用编码器的加载，从而节省约 **150MB - 300MB** 的内存占用。
