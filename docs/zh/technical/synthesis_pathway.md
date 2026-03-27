# LunaVox 合成链路编码器需求分析报告

本报告针对 LunaVox 项目中主流合成链路对 `codec_encoder.fp16.onnx` 和 `speaker_encoder.fp16.onnx` 的依赖性进行了详细评估。

## 1. 核心结论

由于链路规则更新，目前 **不再支持** 对 `custom` 和 `design` 类型的模型执行 `clone` (克隆/参考音频) 模式的操作。`custom` 模型仅支持内置专家发音人或基础合成，`design` 模型仅支持通过指令描述进行声音设计。

在目前的合法链路中，大部分 **不需要** 加载或运行 `codec_encoder` 和 `speaker_encoder`。

这些链路统称为 **“冷推理 (Cold Inference)”** 链路，因为它们不需要从原始音频实时提取特征，而是直接输入文本、指令、预定义音色 ID 或 预计算的 JSON 数据。

## 2. 无需编码器的链路列表

以下链路完全脱离了编码器依赖，仅使用 `talker` (GGUF)、`predictor` (GGUF) 和 `decoder` (ONNX)：

### 2.1 Base 类模型
*   **Base (1.7B/0.6B)**: `Standard TTS` (标准合成模式 - 仅从文字开始)
*   **Base (1.7B/0.6B)**: `Clone (JSON)` (使用之前预提取的 JSON 数据进行复刻，无需音频编码)

### 2.2 Custom 类模型
*   **Custom (1.7B/0.6B)**: `Standard TTS` (使用基础默认音色)
*   **Custom (1.7B/0.6B)**: `Custom Voice` (通过 `--speaker` 指定内置 Expert 音色名，如 Vivian)
    *   系统直接从硬盘加载 `embeddings/` 下对应的预计算矩阵，无需现场分析音频。

### 2.3 Design 类模型
*   **Design (1.7B)**: `Standard TTS`
*   **Design (1.7B)**: `Voice Design` (仅使用 `--instruct` 文本描述动态生成音色)

---

## 3. 为何这些链路不需要编码器？

通过分析核心代码逻辑，可以发现：

1.  **Standard TTS / Design**: `speaker_embedding` 和 `ref_codes` 被显式设为 `nullptr` 或使用模型自身的默认值，Talker 直接从提示词开始预测。
2.  **Custom Voice**: 系统直接采用由 Expert 训练并固化的嵌入向量，绕过了实时语音特征提取阶段。
3.  **Clone (JSON)**: 所有的音色特征（`spk_emb` 和 `codes`）已经以 Base64 或数组形式存储 in JSON 文件中。系统直接解析 JSON 数据，跳过了录音分析。

## 4. 需要编码器的链路

目前，**只有 Base 模型在执行 WAV 参考音频克隆时** 才需要编码器：

*   **Base (1.7B/0.6B)**: `Clone (WAV)`
    *   通过参考音频 (WAV) 現場提取声学特征 (Codes) 和 Speaker Embedding。

---

## 5. 文件清理与资源建议 (Tips)

> [!TIP]
> **资源优化与文件精简**:
> 1.  **Custom/Design 模型的精简**: 既然 `custom` 和 `design` 模型已不再支持 `clone` 模式，您 **可以安全地删除** 这些模型目录下的所有 `speaker_encoder` 与 `codec_encoder` 文件。
> 2.  **Base 模型的精简**: 如果您在 Base 模型的使用场景中始终使用 `Clone (JSON)` 或 `Standard TTS`，也可以删除这两个编码器文件。
> 3.  **收益**: 这样做可以为每个模型目录节省约 **130MB - 140MB** 的磁盘空间。
