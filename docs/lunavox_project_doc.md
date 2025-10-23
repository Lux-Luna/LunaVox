# LunaVox 项目文档

## 简介
LunaVox 是 **GPT-SoVITS Fast Inference Engine** 的推理特化引擎，专为 **GPT-SoVITS** 项目优化。基于本仓库的转换与测试结果，模型体积约为 **v2 ≈ 237MB**，**v2ProPlus ≈ 642MB**。

主要优势：
- **空间优化**：无需数 GB 的整合环境，部署简便。
- **延迟较低**：针对 CPU 路径进行优化；首包延迟与整体推理速度见下文测试数据。
- **代码优化**：重写推理逻辑，优化内存与计算开销。

---

## 推理流程（以日语为例）

GPT-SoVITS 的推理分为四步：

1. **文本转音素 (Openjtalk)**
   - 输入文本转化为标准化音素序列（Phoneme），提供清晰的发音指导。
   - 示例：文本 `こんにちは` → `[k0nichiwasekai]`

2. **参考音频特征提取 (HuBERT)**
   - 使用 **中文 HuBERT** 提取参考音频的 SSL 特征。
   - 保留音色、韵律等风格信息，辅助后续生成。
   - 可实现 Zero-Shot TTS。

2.5 **语义提取 (RoBERTa，可选)**
   - 针对深层语义理解，日语可跳过该步骤。

3. **T2S：文本转语音 Token (GPT)**
   - 使用自回归模型 (GPT) 生成语音 Token。
   - 模型结构为 Transformer Decoder，包含 24+ 层多头注意力。
   - 关键实现：Mask T5 Self-Attention。

4. **S2A：语音合成 (VITS)**
   - 将语义 Token 转换为音频波形。
   - 使用 VITS 声码器输出最终语音。
   - 延迟取决于硬件与文本长度（见下文测试数据）。

---

## 详细性能（本仓库测试）

| 版本 | 首包延迟(平均) | 运行时大小 | 模型大小 |
|------|----------------|------------|----------|
| LunaVox v2 | 约 1.76s | 约 1.34GB | 约 236.7MB |
| LunaVox v2ProPlus | 约 2.08s | 同上 | 约 641.9MB |

> 数据来源：`performance_tests/results/test_results.json`（Windows，默认 ORT CPUExecutionProvider，输入为短句日语）。
> 注：本仓库未在同一环境下获得可复现实测的 GPT-SoVITS PyTorch/ONNX 数值，故不做直接对比结论。

---

## LunaVox 优化点

### 1. KV Cache 管理
- StageDecoder 将 KV Cache 以 **past/present 张量**形式作为 ONNX 输入/输出，Python 端逐步推进自回归解码。
- 未采用图内循环/Scan，仍存在每步 `Session.run` 的开销（但避免了在 PyTorch 中计算注意力的开销）。
- 默认使用 ORT CPUExecutionProvider；GPU Provider 尚未在主推理链路中启用。

### 2. I/O Binding（当前未启用）
- 目前未在主推理流程中启用 onnxruntime 的 I/O Binding；后续可作为优化项以减少拷贝开销。

### 3. CPU 特化优化
- 针对 CPU 路径优化：推理使用 FP32；
- 打包中保留 FP16 权重二进制，加载时转换为 FP32 以提升兼容性（见 `t2s_shared_fp16.bin` → `t2s_shared_fp32.bin`）。

### 4. 多重缓存机制
- **复数参考音频缓存**：支持多条缓存，避免重复计算。
- **多模型缓存**：支持多说话人切换，无需频繁重新加载模型。
- **CPU 专用优化**：对 CPU 平台进行重写优化。

---

## 代码实现关键点

### 自回归解码 (StageDecoder)
```python
input = start_token
for step in range(max_length):
    output = model(input)  # ONNX 图计算
    input = output         # GPU 端循环
```
- 通过 past/present 张量在 ONNX 与 Python 间传递 KV Cache；循环由 Python 驱动。
- 优化了解码阶段的数据流，减少不必要的数据重排与拷贝。

### LunaVox 的改进
- 以 past/present 张量驱动缓存复用，匹配 v2 / v2Pro / v2ProPlus 的 ONNX 模板与权重布局。
- 默认 CPUExecutionProvider，便于在无 GPU 环境快速部署。

---

## 结论

基于当前实现与本仓库测试：
- 模型体积（v2）约 **237MB**，v2ProPlus 约 **642MB**；运行时依赖体积约 **1.34GB**。
- CPU 路径具备较好的首包延迟（短句约 1–2 秒）。
- 多缓存机制适合多说话人/多参考音频场景；中文路径支持 BERT 对齐；v2Pro/v2ProPlus 支持说话人向量。

> **一句话总结**：LunaVox 是 GPT-SoVITS 的高效推理引擎，轻量、低延迟、易部署，适合在 CPU 环境快速落地。

