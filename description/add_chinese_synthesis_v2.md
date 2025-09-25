### 在 LunaVox 中添加中文语音合成功能（仅 v2 模型）

---

## 目标与范围
- 目标：在现有仅支持日语的 LunaVox 基础上，新增中文（普通话）合成支持。
- 限制：仅面向 GPT-SoVITS v2 模型（保持与当前转换与推理框架一致）。

---

## GPT-SoVITS v2 中文前端概览（参考）
- 语言切分：`LangSegmenter` 将混合文本拆分为 `zh`、`en`、`ja` 等段
- 中文清洗与 G2P：`text.chinese2`（默认启用 G2PW，结合多音字修正、变调、儿化）
- 符号映射：`symbols2`（拼音声母/韵母+调号，与 `opencpop-strict.txt` 一致）
- BERT 特征：中文段按 `word2ph` 将字符级 RoBERTa 特征重复到音素级；非中文为零矩阵

---

## LunaVox 需要的新增依赖
- 模型/权重（推理时用）：
  - 中文 RoBERTa（字符级上下文）：`GPT-SoVITS/pretrained_models/chinese-roberta-wwm-ext-large`（可替换为外部同构路径）
  - CN-HuBERT（SSL 特征）：`GPT-SoVITS/pretrained_models/chinese-hubert-base`（已有）
- Python 依赖：
  - `jieba_fast` 或 `jieba`、`transformers`、`torch`（已用于 ONNX 前处理的 CPU/BERT 推理）
  - G2PW ONNX 推理：内置于 `GPT-SoVITS/text/g2pw`（若复用代码逻辑需可选）

提示：LunaVox 目前采用 ONNX 完成 T2S 与 VITS 解码，但 BERT 抽取仍可沿用 GPT-SoVITS 的 `transformers` 推理（CPU 可用）。

---

## LunaVox 中需新增/修改的模块
- 符号集与映射：
  - 方案 A（推荐）：直接复用 `GPT-SoVITS/text/symbols2.py` 的符号顺序与集合，保证与 v2 保持一致；在 LunaVox 新增 `Chinese/SymbolsV2.py`（或通用 `SymbolsV2.py`）导出 `symbols_v2/symbol_to_id_v2`。
  - 方案 B：从 GPT-SoVITS 复制必要符号列表（声母、韵母1-5调、标点、ARPAbet、日/韩/粤），确保顺序与 ID 一致。

- 中文 G2P 与文本清洗：
  - 新增 `src/lunavox_tts/Chinese/ChineseG2P.py`：
    - 规范化：使用与 `text.zh_normalization.TextNormalizer` 类似的标点替换、切句、数字汉字化（可简化为必要子集）。
    - G2P：优先方案为直接集成 `g2pw` ONNX（与 GPT-SoVITS 对齐）；将拼音映射为 `opencpop-strict.txt` 的符号，再转 `symbols2` id。
    - 儿化/变调：按 `text.tone_sandhi.ToneSandhi` 与 `_merge_erhua` 规则实现或复用对应逻辑。
    - 返回：phones（符号序列）与 `word2ph`（逐字对应的音素数量）。

- 中文 BERT 特征：
  - 在 `src/lunavox_tts/Core/Inference.py` 类似 `GPT-SoVITS/TTS_infer_pack/TextPreprocessor.py` 的做法：
    - 新增中文分支：
      - 将输入文本按 `ChineseG2P` 规范化与分词，得到 `norm_text` 与 `word2ph`
      - 用中文 RoBERTa tokenizer/model 取隐层（建议倒数第 2~3 层拼接，保持 1024 维），移除 CLS/SEP
      - 按 `word2ph` 重复到音素级，形成 `(phone_len, 1024)` 的特征矩阵（注意：LunaVox 当前接口为 `(seq_len, 1024)`，需转置为与现有日本语路径一致的形状）
    - 对于非中文段（如标点或后续支持的其他语言），仍返回零矩阵以兼容接口。

- 语言入口与路由：
  - 在 `src/lunavox_tts/Core/Inference.py` 增加中文入口：
    - 提供 `chinese_to_phones(text)`，并在 `tts()` 根据语言选择调用（可先提供独立 API，再合并 LangSegmenter 支持）。
    - 目前 LunaVox 的 `tts()` 假定输入为日语，且 BERT 全零。为支持中文：
      - 在进入 T2S 前生成有效 `text_bert`（中文）；
      - 仍使用现有 ONNX Encoder/Decoder 接口（无需改 onnx）。

---

## 文件与代码建议结构
- `src/lunavox_tts/Chinese/ChineseG2P.py`：中文文本规范化、G2P、`word2ph`、phones→IDs
- `src/lunavox_tts/Chinese/SymbolsV2.py`：导入/复刻 `symbols2`，提供 `symbols_v2/symbol_to_id_v2`
- `src/lunavox_tts/Core/Inference.py`：
  - 新增 `chinese_to_phones()` 与生成中文 `text_bert` 的逻辑
  - `tts()` 支持按语言分支（短期方案可新建 `tts_zh()`）
- 预置模型：
  - `Data/chinese-roberta-wwm-ext-large`（含 tokenizer 与模型）

---

## 兼容性与注意事项
- ID 对齐：必须与 v2 的 `symbols2` 完全一致，否则 T2S 语义预测将错位。
- BERT 维度：保持 1024 维并按 phones 对齐；长度小于 6 的 phones 建议在前加句点以稳定推理（参考 v2 实现）。
- 性能：BERT 与 G2PW 可在 CPU 上运行；长文本需切句避免 token 过长报错。

---

## 最小改动清单（实施指南）
1) 复制或实现 `symbols2` 并生成 `symbols_v2/symbol_to_id_v2`
2) 新增 `ChineseG2P.py`：
   - 规范化 → G2PW → 变调/儿化 → 拼音映射 → phones + `word2ph`
3) 引入中文 RoBERTa，新增中文 `text_bert` 生成：
   - tokenizer + model 前向 → 逐字特征 → 按 `word2ph` 重复至音素级
4) 在 `Inference.py` 里增加中文合成流程（或新建 `tts_zh`）
5) 准备依赖与模型：
   - 放置 `chinese-roberta-wwm-ext-large` 到 `Data/`，或在配置中提供路径
   - 可选：内置 G2PW ONNX（如果不复用 GPT-SoVITS 的 Python 代码）

---

## 示例：中文调用（建议 API）
```python
from lunavox_tts.Core.Inference import tts_client
from lunavox_tts.Audio.ReferenceAudio import ReferenceAudio

# 假设已加载 onnx 模型，并准备好参考音频
sr, audio = tts_client.tts_zh(
    text="今天天气很好，我们去公园散步。",
    prompt_audio=ref,
    encoder=encoder_sess,
    first_stage_decoder=first_stage_sess,
    stage_decoder=stage_decoder_sess,
    vocoder=vocoder_sess,
)
```

