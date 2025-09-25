### 在 LunaVox 中添加英文语音合成功能（仅 v2 模型）

---

## 目标与范围
- 目标：在现有仅支持日语的 LunaVox 基础上，新增英文合成支持。
- 限制：仅面向 GPT-SoVITS v2 模型。

---

## GPT-SoVITS v2 英文前端概览（参考）
- 文本规范化：`text.english.text_normalize`，统一中英文标点、展开缩写
- 英文 G2P：`text.english.en_G2p`
  - 词典优先（`cmudict.rep`、`cmudict-fast.rep`、`engdict-hot.rep`）
  - 多音字按 POS 消歧、姓名词典覆盖（仅首字母大写）
  - OOV：`wordsegment` 分词重组 → 回退 `g2p_en` 预测
  - 输出 ARPAbet 符号，按 `symbols2` 过滤合法 phones
- BERT：英文段返回零矩阵（仅中文段计算 RoBERTa）

---

## LunaVox 需要的新增或复用内容
- 符号集：
  - 复用 `symbols2` 中的 ARPAbet 符号；与现有日语符号同表（保持与 v2 一致）

- 英文 G2P：
  - 方案 A（轻量）：使用 `g2p_en` + `wordsegment`，并附带一个精简版 CMU 字典热更新表（可选）
  - 方案 B（对齐 GPT-SoVITS）：复制 `cmudict*.rep` 与热更新逻辑，基本一致复刻 `text.english` 中的策略

- BERT：
  - 保持英文 `text_bert` 为零矩阵（形状与现有接口一致，`(seq_len, 1024)`）。

---

## LunaVox 中需新增/修改的模块
- `src/lunavox_tts/English/EnglishG2P.py`（新建）：
  - `text_normalize(text)`：处理连续标点、替换中文标点
  - `english_to_phones(text) -> List[int]`：
    - 词典（可选）→ g2p_en → 过滤 `<unk>` 与无效符号 → 映射至 `symbols2` id
  - 注意保留 `'s` 所有格规则、单字母与专有名词处理（可简化）

- `src/lunavox_tts/Core/Inference.py`：
  - 新增 `tts_en()` 或在 `tts()` 中按语言路由：
    - `text_seq = np.array([english_to_phones(text)], dtype=np.int64)`
    - `text_bert = np.zeros((text_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)`
    - 其余与现有日语流程一致（Encoder → FirstStage → StageDecoder → Vocoder）

- 符号表：
  - 若当前 `Japanese/SymbolsV2.py` 已改造成通用 `SymbolsV2.py`（包含 ARPAbet），可直接复用
  - 否则需在英文模块中导入相同的 `symbols_v2/symbol_to_id_v2`

---

## 文件与代码建议结构
- `src/lunavox_tts/English/EnglishG2P.py`
- 通用 `src/lunavox_tts/Symbols/SymbolsV2.py`（可将日语与中文共享的 `symbols2` 放此处，并迁移引用）
- `src/lunavox_tts/Core/Inference.py` 新增 `tts_en()`

---

## 兼容性与注意事项
- ID 对齐：必须严格匹配 v2 的 `symbols2`，否则 T2S 预测将错位。
- 短文本：若英文 phones < 4，可在最前添加 `,` 稳定推理（与 GPT-SoVITS 保持一致策略）。
- 词典热更新：可添加简易的热词文件以覆盖专有名词读音。

---

## 最小改动清单（实施指南）
1) 引入 `symbols2` 并导出 `symbols_v2/symbol_to_id_v2`
2) 新增 `EnglishG2P.py`（g2p_en + 可选词典），输出 phones→IDs
3) 在 `Inference.py` 添加英文合成入口（或统一按语言路由）
4) 保持 `text_bert` 为零矩阵即可，无需新增 BERT 模型

---

## 示例：英文调用（建议 API）
```python
from lunavox_tts.Core.Inference import tts_client

sr, audio = tts_client.tts_en(
    text="Hello, LunaVox!",
    prompt_audio=ref,
    encoder=encoder_sess,
    first_stage_decoder=first_stage_sess,
    stage_decoder=stage_decoder_sess,
    vocoder=vocoder_sess,
)
```

