### LunaVox 英语合成功能实现总结（v2 模型）

---

## 改动概览
- 新增：`src/lunavox_tts/English/EnglishG2P.py`、`src/lunavox_tts/English/__init__.py`
- 路由：`Core/Inference.py` 在 `tts()` 增加 `language` 参数（默认 `ja`，支持 `en`）
- 引用：`Audio/ReferenceAudio.py` 根据参考文本自动选择日/英 G2P（启发式）
- 状态：`Utils/Shared.py` 新增 `current_language` 上下文字段
- 播放：`Core/TTSPlayer.py` 调用 `tts()` 时注入 `context.current_language`
- WebUI：`WebUI/webui.py` 新增语言选择下拉（ja/en），保持日语逻辑不变
- 依赖：`requirements.txt` 增加 `g2p_en`、`wordsegment`、`nltk`

---

## 详细逻辑
- 英文 G2P（与 GPT-SoVITS 接近的路径）：
  - 标点清洗与 English 文本正则展开：`English/en_normalization/expend.py`（移植自 GPT-SoVITS）
  - 使用 `g2p_en` 进行句级推理（避免 NLTK 依赖下载），并做 `symbols2` 过滤
  - 短文本（phones < 4）自动前置逗号
  - 输出映射至 `symbols_v2/symbol_to_id_v2`

- 推理路径：
  - 非中文语言 `text_bert` 维持零矩阵（与 GPT-SoVITS 保持一致）
  - 仅切换文本前端（phones），ONNX Encoder/Decoder/Vocoder 无需调整

---

## WebUI 使用说明
1) 选择角色与参考音频
2) 语言选择：`ja` 或 `en`
3) 输入对应语言文本，点击“开始合成”
4) 参考文本将用于对齐提示（自动日/英 G2P），保持兼容

---

## 测试建议
- 基础用例：
  - EN: "Hello, LunaVox!"（phones >= 4，合成成功）
  - EN 短句："Hi."（自动加逗号，合成成功）
  - JA: 现有用例应保持不变
- 参考音频：
  - 英文参考文本将走英文 G2P；日文参考文本保持原有逻辑
- 兼容性回归：
  - 不选择语言默认为 `ja`（与旧版一致）
  - CLI/API 仍可通过 `tts(language=...)` 指定语言

---

## 后续可选优化
- 词典热更新（CMU 热词表）对齐 GPT-SoVITS 英文前端
- 语言自动检测（参考 GPT-SoVITS LangSegmenter）
- 统一 `SymbolsV2` 至公共目录，减少重复导入


