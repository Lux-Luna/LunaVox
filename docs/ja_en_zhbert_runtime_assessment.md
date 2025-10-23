## LunaVox 运行时与中文语义资源（ZH-BERT）加载检测报告

### 结论
- **运行时体积是否包含中文语义资源**：若打包/部署时包含 `LunaVox/Data/chinese-roberta-wwm-ext-large`，则会被计入磁盘占用（本机约 3.73 GB）。
- **JA/EN 推理是否加载 ZH-BERT**：不会。代码仅在中文路径调用 ZH-BERT，实测 JA/EN 推理未触发 ZH-BERT 文件加载。
- **是否可在仅 JA/EN 场景不携带该资源**：可以。删除或不随包提供 `LunaVox/Data/chinese-roberta-wwm-ext-large` 不影响 JA/EN 合成；仅在中文合成时才需要该资源（且会自动下载到 Data 目录）。

---

### 证据一：代码逻辑只在中文路径加载 ZH-BERT
- 文本特征（推理主流程）：
```28:43:LunaVox/src/lunavox_tts/Core/Inference.py
        elif language == "zh":
            ids, word2ph, norm_text = chinese_clean_g2p_and_norm(text)
            text_seq: np.ndarray = np.array([ids], dtype=np.int64)
            # Full zh-BERT parity: compute 1024-d features and align to phones
            bert_phone = compute_bert_phone_features(norm_text, word2ph)  # (len_phones, 1024)
            if bert_phone.shape[0] != text_seq.shape[1]:
                text_bert = np.zeros((text_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)
            else:
                text_bert = bert_phone
        else:
            text_seq: np.ndarray = np.array([japanese_to_phones(text)], dtype=np.int64)
            text_bert = np.zeros((text_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)
```
- 参考音频文本 BERT（仅中文时计算）：
```141:149:LunaVox/src/lunavox_tts/Audio/ReferenceAudio.py
def _compute_reference_bert(
    language: str, norm_text: str, word2ph: list[int], phone_len: int
) -> np.ndarray:
    if language == "zh" and phone_len:
        bert = compute_bert_phone_features(norm_text, word2ph)
        if bert.shape[0] == phone_len:
            return bert.astype(np.float32)
    return np.zeros((phone_len, BERT_FEATURE_DIM), dtype=np.float32)
```

---

### 证据二：实测结果（脚本 `scripts/check_zh_bert_usage.py`）
- 输出文件：`performance_tests/results/lunavox_zh_bert_check.json`
- 关键结果（本机）：
  - `zh_bert_dir_mb`: 3730.45（`LunaVox/Data/chinese-roberta-wwm-ext-large` 目录体积约 3.73 GB）
  - 文件句柄采样：
    - `ja`: touched_zh_bert_files=false
    - `en`: touched_zh_bert_files=false
    - `zh`: 采样为 false，但控制台日志出现 HuggingFace 加载提示（说明中文路径会触发模型载入）
- 说明：文件句柄是快照采样，中文路径的权重加载较快，可能在采样窗口外关闭；但从控制台日志可见中文时加载了 ZH-BERT 权重。

---

### 证据三：载入来源与回退机制（仅中文用到）
- 当 `ZH_BERT_BASE_PATH` 或本地 Data 目录存在时，优先本地加载；否则自动从 `hfl/chinese-roberta-wwm-ext-large` 下载至 `LunaVox/Data/chinese-roberta-wwm-ext-large`。

---

### 实务建议
- **仅 JA/EN 部署**：
  - 可不携带 `LunaVox/Data/chinese-roberta-wwm-ext-large`，磁盘占用可减少约 3.73 GB；
  - 运行时不会加载 ZH-BERT，JA/EN 推理不受影响；
  - 若后续需要中文，请预置该目录或允许首次自动下载。
- **需包含中文合成**：
  - 建议预置 `LunaVox/Data/chinese-roberta-wwm-ext-large`，以避免线上拉取；
  - 或设定 `ZH_BERT_BASE_PATH` 指向本地只读模型目录。

---

### 复现实测（可选）
- 运行检测脚本（无需改动项目）：
```bash
python scripts/check_zh_bert_usage.py
```
- 查看输出：`performance_tests/results/lunavox_zh_bert_check.json`

---

### 结语
- 当前实现仅在中文路径计算 BERT 特征，JA/EN 路径不加载中文语义模型；
- 因此可以在仅 JA/EN 的发布物中去除中文语义权重目录，以显著缩小运行时体积。
