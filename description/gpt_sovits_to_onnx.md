### GPT-SoVITS 模型（ckpt/pth）转 ONNX：设计与实现说明

本文档说明本项目中“将 GPT-SoVITS 训练产出的 PyTorch 权重（.ckpt/.pth）转换为本项目可用的 ONNX 及外置权重 .bin”的完整实现，包括目录结构、核心逻辑、流程与产出物。

---

## 功能概述

- **目标**：将 GPT-SoVITS 的两类 PyTorch 权重
  - GPT/T2S 相关：`.ckpt`
  - VITS 相关：`.pth`
  转换为项目推理所需的 ONNX 模型与配套外置权重 `.bin`。

- **入口函数**：`src/lunavox_tts/_internal.py` 中的 `convert_to_onnx`。
  - 依赖 PyTorch，仅做权重重排、精度转换与 ONNX external data 绑定；不涉及训练。
  - 示例用法见 `Tutorial/English/How to Convert Models.py`：

```python
import lunavox_tts as lunavox

lunavox.convert_to_onnx(
    torch_pth_path=r"<PATH_TO_TORCH_PTH_FILE>",
    torch_ckpt_path=r"<PATH_TO_TORCH_CKPT_FILE>",
    output_dir=r"<OUTPUT_DIRECTORY>"
)
```

---

## 目录结构与关键文件

- `src/lunavox_tts/_internal.py`
  - `convert_to_onnx(torch_ckpt_path, torch_pth_path, output_dir)`：转换外部公开入口。
  - 内部调用 `src/lunavox_tts/Converter/v2/Converter.py::convert`。

- `src/lunavox_tts/Converter/v2/Converter.py`（调度器）
  - 通过 `importlib.resources` 读取打包在包内的 ONNX 模板与键名清单：
    - 模板 ONNX：
      - `Data/v2/Models/t2s_encoder_fp32.onnx`
      - `Data/v2/Models/t2s_stage_decoder_fp32.onnx`
      - `Data/v2/Models/t2s_first_stage_decoder_fp32.onnx`
      - `Data/v2/Models/vits_fp32.onnx`
    - 键名清单：
      - `Data/v2/Keys/t2s_onnx_keys.txt`
      - `Data/v2/Keys/vits_onnx_keys.txt`
  - 依次执行三个转换器：
    - `T2SModelConverter`（T2S/GPT，输入 `.ckpt`）
    - `VITSConverter`（VITS，输入 `.pth`）
    - `EncoderConverter`（混合：从 `.ckpt` 与 `.pth` 合并 encoder 所需权重）

- `src/lunavox_tts/Converter/v2/T2SConverter.py`
  - 针对 T2S（GPT）权重的转换：
    - `step1_create_fp16_bin_with_key_mapping()`：
      - 读取 `.ckpt` 的 `state_dict['weight']`
      - 按 `t2s_onnx_keys.txt` 的顺序取权重，并将键名从 `transformer_encoder.*` 映射为 ckpt 中的 `model.h*` 路径（规则：`transformer_encoder` → `h`，并加前缀 `model.`）。
      - 以 fp16 写入 `t2s_shared_fp16.bin`，同时建立“按 fp32 尺寸”计算的 `index_table`（offset/length）。
    - `step2_relink_onnx_for_fp32(old_model, new_model)`：
      - 按索引表修改 ONNX initializer：清空 `raw_data`，改为 `EXTERNAL` 存储，并设置 `location/offset/length` 指向未来的 `t2s_shared_fp32.bin`。
    - `step3_reconstruct_fp32_bin_from_fp16(fp16_bin, out_fp32_bin)`（静态方法）：
      - 将 `t2s_shared_fp16.bin` 逐元素转换为 `t2s_shared_fp32.bin`。
    - `run_full_process()`：执行 step1 + step2（不主动执行 step3）。

- `src/lunavox_tts/Converter/v2/VITSConverter.py`
  - 针对 VITS `.pth` 权重的转换：
    - `step1_create_fp16_bin_and_fp32_index()`：
      - 读取 `vits_onnx_keys.txt`；从 `.pth` 的 `state_dict['weight']` 中取对应权重（如有 `vq_model.` 前缀则去掉用于查找）。
      - 以 fp16 写入 `vits_fp16.bin` 并生成“按 fp32 尺寸”计算的索引表。
    - `step2_relink_onnx_for_fp32()`：
      - 将 `vits_fp32.onnx` 的 initializer 改为外部数据，指向稍后生成的 `vits_fp32.bin`。
    - `step3_reconstruct_fp32_bin_from_fp16(fp16_bin, out_fp32_bin)`（静态方法）。
    - `run_full_process()`：执行 step1 + step2（不主动执行 step3）。

- `src/lunavox_tts/Converter/v2/EncoderConverter.py`
  - 为 `t2s_encoder_fp32.onnx` 准备外置全精度权重：
    - 预定义所需 ONNX 初始化器键列表（encoder 与 vits 若干关键权重）。
    - 从 `.ckpt`（前缀 `model.`）与 `.pth`（直接键或去 `vits.` 前缀）提取并拼接为 `t2s_encoder_fp32.bin`（fp32）。
    - 同步修改 `t2s_encoder_fp32.onnx` 为 external data，并正确写入 `location/offset/length`。

- `src/lunavox_tts/Converter/load_state_dict.py`
  - `load_gpt_model(ckpt)`、`load_sovits_model(pth)`：兼容不同打包方式的权重加载。

- 键名清单（决定外置 `.bin` 的布局顺序）
  - `src/lunavox_tts/Data/v2/Keys/t2s_onnx_keys.txt`
  - `src/lunavox_tts/Data/v2/Keys/vits_onnx_keys.txt`

---

## 转换流程（自顶向下）

1) `lunavox_tts.convert_to_onnx(ckpt, pth, out_dir)`
   - 校验本地已安装 PyTorch
   - 调用调度器 `Converter.convert(ckpt, pth, out_dir)`

2) 调度器准备资源
   - 使用包内模板 ONNX 与键名清单（不会联网下载）
   - 创建缓存目录 `./Cache` 与输出目录 `out_dir`

3) 执行三段转换
   - T2S（ckpt → t2s_shared_fp16.bin + relink 两个 T2S 解码器 ONNX）
   - VITS（pth → vits_fp16.bin + relink VITS ONNX）
   - Encoder（ckpt/pth 合成 → t2s_encoder_fp32.bin + relink encoder ONNX）

4) 产物落盘并清理缓存
   - 成功：在 `out_dir` 输出全部 ONNX 与 `.bin`
   - 失败：记录完整堆栈并清理输出与缓存

---

## 输出产物一览（位于 output_dir）

- ONNX（均已改为 external data）：
  - `t2s_encoder_fp32.onnx`
  - `t2s_first_stage_decoder_fp32.onnx`
  - `t2s_stage_decoder_fp32.onnx`
  - `vits_fp32.onnx`

- 权重 `.bin`：
  - 分发用半精度：`t2s_shared_fp16.bin`、`vits_fp16.bin`
  - 全精度（可选，见“运行时处理”）：`t2s_encoder_fp32.bin`（由 EncoderConverter 直接生成）、`t2s_shared_fp32.bin`、`vits_fp32.bin`

说明：T2S 与 VITS 的 fp32 `.bin` 可通过静态方法一步生成，也可在加载阶段自动生成（见下节）。

---

## 推理阶段的配套处理

- `src/lunavox_tts/ModelManager.py` 中：
  - `convert_bins_to_fp32(model_dir)` 会在加载模型前检测 `*_fp32.bin` 是否存在，若缺失则从对应的 `*_fp16.bin` 转换得到，以提升推理速度：
    - `t2s_shared_fp16.bin` → `t2s_shared_fp32.bin`
    - `vits_fp16.bin` → `vits_fp32.bin`
  - 随后通过 `onnxruntime.InferenceSession` 加载上述 ONNX（其 initializer 指向外部 fp32 `.bin`）。

---

## 键映射与一致性保证

- T2S（ckpt → onnx）
  - 参考 `t2s_onnx_keys.txt` 的顺序构建外置 `.bin` 布局。
  - 键映射：`transformer_encoder.*` → `model.h*`（即先替换根名，再加 `model.` 前缀）。

- VITS（pth → onnx）
  - 参考 `vits_onnx_keys.txt` 的顺序。
  - 查权重时若 ONNX 键以 `vq_model.` 开头，则去掉此前缀再在 `.pth` 的 `state_dict['weight']` 中查找。

- Encoder（ckpt + pth → onnx）
  - 预定义少量关键 initializer 键，分别从 ckpt 的 `model.*` 和 pth 的 `vits.*` 路径提取。

---

## 错误处理与日志

- 缺少 PyTorch：入口函数直接报错并返回。
- 缺少输入文件或索引/键名清单：抛出 `FileNotFoundError`。
- 权重键缺失或结构异常：抛出 `ValueError/KeyError` 并打印详细堆栈。
- 输出目录非空：发出警告但继续执行。

---

## 使用说明（简）

1) 安装依赖（简化示例）：
   - 必需：`torch`、`onnx`、`onnxruntime`

2) 运行转换：
   - 参考前文代码片段，提供 `.ckpt`、`.pth` 与输出目录。

3) 推理使用：
   - 将输出目录作为角色模型目录加载（需同时准备 HuBERT 与 OpenJTalk 词典等运行时资源）。

---

## 设计要点回顾

- 使用 ONNX external data 将大权重放入外置 `.bin`，避免 ONNX 文件过大并提升加载灵活性。
- 采用 fp16 分发、推理前转为 fp32（或在转换阶段直接生成 fp32），平衡体积与速度。
- 键名清单驱动的“顺序即布局”策略，确保 `.bin` 的 offset/length 与 ONNX initializer 一一对应。


