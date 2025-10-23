# LunaVox ONNX 与 GPT-SoVITS 原项目（PyTorch/ONNX）策略对比（中文）

## 范围与前置
- 对比对象：LunaVox（ONNX 推理） vs GPT-SoVITS 原项目（PyTorch 推理 / 官方 ONNX 导出脚本）。
- 涵盖版本：v2、v2Pro、v2ProPlus（不含 v3/v4）。
- 依据来源：本仓库实现与 `performance_tests/results/test_results.json` 数据；GPT-SoVITS 项目内的 `inference_webui.py`、`TTS_infer_pack/TTS.py`、`GPT_SoVITS/onnx_export.py` 等。

---

## 一、模型与图结构
- LunaVox：
  - 提供三段 T2S（Encoder / First-Stage Decoder / Stage Decoder）与 VITS 的 ONNX 模型。
  - StageDecoder 通过输入/输出 past/present 张量进行自回归迭代（Python 驱动循环）。
  - v2Pro/v2ProPlus 另使用 STFT/Mel 特征与说话人向量（SV，ONNX 版 ERes2NetV2）。
- GPT-SoVITS（PyTorch）：
  - 端到端在 PyTorch 内循环采样，KV 缓存与注意力在框架内完成。
  - 新版 `TTS_infer_pack/TTS.py` 支持批量、分桶、并行优化。
- GPT-SoVITS（ONNX 脚本）：
  - 提供 `onnx_export.py` 示例，将 T2S/VITS 拆分导出，但仍由 Python 循环推进。

结论：两者的“图内循环”均未启用（未用 Scan 等），性能差异更多来自前后处理/数据布局与运行时。

---

## 二、KV Cache 与解码策略
- LunaVox：
  - 以 past/present 张量在 ONNX 与 Python 间传递，按步 `session.run()`；
  - 兼容不同模板输出（聚合 KV 或分层 KV），并在需要时自动拆分/拼合。
- GPT-SoVITS（PyTorch）：
  - 依赖框架内的注意力与缓存，省去 ONNX 的 I/O 往返，但计算在 PyTorch。

推断：在 CPU 上，LunaVox 通过 ONNXRuntime 的优化有潜在优势；在高性能 GPU 上，PyTorch 端到端可能更具吞吐与灵活性。

---

## 三、I/O 与 Provider 策略
- LunaVox：
  - 默认 `CPUExecutionProvider`；当前主链路未启用 I/O Binding 与 CUDA Provider。
  - 支持多模型缓存与临时权重（FP16→FP32）转换，降低重复加载成本。
- GPT-SoVITS（PyTorch）：
  - GPU 上可用 FP16/混合精度；CPU 上强制 FP32。
- GPT-SoVITS（ONNX）：
  - 官方脚本示例以功能演示为主，未系统化 I/O Binding 与多 Provider 路线。

结论：若 LunaVox 启用 CUDA Provider 与 I/O Binding，有望进一步降低步进开销；当前更偏向 CPU 友好部署。

---

## 四、文本前端与语义/说话人特征
- 共同点：
  - JA/EN/ZH 多语言文本前端；中文支持 BERT 词-音素对齐（ZH-BERT）。
  - v2Pro/v2ProPlus 支持说话人向量（SV）。
- 差异：
  - LunaVox 将 CN-HuBERT、ERes2NetV2 也以 ONNX 方式加载，避免 PyTorch 依赖。

---

## 五、体积与性能（本仓库样例数据）
- 模型尺寸：
  - LunaVox v2 ≈ 236.7MB；v2ProPlus ≈ 641.9MB。
- 运行时大小（依赖）：
  - LunaVox 约 1.34GB（Windows 样例，site-packages 或导出环境）。
- 首包延迟（平均，短句 JA，CPUExecutionProvider）：
  - LunaVox v2 ≈ 1.76s；v2ProPlus ≈ 2.08s。
- GPT-SoVITS 对比数据：
  - 本仓库未在同一环境下获得稳定的 PyTorch 端实测数据，测试用例存在依赖缺失/启动失败，故不作直接数值比较。

---

## 六、理论提升点与“何时不提升”
- 可能的提升来源：
  - **CPU 场景**：ONNXRuntime 的图优化与算子实现对部分注意力/卷积更友好。
  - **多模型缓存**：减少加载；FP16→FP32 临时转换避免精度问题。
  - **ONNX 化的前端/特征模型**：避免拉入 PyTorch 体积与冷启动成本。
- 可能“不明显提升”的情形：
  - **高端 GPU + PyTorch 优化**：PyTorch 端到端可用半精度/融合内核，极短文本时 ONNX 每步 `run()` 成本抵消计算优势。
  - **I/O Binding 未启用**：频繁 host-device 往返或内存拷贝会吞噬收益（当前 LunaVox 主链路未开启）。
  - **图内循环未启用**：仍由 Python 驱动循环，微秒级开销累积在极短句上更显著。

---

## 七、结论与建议
- LunaVox 现状：更适合 CPU 部署、无 PyTorch 环境的落地；体积与依赖可控，短句延迟较低。
- 若追求进一步性能：
  - 在支持环境下启用 `CUDAExecutionProvider` 与 I/O Binding；
  - 探索将解码步进迁入图内（Scan/Loop）以减少 `session.run` 次数；
  - 结合动态量化/运算符替换评估 CPU 性能收益。

> 总结：LunaVox ONNX 路线在“仅推理、轻量部署、CPU 优先”的场景具备优势；若使用高端 GPU 且可接受 PyTorch 依赖，原项目的 PyTorch 推理可能拥有更好的可调度性与峰值吞吐，二者各有侧重。
