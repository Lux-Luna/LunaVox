# 修复：T2S ONNX 模型不存在 `increased_y_emb` 输出导致的报错

## 问题概述
- 运行 `quick_tryout.py` 时，T2S 阶段解码器在运行时请求了名为 `increased_y_emb` 的输出，但实际导出的 ONNX 模型并不包含该输出，导致推理报错。
- 正确的输出名为 `y_emb`。因此只需要将运行时代码中绑定的输出名从 `increased_y_emb` 更正为 `y_emb`。

## 变更内容
- 修正 C++ 运行时的输出名绑定：
  - 文件：`runtime/src/T2SOnnxCPURuntime.cpp`
  - 位置：阶段解码函数 `step_decode` 的输出绑定处
  - 变更要点：将 `stage_decoder_iobinding.BindOutput("increased_y_emb", ...)` 更改为 `stage_decoder_iobinding.BindOutput("y_emb", ...)`

- 改进 Windows 构建脚本兼容性以便顺利本地编译：
  - 文件：`setup.py`
  - 逻辑：当 `CMAKE_GENERATOR` 为 Ninja 时，不再附加 `-A x64`（该参数与 Ninja 生成器不兼容）；使用 VS 生成器时保持原有行为。

## 影响范围与兼容性
- 该修复仅更正输出名称，不涉及推理计算逻辑与张量形状，属于无副作用的小修复。
- Python 侧接口未发生变化，`quick_tryout.py` 与 `lunavox_tts` 其余模块无需修改。
- 构建脚本调整仅影响本地编译流程，不影响运行时行为。

## 验证步骤（已通过）
1. 本地重新编译原生扩展：
   - 建议在 PowerShell 中指定 VS 生成器：`$env:CMAKE_GENERATOR='Visual Studio 17 2022'`
   - 执行：`python setup.py build_ext --inplace`
2. 运行 `quick_tryout.py`：
   - 预期：不再出现请求 `increased_y_emb` 的报错；能够正常加载模型并完成推理流程。

## 相关文件
- `runtime/src/T2SOnnxCPURuntime.cpp`（绑定输出名修正）
- `setup.py`（Windows 生成器兼容性改进）

