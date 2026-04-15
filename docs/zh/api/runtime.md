# 运行时 Engine

运行时绑定是 LunaVox **主要的嵌入接口**。GUI、脚本、notebook 都通过
ctypes 直接调用 `lunavox.runtime.Engine`——不经过 subprocess，不解析 stdout。
C 句柄用 RAII 风格管理，`with Engine(...) as eng:` 代码块自动释放，
不会泄漏。

> **英文 API 参考是唯一真源**：方法签名、参数表、返回类型等详细 autodoc
> 内容都渲染在
> [**English → Runtime Engine**](../../en/api/runtime.md) 页面。本页只负责
> 翻译导言和用法示例，避免中英双份 autodoc 彼此冲突的交叉引用。

## 示例

```python
from pathlib import Path

from lunavox.runtime import Engine, SynthesisParams

with Engine(Path("models/base_small")) as eng:
    params = SynthesisParams(temperature=0.6, top_p=1.0, top_k=50)
    result = eng.synthesize_with_voice_file(
        text="Hello from LunaVox.",
        reference_path="ref/ref_0.6B.json",
        params=params,
    )
    print(f"RTF: {result.stats.rtf:.3f}")
    print(f"Peak RSS: {result.stats.rss_peak_bytes / 1024**2:.1f} MB")
    # result.audio 是 numpy.float32 数组，单通道，范围 [-1, 1]
```

## 导航

- **[`Engine`](../../en/api/runtime.md#lunavox.runtime.binding.Engine)**
  ——上下文管理器、合成入口、生命周期方法
- **[`SynthesisParams`](../../en/api/runtime.md#lunavox.runtime.binding.SynthesisParams)**
  ——`temperature` / `top_p` / `top_k` / `repetition_penalty` 等
- **[`SynthesisResult`](../../en/api/runtime.md#lunavox.runtime.binding.SynthesisResult)**
  ——PCM numpy 数组 + 时间戳 + RTF + 内存高水位
- **[`SynthesisMode`](../../en/api/runtime.md#lunavox.runtime.binding.SynthesisMode)**
  ——Base / CloneFile / CloneSamples / Custom / Design 枚举
- **[`LunavoxLibraryError` / `LunavoxSynthesisError`](../../en/api/runtime.md#lunavox.runtime.binding.LunavoxLibraryError)**
  ——错误分层
- **[`set_log_callback`](../../en/api/runtime.md#lunavox.runtime.binding.set_log_callback)**
  ——安装 Python 端日志 sink，接收每一行 C++ 日志
