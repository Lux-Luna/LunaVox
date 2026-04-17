# 运行时 Engine

运行时绑定是 LunaVox **主要的嵌入接口**。GUI、脚本、notebook 都通过
ctypes 直接调用 `lunavox.runtime.Engine`——不经过 subprocess，不解析 stdout。
C 句柄用 RAII 风格管理，`with Engine(...) as eng:` 代码块自动释放，
不会泄漏。

自 v2.2.0 起，公共 API 收敛成单一入口 `synthesize(text, voice, params)`；
每一种发声模式都通过 `Voice.<factory>()` 构造对象传入，新增模式只需扩
一行工厂方法，不再是新增一个 `synthesize_*` 方法。

> **英文 API 参考是唯一真源**：方法签名、参数表、返回类型等详细 autodoc
> 内容都渲染在
> [**English → Runtime Engine**](../../en/api/runtime.md) 页面。本页只负责
> 翻译导言和用法示例，避免中英双份 autodoc 彼此冲突的交叉引用。

## 示例

```python
from pathlib import Path

from lunavox.runtime import Engine, SynthesisParams, Voice

with Engine(Path("models/base_small")) as eng:
    params = SynthesisParams(temperature=0.6, top_p=1.0, top_k=50)

    # 默认音色
    result = eng.synthesize("你好，来自 LunaVox。", voice=Voice.base(), params=params)

    # 参考音频或预计算特征克隆（.wav 或 .json）
    cloned = eng.synthesize(
        "用克隆出的声音说话。",
        voice=Voice.clone_file("ref/ref_0.6B.json"),
        params=params,
    )

    # 使用内置发音人，可选风格指令
    custom = eng.synthesize(
        "请用生气的语气说这句话。",
        voice=Voice.custom("Vivian", instruct="Use angry tone."),
        params=params,
    )

    # 根据文字描述设计一个新声音
    designed = eng.synthesize(
        "它就在最上面的抽屉里……等等，怎么是空的？",
        voice=Voice.design(instruct="Speak in an incredulous tone."),
        params=params,
    )

    print(f"RTF: {result.stats.rtf:.3f}")
    print(f"Peak RSS delta: {result.stats.mem.rss_peak_delta_bytes / 1024**2:.1f} MB")
    if result.stats.mem.vram_measured:
        print(f"Peak VRAM delta: {result.stats.mem.vram_peak_delta_bytes / 1024**2:.1f} MB")
    # result.audio 是 numpy.float32 数组，单通道，范围 [-1, 1]
```

## 导航

- **[`Engine`](../../en/api/runtime.md#lunavox.runtime.engine.Engine)**
  ——上下文管理器、唯一合成入口、`on_log` 日志回调
- **[`Voice`](../../en/api/runtime.md#lunavox.runtime.voice.Voice)**
  ——`Voice.base()` / `Voice.clone_file()` / `Voice.custom()` / `Voice.design()`
- **[`SynthesisParams`](../../en/api/runtime.md#lunavox.runtime.params.SynthesisParams)**
  ——`temperature` / `top_p` / `top_k` / `repetition_penalty` 等
- **[`SynthesisResult`](../../en/api/runtime.md#lunavox.runtime.params.SynthesisResult)**
  ——PCM numpy 数组 + 时间戳 + RTF + 内存高水位
- **[`SynthesisMode`](../../en/api/runtime.md#lunavox.runtime.params.SynthesisMode)**
  ——`BASE` / `CLONE_FILE` / `CUSTOM` / `DESIGN` 枚举
- **[`LunavoxLibraryError` / `LunavoxSynthesisError`](../../en/api/runtime.md#lunavox.runtime.errors.LunavoxLibraryError)**
  ——错误分层
