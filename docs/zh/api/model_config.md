# 模型目录

模型目录是所有 Qwen3-TTS 变体的**唯一真源**。下载器、转换流水线、CLI 交互提示
——每一个模块都从这里读取 `MODELS`。新增或重命名模型只需要改这一个文件。

> **英文 API 参考是唯一真源**：字段定义与方法签名详见
> [**English → Model Catalog**](../../en/api/model_config.md)。本页只保留
> 中文导言和用法示例。

## MODELS 注册表

```python
from lunavox.model import MODELS, all_models, get_model

# 按内部短名键控的字典
print(list(MODELS.keys()))
# ['base_small', 'custom_small', 'base', 'custom', 'design']

# 保留注册顺序的有序列表
for spec in all_models():
    print(f"{spec.name:15s}  {spec.size}  mode={spec.mode}  repo={spec.repo_id}")

# 直接查询，未知键抛 ValueError
spec = get_model("base_small")
```

## 导航

- **[`ModelSpec`](../../en/api/model_config.md#lunavox.model.config.ModelSpec)**
  ——不可变 dataclass，每一个 Qwen3-TTS 变体的目录项
- **[`MODELS`](../../en/api/model_config.md#lunavox.model.config.MODELS)**
  ——`{name: spec}` 字典，默认遍历顺序
- **[`all_models` / `model_keys` / `get_model`](../../en/api/model_config.md#lunavox.model.config.all_models)**
  ——查询 API
- **[`get_snapshot`](../../en/api/model_config.md#lunavox.model.config.get_snapshot)**
  ——HuggingFace 缓存路径解析
- **[`Models` / `ModelConfig`](../../en/api/model_config.md#lunavox.model.config.Models)**
  ——项目根绑定的视图
