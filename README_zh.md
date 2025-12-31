<div align="center">

# 🔮 LunaVox

**[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 轻量级推理引擎：高品质、极速、跨平台。**

[简体中文](./README_zh.md) | [English](./README.md)

</div>

---

## 🚀 项目简介

**LunaVox** 是专为 GPT-SoVITS 生态打造的轻量化推理引擎。通过解耦重量级依赖并基于纯 ONNX Runtime 实现，它提供了一个便携、高效且易于集成的跨平台语音合成解决方案。

- **极速推理**：针对 ONNX Runtime 深度优化。引入 **I/O Binding** 实现 GPU 内存零拷贝，并结合 **KV-Cache** 策略，大幅降低 autoregressive (自回归) 过程中的推理延迟。
- **轻量便携**：依赖极简，内置自动资源管理（模型及组件按需下载）。
- **全方位支持**：完美支持 GPT-SoVITS V2、Pro 及 Pro Plus 模型，涵盖中、英、日三语。

---

## � 项目结构

- **`src/lunavox_tts`**：核心代码与推理逻辑。
- **`GUI`**：跨平台桌面客户端，包含模型管理、合成、转换及设置。
- **`Tutorial`**：快速上手教程，包含推理示例及模型转换示例。
- **`CharacterData`**：存放本地角色声音模型的目录。
- **`TTSData`**：公共资源目录（G2P、HuBERT、说话人向量等）。*初次运行自动下载*。
- **`RoBERTa`**：中文专用 BERT 资源。*初次运行自动下载*。

---

## 🎙 声音克隆说明

如果要实现完美的角色声音克隆，需要使用 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 项目自行训练或下载他人训练好的 PyTorch 模型（`.ckpt` 和 `.pth`），并采用 LunaVox 的转换工具进行转换。

但如果你不介意无法完美还原角色声音，LunaVox 已经提供了 V2 和 V2 Pro Plus 版本的 **Pretrained（预训练）模型**，你只需要提供一段 `.wav` 格式的参考音频即可开始用于各种角色的推理。

> [!IMPORTANT]
> **重要提示**：为了达到最好的推理效果，强烈建议您使用与**目标语言一致**的参考音频，并设置**正确的参考音频内容文本**。

---

## 🏁 快速开始

### 1. 安装依赖

```bash
pip install lunavox-tts
```

### 2. 启动可视化界面 (GUI)

最推荐的使用方式，提供即开即用的角色切换、推理测试和模型转换功能：

```bash
python GUI/main.py
```

### 3. 代码示例

```python
import lunavox_tts as lunavox

# 1. 加载角色模型 (支持 v2 / v2Pro / v2ProPlus)
lunavox.load_character('my_char', './CharacterData/character_model/v2/pretrained')

# 2. 设置参考音频 (用于音色克隆)
lunavox.set_reference_audio('my_char', './ref.wav', '参考音频的文本内容')

# 3. 执行合成
lunavox.tts('my_char', '你好，我是 LunaVox！', play=True, language='zh')
```

---

## 🛠 进阶功能

### 模型转换 (Conversion)
将原版 PyTorch 的 `.ckpt` 和 `.pth` 转换为高效的 ONNX/BIN 集合，方便在任何环境下运行。
详见：`Tutorial/pytorch_to_onnx_bin_demo.py`

### V2 Pro Plus 支持
支持新版的 Pro/Pro Plus 协议，具备更强的音色还原能力。
详见：`Tutorial/tts_tryout/v2pp_en.py`

### 后端服务
可一键部署为高性能的 FastAPI 后端接口。
```python
import lunavox_tts as lunavox
lunavox.start_server(host="0.0.0.0", port=8000)
```

---

## ⛓ 资源手动下载 (中国大陆用户推荐)

由于网络速度限制，建议手动下载预训练资源并解压至对应目录。

| 资源类别 | 对应目录 | 下载链接 |
| :--- | :--- | :--- |
| **LunaVox 全量预训练** | 仓库根目录 | [Hugging Face](https://huggingface.co/Lux-Luna/LunaVox/tree/main) |

*下载后确保目录结构如下：*
```text
LunaVox/
├── CharacterData/
├── TTSData/
└── RoBERTa/
```

---

## ⚙ 运行时配置

LunaVox 使用 `env_manager` 自动管理环境。系统会根据硬件自动安装对应的 ONNX Runtime 及配套的精简版 CUDA 库。

```python
from lunavox_tts.Utils.EnvManager import env_manager
env_manager.set_mode("cpu") # 切换至 cpu 模式 (默认) 或 "gpu" 模式
env_manager.ensure_environment() # 确定环境就绪（如缺失依赖会自动执行静默安装）
```

---

## � 线路图与开发进度 (Roadmap)

### ✅ 已完成 (Completed)
- **多语言核心**
  - [x] 简体中文支持
  - [x] 英语支持
- **模型架构**
  - [x] V2 Pro Plus 模型推理支持
- **性能优化**
  - [x] GPU 加速支持与深度优化 (CUDA/DirectML)
  - [x] IO Binding 内存零拷贝技术 (Device-to-Device)
  - [x] KV Cache 缓存优化 (减少重复计算)

### 🚀 计划中 (Planned)
- **模型扩展**
  - [ ] V2 Pro 模型兼容
- **用户体验**
  - [ ] Windows 一键整合包 (即开即用)
- **高级功能**
  - [ ] 音色固化 (Speaker Embedding 恒定)
  - [ ] 情感控制增强
