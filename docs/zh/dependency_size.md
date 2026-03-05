# 📦 依赖体积分析报告

LunaVox 采用高度**模块化**的设计，支持极致的“按需加载（Lazy Loading）”。您只需要为实际使用的功能付出磁盘空间。

本文档罗列了不同部署场景下的**理论最小磁盘占用**（基于 ONNX Runtime CPU 环境）。

---

## 📊 总结：最小磁盘占用一览

| 模式 | English (英) | Japanese (日) | Chinese (中) | 关键增量说明 |
| :--- | :--- | :--- | :--- | :--- |
| **V2 Persona** | **~307 MB** | **~306 MB** | **~710 MB** | 核心 + G2P (中文需 RoBERTa) |
| **V2 Reference** | **~687 MB** | **~686 MB** | **~1.1 GB** | + HuBERT (~380MB) |
| **V2PP Persona** | **~312 MB** | **~311 MB** | **~715 MB** | V2PP 模型稍大 |
| **V2PP Reference** | **~772 MB** | **~771 MB** | **~1.2 GB** | + PromptEncoder & SpeakerVector |

> **注意**: 以上数据含 Python 源码、依赖库及模型文件。不含系统级库（如 CUDA）。

---

## 🧩 组件拆解

### 1. 基础引擎 (Base Engine) - ~60 MB
所有模式的基石。
*   Python 源码
*   `numpy`, `sounddevice` 等基础库
*   `onnxruntime` (CPU 版)

### 2. 核心声学模型
*   **V2 标准版**: ~240 MB (VITS + 文本编码器 + 解码器)
*   **V2 Pro Plus**: ~245 MB (V2PP VITS + 文本编码器 + 解码器)

### 3. 语言资源 (Language Assets)
*   **英语/日语**: < 10 MB (G2P 字典)
*   **中文**: ~400 MB (RoBERTa 模型 - 必须，用于韵律预测)

### 4.零样本/克隆资源 (Zero-Shot Assets)
*   **HuBERT**: ~380 MB (任何参考音频模式的必须项)
*   **Speaker Vector (SV)**: ~20 MB (V2PP 参考模式必须)
*   **Prompt Encoder**: ~60 MB (V2PP 参考模式必须)

---

## 💡 典型配置场景指南

### 场景 A: 极致轻量化英语 TTS
**目标**: 离线英语 TTS，仅使用预设角色 (Persona)，无需克隆。
*   **基础**: 60 MB
*   **V2 模型**: 240 MB
*   **英语 G2P**: 7 MB
*   **总计**: **~307 MB** 🚀

### 场景 B: 旗舰级中文声音克隆
**目标**: 高质量中文 TTS，支持从音频克隆音色。
*   **基础**: 60 MB
*   **V2PP 模型**: 245 MB
*   **中文 G2P + RoBERTa**: 410 MB
*   **HuBERT**: 380 MB
*   **Prompt Enc + SV**: 80 MB
*   **总计**: **~1.2 GB**

### 场景 C: GPU 加速
如果您需要 GPU 加速 (CUDA)：
*   新增 `onnxruntime-gpu`: +200 MB
*   新增 CUDA 运行库: +500 MB 至 1 GB (若系统未安装)
