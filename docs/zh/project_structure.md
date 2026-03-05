# LunaVox 项目结构指南 🏗️

本文档详细介绍了 LunaVox 的代码组织结构与核心模块设计。LunaVox 采用了模块化分层架构，旨在实现低耦合、高内聚与极致的推理性能。

## 📂 顶层目录概览

LunaVox 的核心源码位于 `src/lunavox_tts` 目录下。

```
src/lunavox_tts/
├── API/              # 📢 公共接口层：对外暴露的稳定 API
├── Core/             # ⚙️ 核心引擎层：TTS 业务逻辑、推理流水线、前端处理
├── Interface/        # 🖥️ 交互层：命令行客户端 (CLI) 与 HTTP 服务端
├── Languages/        # 🌍 多语言层：中文、英文、日文的 G2P 与文本归一化逻辑
├── Resources/        # 📦 资源层：数据结构 (Audio, Persona) 与静态资源管理
├── Utils/            # 🛠️ 工具层：环境管理、资源下载、生命周期管理
├── ModelManager.py   # 🧠 模型生命周期管理 (Facade)
└── ResourceManager.py# 💎 全局资源持有者 (Singleton)
```

---

## 🧩 核心模块详解

### 1. API 层 (`/API`)
这是用户与开发者接触的最外层。它屏蔽了底层复杂的对象管理，提供了简单直观的函数。
*   `synthesis.py`: 核心合成函数（`tts`, `tts_async`）。
*   `characters.py`: 角色加载与卸载、参考音频设置。
*   `state.py`: 维护轻量级的运行时状态配置。

### 2. 核心引擎层 (`/Core`)
TTS 的心脏地带，负责将文本转化为音频流。
*   `TTSPlayer.py`: 业务中控台。管理播放队列、合成线程与回调事件。
*   `Session.py`: 定义单词合成的上下文状态 (`SynthesisSession`)，确保多线程安全。
*   **Processors/**: 各种推理前的特征提取器（如 `feature_extractor.py` 用于提取 SSL/HuBERT 特征）。
*   **Model/**: ONNX 模型加载器与推理策略。
*   **Frontend/**: 文本前端管道，负责分词、注音与韵律预测。

### 3. 资源与模型管理 (The Managers)
这是 LunaVox 架构中最关键的部分，负责平衡性能与内存占用。

*   **`ResourceManager.py` (New)**:
    *   **职责**: 全局共享资源（Global Shared Resources）的**唯一合法持有者**。
    *   **管理对象**: HuBERT 模型 (SSL), BERT 模型 (语义特征)。
    *   **特点**: 真正的单例，确保重型资源在内存中只存在一份。

*   **`ModelManager.py`**:
    *   **职责**: 角色模型（Character Models）的生命周期管理外观 (Facade)。
    *   **功能**: 负责 `v2/v2pp` 模型的加载、LRU 缓存淘汰、以及向 `ResourceManager` 代理查询全局资源。

*   **`Utils/RuntimeManager.py`**:
    *   **职责**: 运行时状态调度员。
    *   **功能**: 并不直接持有资源，而是作为“指挥官”调用各个模块的标准 `unload()` 接口，用于基准测试时的内存彻底大扫除。

*   **`Utils/AssetManager.py`**:
    *   **职责**: 按需下载器。
    *   **功能**: 基于 HuggingFace Hub，根据语言需求（如“仅中文”或“仅日文”）懒加载对应的资源包，极大减少首包体积。

### 4. 资源层 (`/Resources`)
*   **Audio/**: 音频相关的数据结构。
    *   `ReferenceAudio.py`: 核心数据类，封装了原始音频、16k 重采样音频及提取后的特征缓存。
    *   `SpeakerVector.py`: 包含说话人向量提取模型（ERes2NetV2）。
*   **Persona/**: “预设/花火模式” (Persona) 的配置构架与管理器。

---

## 💡 关键架构设计

### 1. 显式的模块化导入
为了避免复杂的循环依赖，项目严格遵守显式导入原则。
*   依赖链方向：`RunTimeManager` -> `ModelManager` -> `ResourceManager`。
*   所有的“重型模块”在 `Utils/RuntimeManager.py` 中通过标准接口进行管理，不再使用函数内隐式导入 (Inline Imports)。

### 2. 极致的懒加载 (Lazy Loading)
LunaVox 默认**不加载任何模型**。
*   只有当用户调用 `load_character("Name")` 时，角色模型才会被加载。
*   只有当合成中文文本时，`ZhBert` 和 `HuBERT` 才会被触发下载并加载。
*   这种设计使得 LunaVox 可以在极低内存环境下启动。

### 3. 可测试性优先
*   引入 `ResourceManager` 使得单元测试可以轻松 Mock 全局资源。
*   `API` 与 `Core` 的分离使得我们可以独立测试推理逻辑，而不受播放硬件的影响。

---
*文档更新日期: 2026-01-02*
