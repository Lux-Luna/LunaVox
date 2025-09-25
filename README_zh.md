<div align="center">
<pre>
 ___       ___  ___  ________   ________  ___      ___ ________     ___    ___ 
|\  \     |\  \|\  \|\   ___  \|\   __  \|\  \    /  /|\   __  \   |\  \  /  /|
\ \  \    \ \  \\\  \ \  \\ \  \ \  \|\  \ \  \  /  / | \  \|\  \  \ \  \/  / /
 \ \  \    \ \  \\\  \ \  \\ \  \ \   __  \ \  \/  / / \ \  \\\  \  \ \    / / 
  \ \  \____\ \  \\\  \ \  \\ \  \ \  \ \  \ \    / /   \ \  \\\  \  /     \/  
   \ \_______\ \_______\ \__\\ \__\ \__\ \__\ \__/ /     \ \_______\/  /\   \  
    \|_______|\|_______|\|__| \|__|\|__|\|__|\|__|/       \|_______/__/ /\ __\ 
                                                                   |__|/ \|__| 
</pre>
</div>

<div align="center">

# 🔮 LunaVox: [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 轻量级推理引擎

**专为 GPT-SoVITS 设计的高性能、轻量级的推理引擎**

[简体中文](./README_zh.md) | [English](./README.md)

</div>

---

**LunaVox** 是基于开源 TTS 项目 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 打造的轻量级推理引擎，集成了
TTS 推理、ONNX 模型转换、API Server 等核心功能，旨在提供更极致的性能与更便捷的体验。

- **✅ 支持模型版本:** GPT-SoVITS V2
- **✅ 支持语言:** 日语 (Japanese)

---

## 🚀 性能优势

LunaVox 对原版模型进行了高度优化，在 CPU 环境下展现了卓越的性能。

| 特性        |  🔮 LunaVox | 官方 Pytorch 模型 | 官方 onnx 模型 |
|:----------|:----------:|:-------------:|:----------:|
| **首包延迟**  | **1.13s**  |     1.35s     |   3.57s    |
| **运行时大小** | **~200MB** |     ~数 GB     | 与 LunaVox 类似 |
| **模型大小**  | **~230MB** |  与 LunaVox 类似   |   ~750MB   |

> 📝 **备注:** 由于 GPU 推理的首包延迟与 CPU 相比未拉开显著差距，我们暂时仅发布 CPU 版本，以提供最佳的开箱即用体验。
>
> 📝 **延迟测试说明:** 所有延迟数据基于一个包含 100 个日语句子的测试集，每句约 20 个字符，取平均值计算。在 CPU i7-13620H
> 上进行推理测试。
---

## 🏁 快速开始 (QuickStart)

> **⚠️ 重要提示:** 建议在 **管理员模式 (Administrator)** 下运行 LunaVox，以避免潜在的严重性能下降问题。

### 📦 安装 (Installation)

通过 pip 安装：

```bash
pip install lunavox-tts
```

> 📝 **备注:** 当您尝试安装 pyopenjtalk 时，可能会遇到安装失败的问题。这是因为 pyopenjtalk 是一个包含 C
> 语言扩展模块的库，而其发布者目前没有提供预编译的二进制包 (wheels)。
> 对于 Windows
> 用户，这意味着您必须安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> ，并确保在安装时勾选了 “使用 C++ 的桌面开发” 工作负载。

### ⚡️ 快速体验 (Quick Tryout)

手上还没有 GPT-SoVITS 模型？没关系！

为了让您能够轻松上手，LunaVox 内置了预设的说话人角色。无需任何模型文件，只需运行下面的脚本，即可立即听到效果：

```bash
python Tutorial/quick_tryout.py
```

该脚本会自动下载所需的依赖文件并播放示例音频。

### 🔗 依赖项下载

对于中国大陆用户，我们强烈建议您手动下载必要的依赖项，并将模型与字典文件放置在根目录Data文件夹下。

| 下载渠道         | 链接                                                                                           |
|:-------------|:---------------------------------------------------------------------------------------------|
| Hugging Face | [https://huggingface.co/Lux-Luna/LunaVox/tree/main](https://huggingface.co/Lux-Luna/LunaVox) |

下载后，请通过环境变量 (os.environ) 指定文件路径。

### 🎤 语音合成最佳实践

下面是一个简单的 TTS 推理示例：

```python
import os

# (可选) 设置 HuBERT 中文模型路径。若不设置，程序将尝试从 Hugging Face 自动下载。
os.environ['HUBERT_MODEL_PATH'] = r"C:\path\to\your\chinese-hubert-base.onnx"

# (可选) 设置 Open JTalk 字典文件夹路径。若不设置，程序将尝试从 Github 自动下载。
os.environ['OPEN_JTALK_DICT_DIR'] = r"C:\path\to\your\open_jtalk_dic_utf_8-1.11"

import lunavox_tts as lunavox

# 步骤 1: 加载角色声音模型
lunavox.load_character(
    character_name='<CHARACTER_NAME>',  # 替换为你的角色名称
    onnx_model_dir=r"<PATH_TO_CHARACTER_ONNX_MODEL_DIR>",  # 替换为包含 ONNX 模型的文件夹路径
)

# 步骤 2: 设置参考音频 (用于情感和语调克隆)
lunavox.set_reference_audio(
    character_name='<CHARACTER_NAME>',  # 确保与加载的角色名称一致
    audio_path=r"<PATH_TO_REFERENCE_AUDIO>",  # 替换为你的参考音频文件路径
    audio_text="<REFERENCE_AUDIO_TEXT>",  # 替换为参考音频对应的文本
)

# 步骤 3: 执行 TTS 推理并生成音频
lunavox.tts(
    character_name='<CHARACTER_NAME>',  # 确保与加载的角色名称一致
    text="<TEXT_TO_SYNTHESIZE>",  # 替换为你想要合成的文本
    play=True,  # 设置为 True 可直接播放生成的音频
    save_path="<OUTPUT_AUDIO_PATH>",  # 替换为期望的音频保存路径
)

print("🎉 音频生成完毕!")
```

## 🔧 模型转换 (Model Conversion)

如果您需要将原始的 GPT-SoVITS 模型转换为 LunaVox 使用的格式，请先确保已安装 `torch`。

```bash
pip install torch
```

然后，您可以使用内置的转换工具。

> **提示:** 目前 `convert_to_onnx` 函数仅支持转换 V2 版本的模型。

```python
import lunavox_tts as lunavox

lunavox.convert_to_onnx(
    torch_pth_path=r"<你的 .pth 模型文件路径>",  # 替换为您的 .pth 模型文件路径
    torch_ckpt_path=r"<你的 .ckpt 检查点文件路径>",  # 替换为您的 .ckpt 检查点文件路径
    output_dir=r"<ONNX 模型输出文件夹路径>"  # 指定 ONNX 模型保存的目录
)
```

## 🌐 启动 FastAPI 服务器

LunaVox 内置了一个简单的 FastAPI 服务器。

```python
import os

os.environ['HUBERT_MODEL_PATH'] = r"C:\path\to\your\chinese-hubert-base.onnx"
os.environ['OPEN_JTALK_DICT_DIR'] = r"C:\path\to\your\open_jtalk_dic_utf_8-1.11"

import lunavox_tts as lunavox

# 启动服务器
lunavox.start_server(
    host="0.0.0.0",  # 监听的主机地址
    port=8000,  # 监听的端口
    workers=1  # 工作进程数
)
```

> 关于服务器的请求格式、接口详情等信息，请参考我们的 [API 服务器使用教程](./Tutorial/English/API%20Server%20Tutorial.py)。

## 🌐 启动 WebUI 界面

LunaVox 提供了一个基于 Gradio 的 Web 界面，让您可以通过浏览器轻松使用 TTS 功能。

### 快速启动

```bash
# Windows 用户
start_webui.bat

# 或直接运行
python WebUI/webui.py
```

### 功能特性

- **🎭 角色管理**: 自动扫描 `Data/character_model` 下的角色模型
- **🎵 参考音频**: 支持上传自定义参考音频或使用预设音频资源
- **📝 文本合成**: 输入日文文本，一键生成语音
- **🎧 在线试听**: 生成的音频可直接在浏览器中播放
- **💾 文件保存**: 所有生成的音频自动保存到 `Output` 目录

### 使用说明

1. 启动 WebUI 后，浏览器会自动打开 `http://127.0.0.1:7860`
2. 选择角色模型（会自动加载）
3. 设置参考音频（上传文件或选择预设音频）
4. 输入要合成的日文文本
5. 点击"开始合成"即可生成并试听音频

## ⌨️ 启动命令行客户端

为了方便快速测试和交互式使用，LunaVox 提供了一个简单的命令行客户端。

```python
import lunavox_tts as lunavox

# 启动命令行客户端
lunavox.launch_command_line_client()
```

## 📝 未来计划 (Roadmap)

- [ ] **🌐 语言扩展**
    - [ ] 增加对 **中文**、**英文** 的支持。

- [ ] **🚀 模型兼容性**
    - [ ] 增加对 `V2Proplus`模型版本的支持。

- [ ] **📦 便捷部署**
    - [ ] 发布 **Docker 镜像**。
    - [ ] 提供开箱即用的 **Windows / Linux 整合包**。

---