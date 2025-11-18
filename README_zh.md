<div align="center">

# 🔮 LunaVox: [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 轻量级推理引擎

**专为 GPT-SoVITS 设计的高性能、轻量级的推理引擎**

[简体中文](./README_zh.md) | [English](./README.md)

</div>

---

**LunaVox** 是基于开源 TTS 项目 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 打造的轻量级推理引擎，集成了
TTS 推理、ONNX 模型转换、API Server 等核心功能，旨在提供更极致的性能与更便捷的体验。

- **✅ 支持模型版本:** GPT-SoVITS V2， GPT-SoVITS V2 pro plus
- **✅ 支持语言:** 日语 (Japanese)，中文（Chinese），英语（English）

LunaVox 继承 GPT-SoVITS 的核心推理链路：文本经多语言前端（如 Open JTalk）转为音素 → HuBERT 提取参考音频特征 → 三段式
T2S（Encoder / First-Stage Decoder / Stage Decoder）生成语音 Token → VITS 声码器合成最终波形。仓库内将上述模块（含中文
HuBERT 与说话人向量模型）拆分为 ONNX 形式，结合缓存机制实现纯 ONNXRuntime 的快速推理。

---

## 🏁 快速开始 (QuickStart)

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

所有示例脚本都位于 `Tutorial/` 目录，并会在启动时调用 `Tutorial/data_setup.py` 自动补齐缺失的模型/词典，同时设置好 `HUBERT_MODEL_PATH` 与 `OPEN_JTALK_DICT_DIR`。在仓库根目录执行以下命令即可：

#### GPT-SoVITS v2 预设（无需说话人向量）

```bash
python Tutorial/v2_quick_tryout/quick_tryout_en.py  # 英文演示
python Tutorial/v2_quick_tryout/quick_tryout_zh.py  # 中文演示
python Tutorial/v2_quick_tryout/quick_tryout_ja.py  # 日文演示
```

#### GPT-SoVITS v2 Pro Plus 预设（需要说话人向量）

```bash
python Tutorial/v2_pro_plus_quick_tryout/quick_tryout_v2proplus_en.py
python Tutorial/v2_pro_plus_quick_tryout/quick_tryout_v2proplus_zh.py
python Tutorial/v2_pro_plus_quick_tryout/quick_tryout_v2proplus_ja.py
```

> 运行 v2 Pro Plus 脚本前，请先按文档将 ERes2NetV2 导出为 `Data/sv/eres2netv2.onnx`，否则无法生成必需的说话人嵌入。

### 🔗 依赖项下载

对于中国大陆用户，我们强烈建议您手动下载必要的依赖项，并将模型与字典文件放置在根目录Data文件夹下。

| 下载渠道         | 链接                                                                                           |
|:-------------|:---------------------------------------------------------------------------------------------|
| Hugging Face | [https://huggingface.co/Lux-Luna/LunaVox/tree/main](https://huggingface.co/Lux-Luna/LunaVox) |

下载后，请通过环境变量 (os.environ) 指定文件路径。

### 🧩 可选依赖

- **中文语义/说话人特征**：执行 `pip install "lunavox-tts[zh]"`，会自动安装 `torch` 与 `transformers`。未安装时，中文路径会退化为零向量，但日语/英语推理不受影响。
- **模型转换 (`convert_to_onnx`)**：执行 `pip install "lunavox-tts[convert]"` 以启用 PyTorch 转换脚本。

### 🎤 语音合成最佳实践

多语言 TTS 推理示例：

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
    audio_language='ja'  # ja 表示日语，zh 表示中文，en 表示英语
)

# 步骤 3: 执行 TTS 推理并生成音频
lunavox.tts(
    character_name='<CHARACTER_NAME>',  # 确保与加载的角色名称一致
    text="<TEXT_TO_SYNTHESIZE>",  # 替换为你想要合成的文本
    play=True,  # 设置为 True 可直接播放生成的音频
    save_path="<OUTPUT_AUDIO_PATH>",  # 替换为期望的音频保存路径
    language='ja'  # 目标语言：ja 表示日语，zh 表示中文，en 表示英语
)

print("🎉 音频生成完毕!")
```

## 📊 性能基准（Intel Core i9-12900K）

以下数据源自 `benchmark/scripts/tts_benchmark.py` 在 Windows 11、Python 3.12、32 GB 内存与 Intel Core i9-12900K
环境下对预设角色执行的 3 次预热 + 100 次循环测试（固定文本 “This is LunaVox speaking English.”）。

| 模型版本 | 模型大小 (MB) | 首包延迟 (s) | 全句延迟 (s) | 吞吐 (次/s) | 加载后 RSS 增量 (MB) |
|---|---|---|---|---|---|
| v2 | 683.54 | 1.15 | 1.15 | 0.96 | 2151.46 |
| v2_pro_plus | 1256.14 | 1.38 | 1.38 | 0.76 | 2917.04 |

- 两个模型的实时因子均约 0.54，能够持续快于实时地产生音频。
- `benchmark/results/v2_results.json` 与 `benchmark/results/v2_pro_plus_results.json` 保存了完整指标与每轮数据。

## 🔧 模型转换 (Model Conversion)

如果您需要将原始的 GPT-SoVITS 模型转换为 LunaVox 使用的格式，请先确保已安装 `torch`。

```bash
pip install "lunavox-tts[convert]"
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

转换工具会将 GPT-SoVITS 的推理链路拆解为多份 ONNX：`t2s_encoder_fp32.onnx`、`t2s_first_stage_decoder_fp32.onnx`、
`t2s_stage_decoder_fp32.onnx` 与 `vits_fp32.onnx`，并保留中文 HuBERT、说话人向量模型等配套依赖。转换过程中默认把
原始 FP16 权重临时升为 FP32，以确保 onnxruntime 在 CPU 环境下具备稳定数值表现。

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

## ⚙️ 运行时配置

- `LUNAVOX_ORT_PROVIDERS`：覆盖 ONNX Runtime Provider 顺序（逗号分隔），示例 `CUDAExecutionProvider,CPUExecutionProvider`。
- `LUNAVOX_USE_IO_BINDING=1`：启用实验性的 IO Binding，可在支持的 GPU Provider 下减少 Host/Device 拷贝。

## 📝 未来计划 (Roadmap)

- [x] **🌐 语言扩展**
    - [x] 增加对 **中文** 的支持。
    - [x] 增加对 **英文** 的支持。

- [x] **🚀 模型兼容性**
    - [x] 增加对 **V2 Pro** 模型版本的支持。
    - [x] 增加对 **V2 Pro Plus** 模型版本的支持。

- [ ] **⚡️ 性能优化**
    - [ ] 发布 **GPU 版本**，进一步提升推理速度。
    - [ ] 实现 **文本切分功能**，优化长文本处理。

- [ ] **📦 便捷部署**
    - [ ] 发布 **Docker 镜像**。
    - [ ] 提供开箱即用的 **Windows / Linux 整合包**。

---