## LunaVox 本地 WebUI 设计与使用说明

本页面介绍基于项目内置的 `lunavox_tts` 推理接口实现的本地 WebUI。该 WebUI 无需额外 API 服务，直接在本地运行，支持：

- 自动扫描 `Data/character_model` 下的角色目录（每个文件夹代表一个角色）
- 启动时默认加载按名称排序最前的角色（若不存在角色则提示）
- 上传参考音频与对应文本，上传后可直接在页面中试听参考音频
- 输入日文文本，一键合成，将结果保存至 `Output` 目录，同时可在页面中试听

### 一、目录结构与依赖

- 角色模型目录：`Data/character_model/<角色名>/`
  - 需包含以下 ONNX 与权重文件（GPT-SoVITS V2）：
    - `t2s_encoder_fp32.onnx`
    - `t2s_first_stage_decoder_fp32.onnx`
    - `t2s_stage_decoder_fp32.onnx`
    - `t2s_shared_fp16.bin`（程序首次运行会自动转换生成 `t2s_shared_fp32.bin`）
    - `vits_fp32.onnx`
    - `vits_fp16.bin`（程序首次运行会自动转换生成 `vits_fp32.bin`）
  - 可选默认参考音频与文本：
    - `prompt.wav`
    - `prompt_wav.json`（从中读取 `Normal.text` 作为默认参考文本）

- 依赖模型与字典（建议本地放置以避免网络下载）：
  - `Data/chinese-hubert-base.onnx`
  - `Data/open_jtalk_dic_utf_8-1.11/`（Open JTalk 字典目录）

WebUI 启动时会自动设置如下环境变量，优先使用本地文件：

- `HUBERT_MODEL_PATH = Data/chinese-hubert-base.onnx`
- `OPEN_JTALK_DICT_DIR = Data/open_jtalk_dic_utf_8-1.11`

### 二、启动方式

Windows 用户可直接双击根目录的 `start_webui.bat`，脚本将：

1. 首次运行自动安装依赖（读取根目录 `requirements.txt`），后续启动默认跳过安装，提升启动速度
2. 始终优先从本地资源加载：
   - `Data/chinese-hubert-base.onnx`
   - `Data/open_jtalk_dic_utf_8-1.11/`
   - `Data/character_model/<角色名>/...`
   - 启动时自动设置 `HF_HUB_OFFLINE=1`，避免联网下载
3. 启动 WebUI（Gradio 会自动打开浏览器至 `http://127.0.0.1:7860`）

如自动打开失败，可手动访问上述地址。

> 注意：`pyopenjtalk` 与 `pyaudio` 在 Windows 下可能需要安装 Visual Studio C++ 构建工具。详见仓库 `README_zh.md` 的说明。

> 如需强制重新安装依赖（例如更换 Python 版本或遇到损坏的环境），可以以参数方式启动：
>
> ```bash
> start_webui.bat --reinstall
> ```

### 三、页面功能说明

- **角色选择**：从 `Data/character_model` 自动抓取角色目录并排序展示。切换角色时会加载相应模型。
- **使用默认参考音频**：若角色目录内存在 `prompt.wav` 与 `prompt_wav.json`，可一键加载为参考音频与文本。
- **上传参考音频与文本**：
  - 支持的音频格式：`.wav`, `.flac`, `.ogg`, `.aiff`, `.aif`
  - 上传后可直接在页面中播放试听
  - 点击“设置参考音频”后，将作为后续合成的参考
- **文本合成**：输入日文文本，点击“开始合成”，输出音频将保存至 `Output/<角色名>_YYYYMMDD_HHMMSS.wav`，并可在页面中播放。

### 四、实现要点

- 通过 `lunavox_tts.load_character(character_name, model_dir)` 加载角色模型
- 通过 `lunavox_tts.set_reference_audio(character_name, audio_path, audio_text)` 设置参考
- 通过 `lunavox_tts.tts(character_name=..., text=..., save_path=...)` 合成音频
- WebUI 基于 `gradio` 实现，位于仓库根目录 `webui.py`
- Windows 启动脚本位于仓库根目录 `start_webui.bat`

### 五、常见问题

- 启动后未发现角色：请检查 `Data/character_model` 下是否存在角色文件夹，且文件夹内包含所需 ONNX/权重文件。
- 合成失败：请查看控制台日志，确认参考音频格式是否受支持、依赖是否安装完整、以及字典/模型路径是否正确。
- 浏览器未自动打开：可直接访问 `http://127.0.0.1:7860`。


