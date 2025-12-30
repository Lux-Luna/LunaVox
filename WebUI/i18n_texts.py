SUPPORTED_LANGUAGES = [("English", "en"), ("中文", "zh")]


GUIDE_TEXTS = {
    "en": """
### LunaVox WebUI Guide

- Select Model Version: choose `v2` or `v2 Pro Plus`.
- Pick a Character: select from scanned folders, then click "Load Character".
- Reference Audio (optional but recommended):
  - Choose from preset audio under `CharacterData/audio_resources/<character>/` or upload your own `.wav`. (MP3 is NOT supported)
  - Provide the matching transcript and set the prompt language.
- Synthesis:
  - Choose Output Language (ja/en/zh), enter text, click "Synthesize".
  - Output is saved under `Output/` and previewed below.

Tips:
- Only .wav format is supported for reference audio.
- Keep reference audio clean and 2–10 seconds for best results.
- If file permissions block access, the app copies audio to a temp folder.
""",
    "zh": """
### LunaVox 使用指引

- 选择模型版本：`v2` 或 `v2 Pro Plus`。
- 选择角色：从扫描到的文件夹中选择角色，然后点击“加载角色”。
- 参考音频（可选但推荐）：
  - 从 `CharacterData/audio_resources/<character>/` 选择预设 `.wav`，或上传自己的 `.wav`。（不支持 MP3 格式）
  - 填写与音频匹配的文本，并设置提示语言。
- 文本合成：
  - 选择输出语言（ja/en/zh），输入文本，点击“开始合成”。
  - 生成的音频保存至 `Output/`，并可在界面中试听。

提示：
- 参考音频仅支持 .wav 格式。
- 建议使用干净的 2–10 秒参考音频以获得更好效果。
- 若出现权限问题，程序会将音频复制到临时目录后再使用。
""",
}


CONVERTER_GUIDE_TEXTS = {
    "en": """
### Model Converter Guide

1) Select `.ckpt` (GPT/T2S) and `.pth` (VITS).
2) Choose an output directory; click "Convert".
3) Converted ONNX files will be written under the specified folder.

Notes:
- `v2` and `v2 Pro Plus` use the same conversion API here.
- Ensure enough disk space and avoid network interruptions.
""",
    "zh": """
### 模型转换指引

1) 选择 `.ckpt`（GPT/T2S）与 `.pth`（VITS）。
2) 指定输出目录并点击“开始转换”。
3) 转换后的 ONNX 文件将保存到该目录下。

说明：
- 此处 `v2` 与 `v2 Pro Plus` 使用相同的转换接口。
- 请确保磁盘空间充足，避免网络中断。
""",
}


def display_to_code(display: str) -> str:
    mapping = {d: c for d, c in SUPPORTED_LANGUAGES}
    return mapping.get((display or "").strip(), "en")


def code_to_display(code: str) -> str:
    mapping = {c: d for d, c in SUPPORTED_LANGUAGES}
    return mapping.get((code or "").strip(), "English")


def get_supported_language_display():
    return [d for d, _ in SUPPORTED_LANGUAGES]


def get_guide_markdown(lang_code: str) -> str:
    return GUIDE_TEXTS.get((lang_code or "").strip(), GUIDE_TEXTS["en"]).strip()


def get_converter_help(lang_code: str) -> str:
    return CONVERTER_GUIDE_TEXTS.get((lang_code or "").strip(), CONVERTER_GUIDE_TEXTS["en"]).strip()


# ------------------------------
# UI string dictionary (labels/placeholders) for i18n
# ------------------------------

UI_STRINGS = {
    "en": {
        "webui": {
            "version_label": "Model Version",
            "character_label": "Character",
            "character_info": "Select a character to load",
            "btn_load": "Load Character",
            "status_ready": "Ready.",
            "ref_section_title": "### Reference Audio",
            "preset_ref_label": "Preset Reference Audio (WAV only)",
            "preset_ref_info": "Choose from CharacterData/audio_resources",
            "ref_lang_label": "Prompt Language",
            "or": "**or**",
            "upload_ref_label": "Upload Reference Audio (WAV only)",
            "auto_filename_label": "Auto use filename as transcript",
            "auto_filename_info": "When uploading, use the filename (without extension) as transcript",
            "ref_text_label": "Reference transcript",
            "ref_text_placeholder": "Enter transcript matching the reference audio",
            "synth_section_title": "### Synthesis",
            "output_lang_label": "Output Language",
            "input_text_label": "Input text",
            "input_text_placeholder": "Enter text to synthesize (ja/en/zh)",
            "btn_tts": "Synthesize",
            "out_audio_label": "Preview",
        },
        "converter": {
            "version_label": "Model Version",
            "in_ckpt_label": ".ckpt (GPT/T2S)",
            "in_pth_label": ".pth (VITS)",
            "out_dir_label": "Output Directory",
            "btn_convert": "Convert",
            "ready": "Ready.",
        },
        "lang_names": {"zh": "Chinese", "en": "English", "ja": "Japanese"},
    },
    "zh": {
        "webui": {
            "version_label": "模型版本",
            "character_label": "角色选择",
            "character_info": "请选择一个角色进行加载",
            "btn_load": "加载角色",
            "status_ready": "准备就绪。",
            "ref_section_title": "### 参考音频",
            "preset_ref_label": "预设参考音频（仅支持 WAV）",
            "preset_ref_info": "从 CharacterData/audio_resources 中选择预设参考音频",
            "ref_lang_label": "参考音频语言",
            "or": "**或**",
            "upload_ref_label": "上传参考音频（仅支持 WAV）",
            "auto_filename_label": "自动使用文件名作为参考文本",
            "auto_filename_info": "上传音频时自动将文件名（去除后缀）作为文本",
            "ref_text_label": "参考音频文本",
            "ref_text_placeholder": "请输入与参考音频匹配的文本",
            "synth_section_title": "### 文本合成",
            "output_lang_label": "输出语言",
            "input_text_label": "输入文本",
            "input_text_placeholder": "请输入要合成的文本（ja/en/zh）",
            "btn_tts": "开始合成",
            "out_audio_label": "合成结果试听",
        },
        "converter": {
            "version_label": "模型版本",
            "in_ckpt_label": ".ckpt（GPT/T2S）",
            "in_pth_label": ".pth（VITS）",
            "out_dir_label": "输出目录",
            "btn_convert": "开始转换",
            "ready": "准备就绪。",
        },
        "lang_names": {"zh": "中文", "en": "英语", "ja": "日语"},
    },
}


def ui_text(lang_code: str, section: str, key: str) -> str:
    lang = (lang_code or "en").strip()
    return UI_STRINGS.get(lang, UI_STRINGS["en"]).get(section, {}).get(key, UI_STRINGS["en"][section].get(key, ""))


def get_prompt_language_choices(app_lang_code: str):
    # Order for prompt language: zh, en, ja
    names = UI_STRINGS.get((app_lang_code or "en").strip(), UI_STRINGS["en"])["lang_names"]
    return [names["zh"], names["en"], names["ja"]]


def get_output_language_choices(app_lang_code: str):
    # Order for output language: ja, en, zh (to match existing UX)
    names = UI_STRINGS.get((app_lang_code or "en").strip(), UI_STRINGS["en"])["lang_names"]
    return [names["ja"], names["en"], names["zh"]]


def display_language_name_for(code: str, app_lang_code: str) -> str:
    names = UI_STRINGS.get((app_lang_code or "en").strip(), UI_STRINGS["en"])["lang_names"]
    return names.get(code, code)


