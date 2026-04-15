"""Translation strings for the new three-view GUI.

Trimmed from the 200+-key legacy dict to only what the new surface
actually shows. Keys are grouped by view; adding a view means
appending one block here, not editing every existing string.
"""

from __future__ import annotations

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # App chrome
        "app.title": "LunaVox",
        "app.subtitle": "Qwen3-TTS inference",
        "nav.synthesize": "Synthesize",
        "nav.library": "Library",
        "nav.settings": "Settings",
        "lang.toggle": "中文",
        # Synthesize view
        "synth.title": "Synthesize",
        "synth.text_label": "Text",
        "synth.text_placeholder": "Type or paste text to speak…",
        "synth.voice_label": "Voice",
        "synth.voice.base": "Default voice",
        "synth.voice.clone": "Clone from reference",
        "synth.voice.custom": "Catalog speaker",
        "synth.voice.design": "Design from description",
        "synth.reference_label": "Reference (WAV or JSON)",
        "synth.speaker_label": "Speaker id",
        "synth.instruct_label": "Style instruction",
        "synth.params_label": "Advanced parameters",
        "synth.generate": "Generate",
        "synth.generating": "Generating…",
        "synth.play": "Play",
        "synth.save": "Save WAV…",
        "synth.regenerate": "Regenerate",
        "synth.error": "Synthesis failed",
        "synth.browse": "Browse…",
        # Stats card
        "stats.title": "Last run",
        "stats.rtf": "Real-time factor",
        "stats.duration": "Audio duration",
        "stats.total": "Total latency",
        "stats.ram": "Peak RAM",
        "stats.tokenize": "Tokenize",
        "stats.encode": "Encode",
        "stats.generate": "Generate",
        "stats.decode": "Decode",
        # Library view
        "lib.title": "Library",
        "lib.models_section": "Models",
        "lib.references_section": "Reference voices",
        "lib.no_models": "No models installed. Run `lunavox model pull`.",
        "lib.no_references": "Drop .wav or .json files into ref/ to see them here.",
        # Settings view
        "settings.title": "Settings",
        "settings.profile_label": "Active profile",
        "settings.model_label": "Default model",
        "settings.backend_label": "Backend",
        "settings.threads_label": "Threads",
        "settings.config_path_label": "Config file",
        "settings.apply": "Apply",
        "settings.applied": "Applied (session only — edit config.toml to persist)",
    },
    "zh": {
        # App chrome
        "app.title": "LunaVox",
        "app.subtitle": "Qwen3-TTS 推理引擎",
        "nav.synthesize": "合成",
        "nav.library": "素材库",
        "nav.settings": "设置",
        "lang.toggle": "English",
        # Synthesize view
        "synth.title": "语音合成",
        "synth.text_label": "文本",
        "synth.text_placeholder": "在此输入或粘贴要合成的文本…",
        "synth.voice_label": "音色模式",
        "synth.voice.base": "默认音色",
        "synth.voice.clone": "参考克隆",
        "synth.voice.custom": "内置发音人",
        "synth.voice.design": "文本设计",
        "synth.reference_label": "参考文件 (WAV 或 JSON)",
        "synth.speaker_label": "发音人 ID",
        "synth.instruct_label": "风格描述",
        "synth.params_label": "高级参数",
        "synth.generate": "开始合成",
        "synth.generating": "合成中…",
        "synth.play": "播放",
        "synth.save": "保存 WAV…",
        "synth.regenerate": "重新合成",
        "synth.error": "合成失败",
        "synth.browse": "浏览…",
        # Stats card
        "stats.title": "最近一次",
        "stats.rtf": "实时率",
        "stats.duration": "音频时长",
        "stats.total": "总耗时",
        "stats.ram": "内存峰值",
        "stats.tokenize": "分词",
        "stats.encode": "编码",
        "stats.generate": "生成",
        "stats.decode": "解码",
        # Library view
        "lib.title": "素材库",
        "lib.models_section": "已安装模型",
        "lib.references_section": "参考音色",
        "lib.no_models": "未安装任何模型，请先执行 `lunavox model pull`。",
        "lib.no_references": "把 .wav 或 .json 文件放到 ref/ 目录下即可在此显示。",
        # Settings view
        "settings.title": "设置",
        "settings.profile_label": "当前 profile",
        "settings.model_label": "默认模型",
        "settings.backend_label": "运行后端",
        "settings.threads_label": "线程数",
        "settings.config_path_label": "配置文件",
        "settings.apply": "应用",
        "settings.applied": "已应用 (仅本次会话；如需持久化请编辑 config.toml)",
    },
}


class Translator:
    """Tiny lookup helper so callers can swap languages at runtime."""

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang if lang in TRANSLATIONS else "en"

    def set_lang(self, lang: str) -> None:
        if lang in TRANSLATIONS:
            self.lang = lang

    def __call__(self, key: str) -> str:
        return TRANSLATIONS[self.lang].get(key, TRANSLATIONS["en"].get(key, key))
