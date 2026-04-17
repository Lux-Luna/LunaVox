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
        "synth.model_label": "Model",
        "synth.no_models": "No models installed. Open Library → Pull to install one.",
        "synth.reference_label": "Reference (WAV or JSON)",
        "synth.ref_preset_label": "From ref/ folder",
        "synth.ref_custom": "(custom / browse)",
        "synth.advanced_expand": "Advanced parameters  ▸",
        "synth.advanced_collapse": "Advanced parameters  ▾",
        "synth.speaker_label": "Speaker id",
        "synth.instruct_label": "Style instruction",
        "synth.preset_label": "Style preset",
        "synth.params_label": "Advanced parameters",
        "synth.generate": "Generate",
        "synth.generating": "Generating…",
        "synth.play": "Play",
        "synth.save": "Save WAV…",
        "synth.regenerate": "Regenerate",
        "synth.error": "Synthesis failed",
        "synth.browse": "Browse…",
        "synth.persistent_label": "Pre-load",
        "synth.persistent_loading": "Loading model…",
        "synth.persistent_loaded": "Model loaded",
        "synth.persistent_load_failed": "Model load failed",
        # Stats card
        "stats.title": "Last run",
        "stats.rtf": "Real-time factor",
        "stats.ttfb": "TTFB",
        "stats.duration": "Audio duration",
        "stats.total": "Total latency",
        "stats.tokenize": "Tokenize",
        "stats.encode": "Encode",
        "stats.generate": "Generate",
        "stats.decode": "Decode",
        # Library view
        "lib.title": "Library",
        "lib.models_section": "Models",
        "lib.references_section": "Reference voices",
        "lib.runtimes_section": "Native runtimes (lib/)",
        "lib.runtime_installed": "{label} · {version}",
        "lib.runtime_missing": "Not installed",
        "lib.runtime_drop_hint": (
            "Drop a backend archive onto a row to install it, or click Download "
            "to fetch a prebuilt release. For other versions or custom builds, "
            "place the extracted llama.cpp / onnxruntime tree under lib/llama "
            "or lib/onnx."
        ),
        "lib.runtime_download": "Download…",
        "lib.runtime_picker_title": "Choose backend",
        "lib.runtime_install_title": "Installing {lib} ({backend})",
        "lib.runtime_install_drop": "Installing dropped archive into lib/{lib}…",
        "lib.runtime_install_done": "Installed {lib} ({backend}).",
        "lib.runtime_install_failed": "Install failed: {err}",
        "lib.no_models": "No models installed. Run `lunavox model pull`.",
        "lib.no_references": "Drop .wav or .json files into ref/ to see them here.",
        "lib.pull_btn": "Pull",
        "lib.pulling_title": "Pulling {name}",
        "lib.task_pull_start": "Pulling {name} from HuggingFace…",
        "lib.task_pull_plan": "Expecting {count} files in {name}/.",
        "lib.task_pull_progress_counted": "… {files}/{total} files · {size} · elapsed {elapsed}",
        "lib.task_pull_progress_simple": "… {files} files · {size} · elapsed {elapsed}",
        "lib.task_done": "Done.",
        "lib.task_failed": "Failed: {err}",
        "lib.task_cancelled": "Cancelled by user.",
        "lib.dialog_cancel": "Cancel",
        "lib.dialog_close": "Close",
        # Settings view
        "settings.title": "Settings",
        "settings.threads_label": "Threads",
        "settings.runtimes_label": "Active runtimes",
        "settings.runtime_llama": "llama.cpp backend",
        "settings.runtime_onnx": "ONNX Runtime provider",
        "settings.runtime_unknown": "(not installed — open Library to download)",
        "settings.apply": "Apply",
        "settings.applied": "Applied (session only).",
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
        "synth.model_label": "模型",
        "synth.no_models": "尚未安装任何模型，请前往「素材库 → 下载」。",
        "synth.reference_label": "参考文件 (WAV 或 JSON)",
        "synth.ref_preset_label": "ref/ 中的预设",
        "synth.ref_custom": "（自定义 / 浏览）",
        "synth.advanced_expand": "高级参数  ▸",
        "synth.advanced_collapse": "高级参数  ▾",
        "synth.speaker_label": "发音人 ID",
        "synth.instruct_label": "风格描述",
        "synth.preset_label": "风格预设",
        "synth.params_label": "高级参数",
        "synth.generate": "开始合成",
        "synth.generating": "合成中…",
        "synth.play": "播放",
        "synth.save": "保存 WAV…",
        "synth.regenerate": "重新合成",
        "synth.error": "合成失败",
        "synth.browse": "浏览…",
        "synth.persistent_label": "预加载",
        "synth.persistent_loading": "正在加载模型…",
        "synth.persistent_loaded": "模型已加载",
        "synth.persistent_load_failed": "模型加载失败",
        # Stats card
        "stats.title": "最近一次",
        "stats.rtf": "实时率",
        "stats.ttfb": "首音延迟",
        "stats.duration": "音频时长",
        "stats.total": "总耗时",
        "stats.tokenize": "分词",
        "stats.encode": "编码",
        "stats.generate": "生成",
        "stats.decode": "解码",
        # Library view
        "lib.title": "素材库",
        "lib.models_section": "已安装模型",
        "lib.references_section": "参考音色",
        "lib.runtimes_section": "原生运行库 (lib/)",
        "lib.runtime_installed": "{label} · {version}",
        "lib.runtime_missing": "未安装",
        "lib.runtime_drop_hint": (
            "可将后端压缩包拖拽到对应行进行安装，或点击「下载」获取预编译包。"
            "如需其他版本或自定义构建，请将解压后的 llama.cpp / onnxruntime 目录"
            "放入 lib/llama 或 lib/onnx。"
        ),
        "lib.runtime_download": "下载…",
        "lib.runtime_picker_title": "选择后端",
        "lib.runtime_install_title": "正在安装 {lib} ({backend})",
        "lib.runtime_install_drop": "正在将拖入的压缩包安装到 lib/{lib}…",
        "lib.runtime_install_done": "{lib} ({backend}) 安装完成。",
        "lib.runtime_install_failed": "安装失败：{err}",
        "lib.no_models": "未安装任何模型，请先执行 `lunavox model pull`。",
        "lib.no_references": "把 .wav 或 .json 文件放到 ref/ 目录下即可在此显示。",
        "lib.pull_btn": "下载",
        "lib.pulling_title": "正在下载 {name}",
        "lib.task_pull_start": "正在从 HuggingFace 下载 {name}…",
        "lib.task_pull_plan": "预计需要下载 {count} 个文件到 {name}/。",
        "lib.task_pull_progress_counted": "… 已下载 {files}/{total} 个文件 · {size} · 用时 {elapsed}",
        "lib.task_pull_progress_simple": "… 已下载 {files} 个文件 · {size} · 用时 {elapsed}",
        "lib.task_done": "完成。",
        "lib.task_failed": "失败：{err}",
        "lib.task_cancelled": "已取消。",
        "lib.dialog_cancel": "取消",
        "lib.dialog_close": "关闭",
        # Settings view
        "settings.title": "设置",
        "settings.threads_label": "线程数",
        "settings.runtimes_label": "当前运行库",
        "settings.runtime_llama": "llama.cpp 后端",
        "settings.runtime_onnx": "ONNX Runtime 提供方",
        "settings.runtime_unknown": "（未安装 — 请到「素材库」下载）",
        "settings.apply": "应用",
        "settings.applied": "已应用（仅本次会话）。",
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
