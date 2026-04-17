"""Curated style-instruction presets for the VoicePicker.

Two flavours:

- :data:`CUSTOM_PRESETS` — short tone modifiers for catalog-speaker
  (custom) mode. The speaker identity comes from the speaker id
  dropdown; the instruction only nudges emotion.
- :data:`DESIGN_PRESETS` — full voice descriptions for design mode, where
  the instruction alone defines the voice.

Each list is a sequence of ``(label, instruct)`` tuples. The first
entry's empty instruct means "free form / don't override whatever the
user typed".
"""

from __future__ import annotations

PresetList = list[tuple[str, str]]

CUSTOM_PRESETS: dict[str, PresetList] = {
    "en": [
        ("(free form)", ""),
        ("Angry", "Use angry tone."),
        ("Cheerful", "Speak in a cheerful, upbeat tone."),
        ("Calm", "Speak in a calm, soothing tone."),
        ("Excited", "Speak in an excited, energetic tone."),
        ("Whisper", "Speak softly, almost in a whisper."),
        ("Incredulous", "Speak in an incredulous tone."),
        ("Sad", "Speak in a sad, subdued tone."),
    ],
    "zh": [
        ("(自定义)", ""),
        ("愤怒", "用愤怒的语气说话。"),
        ("欢快", "用欢快、轻松的语气说话。"),
        ("平静", "用平静、舒缓的语气说话。"),
        ("兴奋", "用兴奋、充满活力的语气说话。"),
        ("耳语", "用轻柔、几近耳语的方式说话。"),
        ("难以置信", "用难以置信的语气说话。"),
        ("悲伤", "用低沉、悲伤的语气说话。"),
    ],
}

DESIGN_PRESETS: dict[str, PresetList] = {
    "en": [
        ("(free form)", ""),
        (
            "Warm female",
            "A warm female voice, speaking gently with a hint of a smile.",
        ),
        (
            "Authoritative male",
            "A deep, authoritative male voice speaking with calm confidence.",
        ),
        (
            "Bright young",
            "A bright, youthful voice speaking with enthusiasm and clarity.",
        ),
        (
            "Narrator",
            "A measured documentary-style narrator, articulate and unhurried.",
        ),
        (
            "Incredulous",
            "Speak in an incredulous tone, as if questioning what was just said.",
        ),
        (
            "Gentle whisper",
            "A soft, close-miked whisper, intimate and slightly breathy.",
        ),
    ],
    "zh": [
        ("(自定义)", ""),
        ("温暖女声", "温暖的女声，语气轻柔，带着一丝笑意。"),
        ("权威男声", "低沉而有权威感的男声，语气沉稳自信。"),
        ("明亮少年", "明亮的年轻声音，充满热情、吐字清晰。"),
        ("纪录片旁白", "纪录片风格的旁白，语速平稳、吐字清楚。"),
        ("质疑语气", "用难以置信的语气说话，仿佛在质疑刚刚听到的话。"),
        ("轻柔耳语", "贴近麦克风的轻声耳语，亲密而略带气声。"),
    ],
}


def custom_presets(lang: str) -> PresetList:
    return CUSTOM_PRESETS.get(lang, CUSTOM_PRESETS["en"])


def design_presets(lang: str) -> PresetList:
    return DESIGN_PRESETS.get(lang, DESIGN_PRESETS["en"])
