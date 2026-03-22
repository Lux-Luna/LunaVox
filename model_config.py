"""
LunaVox 多模型配置中心

支持的模型变体:
  - base_small  : Qwen3-TTS-12Hz-0.6B-Base
  - custom_small: Qwen3-TTS-12Hz-0.6B-CustomVoice
  - base        : Qwen3-TTS-12Hz-1.7B-Base
  - custom      : Qwen3-TTS-12Hz-1.7B-CustomVoice
  - design      : Qwen3-TTS-12Hz-1.7B-VoiceDesign

使用方式:
  1. 直接修改本文件最底部的 `model = Models.xxx` 行来切换模型
  2. 或通过命令行传入 --model <name> 参数
"""

from dataclasses import dataclass
from pathlib import Path

# HuggingFace hub 本地缓存根目录
HF_HUB_ROOT = Path(r'C:\Users\kwong\.cache\huggingface\hub')

# 项目根目录
REPO_ROOT = Path(__file__).resolve().parent


def get_snapshot(repo_name: str) -> Path:
    """定位 HuggingFace 缓存中模型的实际快照路径"""
    snap_dir = HF_HUB_ROOT / f'models--Qwen--{repo_name}' / 'snapshots'
    if snap_dir.exists():
        snaps = list(snap_dir.iterdir())
        if snaps:
            return snaps[0]
    return HF_HUB_ROOT / f'models--Qwen--{repo_name}'


@dataclass
class ModelConfig:
    """单个模型变体的路径配置"""
    name: str       # 简称标识 (用于 CLI --model 参数)
    source: Path    # 原始 HF 权重路径
    dest: Path      # 转换产物输出目录


class Models:
    """全部可用模型变体"""
    base = ModelConfig(
        "base",
        get_snapshot('Qwen3-TTS-12Hz-1.7B-Base'),
        REPO_ROOT / 'models' / 'base',
    )
    custom = ModelConfig(
        "custom",
        get_snapshot('Qwen3-TTS-12Hz-1.7B-CustomVoice'),
        REPO_ROOT / 'models' / 'custom',
    )
    design = ModelConfig(
        "design",
        get_snapshot('Qwen3-TTS-12Hz-1.7B-VoiceDesign'),
        REPO_ROOT / 'models' / 'design',
    )
    base_small = ModelConfig(
        "base_small",
        get_snapshot('Qwen3-TTS-12Hz-0.6B-Base'),
        REPO_ROOT / 'models' / 'base_small',
    )
    custom_small = ModelConfig(
        "custom_small",
        get_snapshot('Qwen3-TTS-12Hz-0.6B-CustomVoice'),
        REPO_ROOT / 'models' / 'custom_small',
    )

    ALL = [base, custom, design, base_small, custom_small]

    @classmethod
    def by_name(cls, name: str) -> ModelConfig:
        """按名称查找模型配置，不存在则抛出异常"""
        for m in cls.ALL:
            if m.name == name:
                return m
        valid = ', '.join(m.name for m in cls.ALL)
        raise ValueError(f"Unknown model '{name}'. Available: {valid}")


# ============================================================
# 当前选定模型（直接修改此行或通过 CLI --model 覆盖）
# ============================================================
model = Models.base_small

MODEL_DIR = model.source
EXPORT_DIR = model.dest
