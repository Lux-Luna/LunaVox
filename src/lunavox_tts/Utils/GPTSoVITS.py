import os
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Optional


@lru_cache(maxsize=1)
def find_repo_root() -> Optional[Path]:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "GPT-SoVITS"
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def find_text_root() -> Optional[Path]:
    repo_root = find_repo_root()
    if not repo_root:
        return None
    text_root = repo_root / "GPT_SoVITS"
    if text_root.exists():
        return text_root
    return None


def ensure_text_on_path() -> Optional[Path]:
    text_root = find_text_root()
    if text_root and str(text_root) not in sys.path:
        sys.path.insert(0, str(text_root))
    return text_root


def ensure_default_bert_env() -> Optional[Path]:
    current = os.environ.get("bert_path")
    if current and os.path.exists(current):
        return Path(current)
    repo_root = find_repo_root()
    if not repo_root:
        return None
    default_path = repo_root / "pretrained_models" / "chinese-roberta-wwm-ext-large"
    if default_path.exists():
        os.environ.setdefault("bert_path", str(default_path))
        return default_path
    return None


@contextmanager
def use_repo_cwd():
    repo_root = find_repo_root()
    if not repo_root:
        yield
        return
    previous = Path.cwd()
    try:
        os.chdir(str(repo_root))
        yield
    finally:
        os.chdir(str(previous))
