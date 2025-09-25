import os
import sys
import types
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ..Utils.GPTSoVITS import ensure_default_bert_env, find_repo_root

if "torchvision" not in sys.modules:
    tv_stub = types.ModuleType("torchvision")
    tv_stub.__all__ = []
    tv_transforms = types.ModuleType("torchvision.transforms")
    tv_transforms.InterpolationMode = object
    sys.modules["torchvision"] = tv_stub
    sys.modules["torchvision.transforms"] = tv_transforms

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("DISABLE_TRANSFORMERS_IMAGE_TRANSFORMS", "1")

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForMaskedLM] = None


def _resolve_bert_base_path() -> Optional[str]:
    env_path = os.getenv("ZH_BERT_BASE_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    repo_root = Path(__file__).resolve().parents[3]
    local_root = repo_root / "Data" / "chinese-roberta-wwm-ext-large"
    candidate = _locate_snapshot(local_root)
    if candidate:
        return str(candidate)

    gpt_root = find_repo_root()
    if gpt_root:
        candidate = _locate_snapshot(gpt_root / "pretrained_models" / "chinese-roberta-wwm-ext-large")
        if candidate:
            return str(candidate)

    return None


def _locate_snapshot(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    if (root / "config.json").exists():
        return root
    snapshots_dir = root / "snapshots"
    if snapshots_dir.exists():
        for child in sorted(snapshots_dir.iterdir()):
            if (child / "config.json").exists():
                return child
    for child in root.glob("models--*--*"):
        snapshots = child / "snapshots"
        if snapshots.exists():
            for snap in sorted(snapshots.iterdir()):
                if (snap / "config.json").exists():
                    return snap
    return None


def _load_model() -> None:
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return

    ensure_default_bert_env()
    base_path = _resolve_bert_base_path()

    if base_path:
        _tokenizer = AutoTokenizer.from_pretrained(base_path)
        _model = AutoModelForMaskedLM.from_pretrained(base_path)
    else:
        repo_root = Path(__file__).resolve().parents[3]
        base_dir = repo_root / "Data" / "chinese-roberta-wwm-ext-large"
        base_dir.mkdir(parents=True, exist_ok=True)
        model_id = "hfl/chinese-roberta-wwm-ext-large"
        _tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(base_dir))
        _model = AutoModelForMaskedLM.from_pretrained(model_id, cache_dir=str(base_dir))
        _tokenizer.save_pretrained(str(base_dir))
        _model.save_pretrained(str(base_dir))

    _model.eval()


def compute_bert_phone_features(norm_text: str, word2ph: List[int]) -> np.ndarray:
    if not norm_text:
        return np.zeros((sum(word2ph), 1024), dtype=np.float32)
    if len(word2ph) != len(norm_text):
        return np.zeros((sum(word2ph), 1024), dtype=np.float32)

    try:
        _load_model()
    except Exception:
        return np.zeros((sum(word2ph), 1024), dtype=np.float32)

    assert _tokenizer is not None and _model is not None

    device = torch.device("cpu")
    with torch.no_grad():
        inputs = _tokenizer(norm_text, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        outputs = _model(**inputs, output_hidden_states=True)
        hidden = torch.cat(outputs["hidden_states"][-3:-2], dim=-1)[0].cpu()[1:-1]

    phone_features = []
    for idx, repeat in enumerate(word2ph):
        if repeat <= 0:
            continue
        phone_features.append(hidden[idx].repeat(repeat, 1))

    if not phone_features:
        return np.zeros((0, 1024), dtype=np.float32)

    stacked = torch.cat(phone_features, dim=0)
    return stacked.numpy().astype(np.float32)

