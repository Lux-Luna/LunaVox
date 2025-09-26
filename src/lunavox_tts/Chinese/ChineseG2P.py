from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from ..Japanese.SymbolsV2 import symbols_v2  # noqa: F401  # Ensure symbols_v2 is initialized
from ..Utils.GPTSoVITS import ensure_default_bert_env, ensure_text_on_path, find_repo_root, use_repo_cwd

ensure_text_on_path()
ensure_default_bert_env()


@lru_cache(maxsize=1)
def _load_cleaner():
    with use_repo_cwd():
        from text.g2pw import onnx_api  # type: ignore

        original_download = onnx_api.download_and_decompress

        def _download_and_decompress(model_dir: str = "G2PWModel/") -> str:
            # Prefer LunaVox local text path if available
            this_file = Path(__file__).resolve()
            src_root = this_file.parents[2]  # .../src
            local_text = src_root / "text"
            local_g2pw = local_text / "G2PWModel"
            if local_g2pw.exists():
                return str(local_g2pw)

            # Fallback: resolve under GPT-SoVITS repo
            candidate = Path(model_dir)
            if not candidate.is_absolute():
                repo_root = find_repo_root()
                if repo_root:
                    candidate = (repo_root / model_dir).resolve()
            return original_download(str(candidate))

        onnx_api.download_and_decompress = _download_and_decompress  # type: ignore

        from text.cleaner import clean_text as _clean_text  # type: ignore
        from text import cleaned_text_to_sequence as _cleaned_text_to_sequence  # type: ignore
    return _clean_text, _cleaned_text_to_sequence


def _run_cleaner(text: str):
    clean_text_fn, sequence_fn = _load_cleaner()
    with use_repo_cwd():
        phones, word2ph, norm_text = clean_text_fn(text, "zh", "v2")
    word2ph = list(map(int, word2ph or []))
    ids = list(map(int, sequence_fn(phones, "v2")))
    return phones, ids, word2ph, norm_text


def chinese_clean_and_g2p(text: str) -> Tuple[List[str], List[int], str]:
    phones, ids, _, norm_text = _run_cleaner(text)
    return phones, ids, norm_text


def chinese_to_phones_and_word2ph(text: str) -> Tuple[List[int], List[int]]:
    _, ids, word2ph, _ = _run_cleaner(text)
    return ids, word2ph


def chinese_clean_g2p_and_norm(text: str) -> Tuple[List[int], List[int], str]:
    _, ids, word2ph, norm_text = _run_cleaner(text)
    return ids, word2ph, norm_text


