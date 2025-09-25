from __future__ import annotations

from ..Utils.GPTSoVITS import ensure_text_on_path, use_repo_cwd

_text_root = ensure_text_on_path()
if not _text_root:
    raise ImportError("Unable to locate GPT-SoVITS text modules. Please place GPT-SoVITS alongside LunaVox.")

with use_repo_cwd():
    from text import symbols2 as _gpt_symbols2  # type: ignore

symbols_v2: list[str] = list(_gpt_symbols2.symbols)
symbol_to_id_v2: dict[str, int] = {symbol: idx for idx, symbol in enumerate(symbols_v2)}

