from __future__ import annotations

# Use vendored exact replica of GPT-SoVITS symbols to avoid external dependency
from ..Symbols.symbols2_exact import symbols_v2_exact as symbols_v2  # type: ignore

symbol_to_id_v2: dict[str, int] = {symbol: idx for idx, symbol in enumerate(symbols_v2)}

