"""Punctuation tables driving multi-language sentence splitting.

The splitting algorithms in this package are **data-driven** — the
only place that knows about specific languages or scripts is this
file. The splitter and streaming buffer just consume these tables
and the regexes built from them, so adding support for a new
writing system means appending characters here, not editing logic.

Two terminator tiers, two trailing-context modes per tier:

* **Strong** terminators end a sentence: ``.`` ``!`` ``?`` and
  language-specific equivalents (``。`` ``！`` ``？`` ``…`` ``．``
  ``।`` ``۔`` ``؟`` ``။`` ``၊`` ``။``). The splitter prefers
  these — a chunk only falls back to weaker boundaries when no
  strong terminator divides it sensibly.
* **Weak** terminators are clause boundaries: ``,`` ``;`` ``:``
  ``、`` ``，`` ``；`` ``：`` and friends. Used as a second-pass
  fallback when strong-split chunks still exceed ``max_chars``.

Each tier splits into:

* **Self-terminating** characters that end a sentence on their own
  (CJK / Indic / Arabic punctuation — no trailing space convention).
* **Trail-terminating** characters that need following whitespace
  (or end-of-input) to count (Western punctuation — avoids
  splitting "Mr." or decimal points like ``3.14``).

Coverage targets the 10 languages the LunaVox model supports today
(per ``README.md``): Chinese, English, Japanese, Korean, Russian,
German, French, Italian, Spanish, Portuguese. Indic / Arabic / Thai
characters are included so the same algorithm scales to those
scripts when models add support — no separate code path needed.
"""

from __future__ import annotations

# --- strong terminators (sentence boundaries) -----------------------

# Self-terminating punctuation: CJK fullwidth + ideographic stops, plus
# Devanagari/Urdu/Arabic/Burmese/Thai equivalents. The splitter treats
# any of these as an immediate sentence end without requiring a
# following space — these scripts have no trailing-space convention.
STRONG_TERMINATORS_SELF: str = (
    "\u3002"  # 。 ideographic full stop (CN/JP)
    "\uff01"  # ！ fullwidth exclamation
    "\uff1f"  # ？ fullwidth question
    "\u2026"  # … horizontal ellipsis
    "\uff0e"  # ． fullwidth full stop
    "\u0964"  # । devanagari danda (Hindi/Sanskrit/Bengali)
    "\u0965"  # ॥ double danda
    "\u06d4"  # ۔ Arabic full stop (Urdu)
    "\u061f"  # ؟ Arabic question mark
    "\u0e2f"  # ฯ Thai paiyannoi
    "\u104a"  # ။ Burmese little section
    "\u104b"  # ။ Burmese section
)

# Trail-terminating punctuation: Western dot/exclaim/question. The
# splitter only counts these as a sentence boundary when followed by
# whitespace OR end-of-input, so abbreviations ("Mr.") and decimals
# ("3.14") aren't sliced mid-sentence.
STRONG_TERMINATORS_TRAIL: str = ".!?"

# --- weak terminators (clause boundaries) ---------------------------

# Self-terminating clause marks: CJK comma family, semicolon family.
WEAK_TERMINATORS_SELF: str = (
    "\u3001"  # 、 ideographic comma
    "\uff0c"  # ， fullwidth comma
    "\uff1b"  # ； fullwidth semicolon
    "\uff1a"  # ： fullwidth colon
    "\u060c"  # ، Arabic comma
    "\u061b"  # ؛ Arabic semicolon
)

# Trail-terminating clause marks: ASCII punctuation. Same trailing
# whitespace requirement as the strong tier.
WEAK_TERMINATORS_TRAIL: str = ",;:"

# --- soft break points (whitespace) --------------------------------

# Used as the third fallback when neither strong nor weak split keeps
# a segment under max_chars (e.g. a long unpunctuated phrase).
SOFT_BREAK: str = " \t\n\r\u3000"  # incl. ideographic space


def is_terminator(char: str, *, strong: bool = True) -> bool:
    """``True`` iff ``char`` is a single-char sentence (or clause) terminator.

    The tier table membership is the only language-aware check in the
    splitter — see module docstring for why this is data-driven.
    """
    if not char:
        return False
    if strong:
        return char in STRONG_TERMINATORS_SELF or char in STRONG_TERMINATORS_TRAIL
    return char in WEAK_TERMINATORS_SELF or char in WEAK_TERMINATORS_TRAIL
