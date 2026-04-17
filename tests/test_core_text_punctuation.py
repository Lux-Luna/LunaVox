"""Unit tests for :mod:`lunavox.core.text.punctuation`.

The module is pure data + a small predicate. These tests pin down
which characters belong to which tier so regressions in the splitter
or streaming buffer are easy to localise — if the splitter starts
dropping a language, check here first for table coverage.
"""

from __future__ import annotations

from lunavox.core.text.punctuation import (
    SOFT_BREAK,
    STRONG_TERMINATORS_SELF,
    STRONG_TERMINATORS_TRAIL,
    WEAK_TERMINATORS_SELF,
    WEAK_TERMINATORS_TRAIL,
    is_terminator,
)


def test_strong_trail_is_ascii_sentence_punctuation():
    assert set(STRONG_TERMINATORS_TRAIL) == {".", "!", "?"}


def test_cjk_full_width_stops_are_strong_self_terminators():
    for ch in ("\u3002", "\uff01", "\uff1f"):  # 。！？
        assert ch in STRONG_TERMINATORS_SELF


def test_ellipsis_and_fullwidth_stop_are_strong_self_terminators():
    assert "\u2026" in STRONG_TERMINATORS_SELF  # …
    assert "\uff0e" in STRONG_TERMINATORS_SELF  # ．


def test_indic_danda_is_a_strong_self_terminator():
    assert "\u0964" in STRONG_TERMINATORS_SELF  # ।


def test_arabic_question_mark_is_strong_self_terminator():
    assert "\u061f" in STRONG_TERMINATORS_SELF  # ؟


def test_weak_tier_covers_ascii_and_cjk_clause_marks():
    for ch in (",", ";", ":"):
        assert ch in WEAK_TERMINATORS_TRAIL
    for ch in ("\u3001", "\uff0c", "\uff1b", "\uff1a"):
        assert ch in WEAK_TERMINATORS_SELF


def test_soft_break_includes_ideographic_space():
    """Ideographic space (U+3000) shows up after CJK punctuation in
    formally laid-out text and must count as whitespace."""
    assert "\u3000" in SOFT_BREAK
    assert " " in SOFT_BREAK
    assert "\n" in SOFT_BREAK


def test_is_terminator_strong_true_for_ascii_and_cjk_terminators():
    assert is_terminator(".")
    assert is_terminator("!")
    assert is_terminator("\u3002")
    assert is_terminator("\uff1f")


def test_is_terminator_strong_false_for_weak_marks():
    assert not is_terminator(",")
    assert not is_terminator("\u3001")
    assert not is_terminator(" ")


def test_is_terminator_weak_true_for_ascii_and_cjk_clause_marks():
    assert is_terminator(",", strong=False)
    assert is_terminator(":", strong=False)
    assert is_terminator("\u3001", strong=False)


def test_is_terminator_empty_input_is_false():
    assert not is_terminator("")
