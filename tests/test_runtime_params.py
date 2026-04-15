"""Unit tests for :class:`lunavox.runtime.SynthesisParams` and
:class:`SynthesisResult` / :class:`SynthesisStats`.

No liblunavox needed — these only touch the Python dataclasses.
"""

from __future__ import annotations

from lunavox.runtime import (
    SynthesisMode,
    SynthesisParams,
    SynthesisResult,
    SynthesisStats,
    default_params,
)


def test_default_params_is_a_fresh_instance():
    a = default_params()
    b = default_params()
    assert a == b
    assert a is not b, "default_params must not return a shared singleton"


def test_synthesis_params_override():
    p = SynthesisParams(temperature=0.9, top_k=20, ref_text="hello")
    assert p.temperature == 0.9
    assert p.top_k == 20
    assert p.ref_text == "hello"
    # Untouched fields keep their defaults.
    assert p.top_p == 1.0
    assert p.language_id == -1


def test_synthesis_stats_defaults_are_zero():
    s = SynthesisStats()
    assert s.t_total_ms == 0
    assert s.rtf == 0.0
    assert s.rss_peak_bytes == 0


def test_synthesis_result_holds_mode_for_telemetry():
    stats = SynthesisStats(t_total_ms=1234, rtf=0.25)
    r = SynthesisResult(
        audio=[0.0, 0.0, 0.0],
        sample_rate=24000,
        stats=stats,
        mode=SynthesisMode.CLONE_FILE,
    )
    assert r.mode is SynthesisMode.CLONE_FILE
    assert r.sample_rate == 24000
    assert r.stats.t_total_ms == 1234
