"""Unit tests for :class:`lunavox.runtime.SynthesisParams` and
:class:`SynthesisResult` / :class:`SynthesisStats`.

No liblunavox needed — these only touch the Python dataclasses.
"""

from __future__ import annotations

from lunavox.runtime import (
    MemStats,
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
    assert s.mem.rss_peak_bytes == 0
    assert s.mem.vram_measured is False


def test_memstats_delta_properties_handle_missing_start():
    """`*_delta_bytes` / `*_leak_bytes` must clamp at zero — a peak
    sample taken before the start marker was set (shouldn't happen, but
    guard against the result being negative rather than silently
    rendering a wrap-around unsigned value)."""
    mem = MemStats(
        rss_start_bytes=5 * 1024**3,
        rss_end_bytes=6 * 1024**3,
        rss_peak_bytes=8 * 1024**3,
        vram_start_bytes=1 * 1024**3,
        vram_end_bytes=2 * 1024**3,
        vram_peak_bytes=3 * 1024**3,
        vram_measured=True,
    )
    assert mem.rss_peak_delta_bytes == 3 * 1024**3
    assert mem.rss_leak_bytes == 1024**3
    assert mem.vram_peak_delta_bytes == 2 * 1024**3
    assert mem.vram_leak_bytes == 1024**3


def test_memstats_unmeasured_vram_does_not_affect_rss():
    """`vram_measured=False` is a per-device NVML signal — RSS readings
    are independent and must still compute."""
    mem = MemStats(rss_start_bytes=1_000, rss_peak_bytes=5_000)
    assert mem.vram_measured is False
    assert mem.rss_peak_delta_bytes == 4_000


def test_from_overrides_applies_only_non_none_fields():
    """``None`` means "no override" — keep the dataclass default.
    Useful for entries that pass optional CLI flags straight through."""
    p = SynthesisParams.from_overrides(temperature=0.9, top_p=None, top_k=None)
    assert p.temperature == 0.9
    # Untouched fields keep their defaults.
    assert p.top_p == 1.0
    assert p.top_k == 50


def test_from_overrides_with_all_none_returns_pure_defaults():
    p = SynthesisParams.from_overrides(temperature=None, top_p=None)
    assert p == SynthesisParams()


def test_from_overrides_rejects_unknown_fields():
    """Typos must fail fast, not silently ignore the override."""
    import pytest

    with pytest.raises(TypeError, match="Unknown SynthesisParams field"):
        SynthesisParams.from_overrides(temprature=0.7)  # missing 'e'


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
