"""Stats JSON parsing — mirrors the contract between ``src/main.cpp``
(producer) and ``benchmark/run_benchmark.py`` / GUI (consumer)."""

from __future__ import annotations

import json

import pytest

from lunavox.core.stats_schema import ParsedStats, parse_stats_json


def test_parse_full_payload():
    data = {
        "t_load_ms": 1234,
        "t_warmup_ms": 567,
        "runs": [
            {
                "run_id": 0,
                "rtf": 0.21,
                "sample_rate": 48000,
                "n_samples": 120000,
                "timing_ms": {"tokenize": 5, "encode": 10, "generate": 80, "decode": 30},
                "stream": {"first_chunk_frames": 8, "t_first_audio_ms": 180},
                "mem": {"rss_peak": 1_500_000_000},
            }
        ],
    }
    parsed = parse_stats_json(data)
    assert isinstance(parsed, ParsedStats)
    assert parsed.load_ms == 1234
    assert parsed.warmup_ms == 567
    assert len(parsed.runs) == 1
    assert parsed.rtf(0) == pytest.approx(0.21)
    assert parsed.first_run["run_id"] == 0


def test_parse_missing_optional_fields():
    data = {"runs": []}
    parsed = parse_stats_json(data)
    assert parsed.load_ms == 0
    assert parsed.warmup_ms == 0
    assert parsed.runs == []
    assert parsed.first_run == {}
    assert parsed.rtf(0) == 0.0
    assert parsed.rtf(999) == 0.0


def test_parse_from_json_file(tmp_path):
    payload = {"t_load_ms": 42, "runs": [{"rtf": 0.5}]}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    parsed = parse_stats_json(path)
    assert parsed.load_ms == 42
    assert parsed.rtf(0) == 0.5


def test_parse_rejects_non_dict():
    with pytest.raises(ValueError, match="object"):
        parse_stats_json([1, 2, 3])


def test_parse_rejects_non_list_runs():
    with pytest.raises(ValueError, match="runs"):
        parse_stats_json({"t_load_ms": 0, "runs": "not a list"})


def test_parse_coerces_none_timings():
    """Producer emits null for missing scalars; consumer must not crash."""
    parsed = parse_stats_json({"t_load_ms": None, "t_warmup_ms": None, "runs": []})
    assert parsed.load_ms == 0
    assert parsed.warmup_ms == 0
