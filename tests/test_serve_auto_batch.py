"""Tests for :mod:`lunavox.serve.auto_batch`.

The pynvml probe is mocked so the suite stays deterministic across
hosts (CI runners have no GPU; dev hosts have varying VRAM).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip(
    "pydantic",
    reason="serve tests need the [serve] extra",
    exc_type=ImportError,
)


def test_integer_passthrough_clamped(tmp_path: Path):
    from lunavox.serve.auto_batch import resolve_batch_size

    assert resolve_batch_size(4, model_dir=tmp_path) == 4
    assert resolve_batch_size(1, model_dir=tmp_path) == 1
    # Below MIN_BATCH clamps up.
    assert resolve_batch_size(0, model_dir=tmp_path) == 1
    # Above MAX_BATCH clamps down.
    assert resolve_batch_size(99, model_dir=tmp_path) == 16


def test_string_integer_accepted(tmp_path: Path):
    from lunavox.serve.auto_batch import resolve_batch_size

    assert resolve_batch_size("2", model_dir=tmp_path) == 2
    assert resolve_batch_size("8", model_dir=tmp_path) == 8


def test_invalid_string_raises(tmp_path: Path):
    from lunavox.serve.auto_batch import resolve_batch_size

    with pytest.raises(ValueError, match="auto"):
        resolve_batch_size("nonsense", model_dir=tmp_path)


def test_auto_falls_back_when_probe_unavailable(tmp_path: Path, monkeypatch):
    from lunavox.serve import auto_batch

    monkeypatch.setattr(auto_batch, "_probe_free_vram_mb", lambda: None)
    result = auto_batch.resolve_batch_size("auto", model_dir=tmp_path)
    assert result == auto_batch.DEFAULT_FALLBACK


def test_auto_picks_more_slots_with_more_vram(tmp_path: Path, monkeypatch):
    from lunavox.serve import auto_batch

    # 16 GB free + 1.1 GB per slot + 80% headroom → ~11 slots, clamped to 16
    monkeypatch.setattr(auto_batch, "_probe_free_vram_mb", lambda: 16384)
    small_dir = tmp_path / "base_small"
    small_dir.mkdir()
    result = auto_batch.resolve_batch_size("auto", model_dir=small_dir)
    assert result >= 8


def test_auto_picks_one_slot_on_tiny_gpu(tmp_path: Path, monkeypatch):
    from lunavox.serve import auto_batch

    # 1 GB free, 0.6B per-slot ~1.1 GB → headroom puts us under, fall to 1.
    monkeypatch.setattr(auto_batch, "_probe_free_vram_mb", lambda: 1024)
    small_dir = tmp_path / "base_small"
    small_dir.mkdir()
    result = auto_batch.resolve_batch_size("auto", model_dir=small_dir)
    assert result == 1


def test_per_slot_override_env(tmp_path: Path, monkeypatch):
    from lunavox.serve import auto_batch

    # 8 GB free, 4 GB per slot override → 8 * 0.8 / 4 = 1.6 → 1
    monkeypatch.setenv(auto_batch.PER_SLOT_OVERRIDE_ENV, "4096")
    monkeypatch.setattr(auto_batch, "_probe_free_vram_mb", lambda: 8192)
    result = auto_batch.resolve_batch_size("auto", model_dir=tmp_path)
    assert result == 1


def test_large_model_picks_smaller_pool(tmp_path: Path, monkeypatch):
    from lunavox.serve import auto_batch

    # 8 GB free + ~3.1 GB per large-model slot → 8*0.8/3.1 ≈ 2 slots
    monkeypatch.delenv(auto_batch.PER_SLOT_OVERRIDE_ENV, raising=False)
    monkeypatch.setattr(auto_batch, "_probe_free_vram_mb", lambda: 8192)
    large_dir = tmp_path / "base"
    large_dir.mkdir()
    result = auto_batch.resolve_batch_size("auto", model_dir=large_dir)
    assert 1 <= result <= 3
