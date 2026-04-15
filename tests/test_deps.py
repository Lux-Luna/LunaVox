"""Dependency-group policy — the decision table that gates auto-install."""

from __future__ import annotations

import pytest

from lunavox.core.deps import (
    DEPENDENCY_GROUPS,
    DependencyPolicy,
    ensure_dependency_group,
    has_module,
    missing_modules,
)


def test_convert_group_declared():
    assert "convert" in DEPENDENCY_GROUPS
    # Touching this list should be intentional — assert a few core
    # entries that every Qwen3-TTS conversion path needs.
    names = {name for name, _pip in DEPENDENCY_GROUPS["convert"]}
    for required in ("torch", "numpy", "safetensors", "gguf", "transformers"):
        assert required in names, f"convert group missing required dep: {required}"


def test_has_module_for_stdlib():
    assert has_module("json") is True
    assert has_module("definitely_not_a_real_module_xyz") is False


def test_missing_modules_returns_missing_only():
    entries = [("json", "json"), ("definitely_not_real_xyz", "definitely_not_real_xyz")]
    missing = missing_modules(entries)
    assert missing == [("definitely_not_real_xyz", "definitely_not_real_xyz")]


def test_ensure_unknown_group_raises():
    with pytest.raises(RuntimeError, match="Unknown dependency group"):
        ensure_dependency_group("not_a_group", DependencyPolicy())


def test_ensure_no_install_raises_when_missing(monkeypatch):
    """With --no-install, ensure_dependency_group must raise rather than
    prompt or subprocess-install."""
    monkeypatch.setattr(
        "lunavox.core.deps.DEPENDENCY_GROUPS",
        {"synthetic": [("definitely_not_real_xyz", "fake-pip-name")]},
    )
    with pytest.raises(RuntimeError, match="Missing required dependencies"):
        ensure_dependency_group("synthetic", DependencyPolicy(no_install=True))


def test_ensure_returns_when_all_present():
    """A group whose modules all exist should be a no-op regardless of
    policy flags."""
    # `json` is always present.
    import lunavox.core.deps as deps_mod

    original = deps_mod.DEPENDENCY_GROUPS
    try:
        deps_mod.DEPENDENCY_GROUPS = {"noop": [("json", "json")]}
        ensure_dependency_group("noop", DependencyPolicy(no_install=True))
    finally:
        deps_mod.DEPENDENCY_GROUPS = original
