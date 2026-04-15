"""Catalog integrity checks for ``lunavox.model.config``."""

from __future__ import annotations

import pytest

from lunavox.model.config import (
    MODELS,
    ModelSpec,
    all_models,
    get_model,
    model_keys,
)


def test_models_registry_nonempty():
    assert len(MODELS) >= 5, "Expected at least the five Qwen3-TTS catalog entries"


def test_all_models_order_matches_keys():
    keys = model_keys()
    specs = all_models()
    assert [s.name for s in specs] == keys


def test_every_spec_has_required_fields():
    for spec in all_models():
        assert isinstance(spec, ModelSpec)
        assert spec.name
        assert spec.repo
        assert spec.display_name
        assert spec.size in {"0.6B", "1.7B"}
        assert spec.mode in {"base", "custom", "design"}


def test_repo_id_has_org_prefix():
    for spec in all_models():
        assert spec.repo_id.startswith("Qwen/")
        assert spec.repo_id.endswith(spec.repo)


def test_get_model_round_trip():
    for key in model_keys():
        assert get_model(key).name == key


def test_get_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("nonexistent_model")


def test_modelspec_is_frozen():
    """Catalog entries must be immutable so downstream code can cache
    them without worrying about mutation."""
    spec = all_models()[0]
    with pytest.raises((AttributeError, TypeError)):
        spec.name = "other"  # type: ignore[misc]
