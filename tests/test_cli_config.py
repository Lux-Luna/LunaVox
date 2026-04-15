"""Tests for the ``lunavox.cli._config`` profile loader.

Covers the precedence chain: CLI overrides > env > profile table >
default table > hardcoded defaults.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lunavox.cli._config import ResolvedConfig, load_config


@pytest.fixture
def fake_project_root(tmp_path: Path) -> Path:
    """A throwaway directory that looks enough like a lunavox checkout
    that ``resolve_project_root`` accepts it. ``_looks_like_project_root``
    requires both ``CMakeLists.txt`` and a ``src/`` subdirectory."""
    (tmp_path / "CMakeLists.txt").write_text("project(lunavox)\n")
    (tmp_path / "src").mkdir()
    return tmp_path


@pytest.fixture
def write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point LUNAVOX_CONFIG at a temp file and return a writer helper."""

    config_file = tmp_path / "config.toml"

    def _write(body: str) -> Path:
        config_file.write_text(body, encoding="utf-8")
        monkeypatch.setenv("LUNAVOX_CONFIG", str(config_file))
        return config_file

    return _write


def test_defaults_when_no_config_file(fake_project_root: Path, monkeypatch):
    monkeypatch.delenv("LUNAVOX_CONFIG", raising=False)
    cfg = load_config(project_root=fake_project_root)
    assert isinstance(cfg, ResolvedConfig)
    assert cfg.model == "base_small"
    assert cfg.backend == "auto"
    assert cfg.temperature == 0.6
    assert cfg.profile_name is None


def test_default_table_overrides_hardcoded(fake_project_root, write_config):
    write_config(
        """
[default]
model = "alt_model"
backend = "vulkan"
temperature = 0.8
"""
    )
    cfg = load_config(project_root=fake_project_root)
    assert cfg.model == "alt_model"
    assert cfg.backend == "vulkan"
    assert cfg.temperature == 0.8
    # Untouched fields keep their hardcoded defaults.
    assert cfg.n_threads == 4


def test_profile_overrides_default_table(fake_project_root, write_config):
    write_config(
        """
[default]
backend = "cpu"
temperature = 0.6

[profile.fast]
backend = "vulkan+dml"
temperature = 0.9
"""
    )
    cfg = load_config(profile="fast", project_root=fake_project_root)
    assert cfg.backend == "vulkan+dml"
    assert cfg.temperature == 0.9
    assert cfg.profile_name == "fast"


def test_unknown_profile_raises(fake_project_root, write_config):
    write_config(
        """
[profile.fast]
backend = "vulkan"
"""
    )
    with pytest.raises(RuntimeError, match="Profile 'slow'"):
        load_config(profile="slow", project_root=fake_project_root)


def test_env_overrides_profile(fake_project_root, write_config, monkeypatch):
    write_config(
        """
[profile.fast]
backend = "vulkan"
temperature = 0.9
"""
    )
    monkeypatch.setenv("LUNAVOX_BACKEND", "cuda")
    monkeypatch.setenv("LUNAVOX_TOP_K", "30")
    cfg = load_config(profile="fast", project_root=fake_project_root)
    assert cfg.backend == "cuda"  # env wins
    assert cfg.temperature == 0.9  # profile still visible
    assert cfg.top_k == 30  # env casts to int


def test_cli_override_wins_over_everything(fake_project_root, write_config, monkeypatch):
    write_config(
        """
[default]
backend = "cpu"
temperature = 0.6
"""
    )
    monkeypatch.setenv("LUNAVOX_BACKEND", "vulkan")
    cfg = load_config(
        project_root=fake_project_root,
        overrides={"backend": "cuda", "temperature": 0.42},
    )
    assert cfg.backend == "cuda"  # CLI beats env
    assert cfg.temperature == 0.42  # CLI beats default table


def test_session_flags_set_correctly(fake_project_root):
    cfg = load_config(
        project_root=fake_project_root,
        yes=True,
        no_install=True,
        verbose=True,
    )
    assert cfg.yes is True
    assert cfg.no_install is True
    assert cfg.verbose is True
