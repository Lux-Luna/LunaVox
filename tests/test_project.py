"""Project-root resolution — env-var override, upward probing, and
hard failure when no markers are found."""

from __future__ import annotations

from pathlib import Path

import pytest

from lunavox.core.project import DEPLOYMENT_MARKER, resolve_project_root


def _make_fake_root(base: Path) -> Path:
    """Dev-checkout layout: ``CMakeLists.txt`` + ``src/``."""
    base.mkdir(parents=True, exist_ok=True)
    (base / "src").mkdir()
    (base / "CMakeLists.txt").write_text("project(lunavox)\n", encoding="utf-8")
    return base


def _make_deployment_root(base: Path) -> Path:
    """Deployment layout: just the ``.lunavox-root`` marker file."""
    base.mkdir(parents=True, exist_ok=True)
    (base / DEPLOYMENT_MARKER).write_text("lunavox 2.2.2 deployment\n", encoding="utf-8")
    return base


def test_resolves_via_explicit_argument(tmp_path, monkeypatch):
    root = _make_fake_root(tmp_path / "fake_repo")
    monkeypatch.delenv("LUNAVOX_PROJECT_ROOT", raising=False)
    resolved = resolve_project_root(root)
    assert resolved == root.resolve()
    assert (resolved / "models").exists()
    assert (resolved / "lib").exists()


def test_resolves_via_env_var(tmp_path, monkeypatch):
    root = _make_fake_root(tmp_path / "env_repo")
    monkeypatch.setenv("LUNAVOX_PROJECT_ROOT", str(root))
    monkeypatch.chdir(tmp_path)
    resolved = resolve_project_root()
    assert resolved == root.resolve()


def test_resolves_via_cwd_ancestry(tmp_path, monkeypatch):
    root = _make_fake_root(tmp_path / "ancestry_repo")
    deep = root / "a" / "b" / "c"
    deep.mkdir(parents=True)
    monkeypatch.delenv("LUNAVOX_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(deep)
    resolved = resolve_project_root()
    assert resolved == root.resolve()


def test_rejects_invalid_explicit_root(tmp_path, monkeypatch):
    monkeypatch.delenv("LUNAVOX_PROJECT_ROOT", raising=False)
    empty = tmp_path / "not_a_repo"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="Invalid --project-root"):
        resolve_project_root(empty)


def test_resolves_deployment_layout_via_marker_file(tmp_path, monkeypatch):
    """A directory with just ``.lunavox-root`` is a valid root — this
    is the layout the Docker image and standalone bundles use."""
    root = _make_deployment_root(tmp_path / "fake_deployment")
    monkeypatch.delenv("LUNAVOX_PROJECT_ROOT", raising=False)
    resolved = resolve_project_root(root)
    assert resolved == root.resolve()
    # models/ and lib/ are auto-created by the resolver either way.
    assert (resolved / "models").exists()
    assert (resolved / "lib").exists()


def test_deployment_layout_env_var(tmp_path, monkeypatch):
    """Deployment roots are also reachable via ``LUNAVOX_PROJECT_ROOT``."""
    root = _make_deployment_root(tmp_path / "env_deployment")
    monkeypatch.setenv("LUNAVOX_PROJECT_ROOT", str(root))
    monkeypatch.chdir(tmp_path)
    resolved = resolve_project_root()
    assert resolved == root.resolve()


def test_falls_back_to_error_when_nothing_matches(tmp_path, monkeypatch):
    monkeypatch.delenv("LUNAVOX_PROJECT_ROOT", raising=False)
    # Descend into a directory with no repo markers anywhere upward.
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    monkeypatch.chdir(isolated)
    # Don't assume the ancestors of tmp_path — Windows temp paths may
    # live on a drive root. Use a synthetic PWD by also clearing env var.
    # If the real ancestry accidentally contains a LunaVox repo we skip
    # the assertion rather than fail.
    try:
        result = resolve_project_root()
    except RuntimeError as err:
        assert "project root" in str(err).lower()
        return
    # Ancestry hit an actual LunaVox checkout — accept and move on.
    assert (result / "src").exists()
