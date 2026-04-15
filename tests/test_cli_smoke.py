"""CLI import-and-help smoke — catches the most common breakage
mode (``lunavox --help`` crashes on import due to a typer signature
change or missing dependency) without touching the C engine.

The command surface is locked in here, so any accidental removal of
a subcommand trips the test suite before it hits users.
"""

from __future__ import annotations

from typer.testing import CliRunner


def test_cli_module_imports():
    from lunavox.cli.main import app  # noqa: F401


def test_cli_help_exits_cleanly():
    from lunavox.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "lunavox" in result.output.lower() or "Usage" in result.output


def test_cli_top_level_commands_present():
    """Lock in the public top-level command surface. ``model`` and
    ``build`` are typer groups; ``synth``, ``serve``, ``gui``,
    ``bootstrap``, and ``doctor`` are leaf commands."""
    from lunavox.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    expected = {"model", "build", "synth", "serve", "gui", "bootstrap", "doctor"}
    missing = {cmd for cmd in expected if cmd not in result.output}
    assert not missing, f"CLI missing expected commands: {missing}\nOutput:\n{result.output}"


def test_model_group_has_pull_convert_list():
    from lunavox.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["model", "--help"])
    assert result.exit_code == 0, result.output
    for sub in ("pull", "convert", "list"):
        assert sub in result.output, f"model {sub} missing:\n{result.output}"


def test_build_group_has_libs_subcommand():
    from lunavox.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["build", "--help"])
    assert result.exit_code == 0, result.output
    assert "libs" in result.output


def test_every_leaf_command_help_exits_cleanly():
    from lunavox.cli.main import app

    runner = CliRunner()
    leafs = [
        ["model", "pull"],
        ["model", "convert"],
        ["model", "list"],
        ["build", "libs"],
        ["synth"],
        ["serve"],
        ["gui"],
        ["bootstrap"],
        ["doctor"],
    ]
    for cmd in leafs:
        result = runner.invoke(app, [*cmd, "--help"])
        assert result.exit_code == 0, f"{' '.join(cmd)} --help failed:\n{result.output}"
