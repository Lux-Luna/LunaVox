"""CLI import-and-help smoke — catches the most common breakage
mode (``lunavox --help`` crashes on import due to a typer signature
change or missing dependency) without touching the C engine."""

from __future__ import annotations

from typer.testing import CliRunner


def test_cli_module_imports():
    from lunavox.cli.main import app  # noqa: F401


def test_cli_help_exits_cleanly():
    from lunavox.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    # Help output mentions the top-level command name.
    assert "lunavox" in result.output.lower() or "Usage" in result.output


def test_cli_subcommands_present():
    """Lock in the public subcommand surface. Adding a new command is
    fine — this test catches accidental *removals*."""
    from lunavox.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    expected = {"convert", "pull-model", "build", "bootstrap", "download-libs", "doctor"}
    missing = {cmd for cmd in expected if cmd not in result.output}
    assert not missing, f"CLI missing expected subcommands: {missing}\nOutput:\n{result.output}"


def test_subcommand_help_each_exits_cleanly():
    from lunavox.cli.main import app

    runner = CliRunner()
    for cmd in ("convert", "pull-model", "build", "bootstrap", "download-libs", "doctor"):
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0, f"{cmd} --help failed:\n{result.output}"
