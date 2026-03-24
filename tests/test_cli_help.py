from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from typer.testing import CliRunner

from lunavox.cli.main import app

TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "_tmp_tests"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _new_case_dir(prefix: str) -> Path:
    case_dir = TEST_TMP_ROOT / f"{prefix}_{uuid.uuid4().hex}"
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _init_project(path: Path) -> None:
    (path / "src").mkdir(parents=True, exist_ok=True)
    (path / "lib").mkdir(parents=True, exist_ok=True)
    (path / "models").mkdir(parents=True, exist_ok=True)
    (path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.16)\n", encoding="utf-8")


class CliHelpTests(unittest.TestCase):
    def test_help_smoke(self) -> None:
        tmp = _new_case_dir("help")
        try:
            root = tmp / "project"
            _init_project(root)
            runner = CliRunner()
            result = runner.invoke(app, ["--project-root", str(root), "--help"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("LunaVox unified CLI", result.stdout)
            self.assertIn("setup", result.stdout)
            self.assertIn("build", result.stdout)
            self.assertIn("bootstrap", result.stdout)
            self.assertIn("download", result.stdout)
            self.assertIn("doctor", result.stdout)
            self.assertNotIn("convert", result.stdout)
            self.assertNotIn("quantize", result.stdout)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_setup_help_has_no_quant_options(self) -> None:
        tmp = _new_case_dir("setup_help")
        try:
            root = tmp / "project"
            _init_project(root)
            runner = CliRunner()
            result = runner.invoke(app, ["--project-root", str(root), "setup", "--help"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("--model", result.stdout)
            self.assertIn("--models-dir", result.stdout)
            self.assertIn("--force", result.stdout)
            self.assertNotIn("quant", result.stdout.lower())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_removed_commands_are_unknown(self) -> None:
        tmp = _new_case_dir("removed_cmds")
        try:
            root = tmp / "project"
            _init_project(root)
            runner = CliRunner()
            for cmd in (
                ["--project-root", str(root), "convert", "onnx"],
                ["--project-root", str(root), "quantize", "onnx"],
                ["--project-root", str(root), "download", "model"],
            ):
                result = runner.invoke(app, cmd)
                self.assertNotEqual(result.exit_code, 0)
                merged = (result.stdout or "") + "\n" + (getattr(result, "stderr", "") or "")
                self.assertIn("No such command", merged)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
