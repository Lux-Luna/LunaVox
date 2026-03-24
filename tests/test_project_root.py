from __future__ import annotations

import os
import shutil
import unittest
import uuid
from pathlib import Path

from lunavox.core.project import resolve_project_root

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


class ProjectRootTests(unittest.TestCase):
    def test_resolve_project_root_with_explicit(self) -> None:
        tmp = _new_case_dir("explicit")
        try:
            root = tmp / "project"
            _init_project(root)
            resolved = resolve_project_root(root)
            self.assertEqual(resolved, root.resolve())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_resolve_project_root_from_cwd(self) -> None:
        tmp = _new_case_dir("cwd")
        try:
            root = tmp / "project"
            _init_project(root)
            nested = root / "a" / "b"
            nested.mkdir(parents=True, exist_ok=True)
            old_cwd = Path.cwd()
            try:
                os.chdir(nested)
                resolved = resolve_project_root()
                self.assertEqual(resolved, root.resolve())
            finally:
                os.chdir(old_cwd)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_resolve_project_root_invalid_explicit(self) -> None:
        tmp = _new_case_dir("invalid")
        try:
            bad_root = tmp / "not_project"
            bad_root.mkdir(parents=True, exist_ok=True)
            with self.assertRaises(RuntimeError):
                resolve_project_root(bad_root)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
