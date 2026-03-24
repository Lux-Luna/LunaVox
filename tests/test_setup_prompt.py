from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from lunavox.cli.main import RuntimeState, _setup_internal

TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "_tmp_tests"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _new_case_dir(prefix: str) -> Path:
    case_dir = TEST_TMP_ROOT / f"{prefix}_{uuid.uuid4().hex}"
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


class SetupPromptTests(unittest.TestCase):
    def test_missing_source_prompts_in_english_and_can_abort(self) -> None:
        tmp = _new_case_dir("setup_prompt")
        try:
            project_root = tmp / "project"
            project_root.mkdir(parents=True, exist_ok=True)
            missing_source = tmp / "missing_source"
            model_dest = tmp / "model_dest"

            cfg = SimpleNamespace(name="base_small", source=missing_source, dest=model_dest)
            models_obj = MagicMock()
            models_obj.by_name.return_value = cfg

            state = RuntimeState(
                project_root=project_root,
                yes=False,
                no_install=False,
                verbose=False,
                latest_log=project_root / "logs" / "latest.log",
            )

            with patch("lunavox.cli.main._ensure_convert_deps"), patch(
                "lunavox.cli.main.Models", return_value=models_obj
            ), patch("lunavox.cli.main._confirm_or_fail", return_value=False) as confirm_mock:
                with self.assertRaises(RuntimeError):
                    _setup_internal(state, model="base_small", models_dir="", force=False)

                self.assertEqual(
                    confirm_mock.call_args[0][1],
                    "Model 'base_small' source files are missing. Download from HuggingFace now?",
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

