from __future__ import annotations

import unittest

from lunavox.core import deps
from lunavox.core.deps import DependencyPolicy, ensure_dependency_group


class DependencyPolicyTests(unittest.TestCase):
    def test_no_install_errors(self) -> None:
        original = deps.DEPENDENCY_GROUPS.get("convert", [])
        deps.DEPENDENCY_GROUPS["convert"] = [("definitely_missing_pkg_xyz", "definitely-missing-pkg-xyz")]
        try:
            with self.assertRaises(RuntimeError):
                ensure_dependency_group("convert", DependencyPolicy(yes=False, no_install=True))
        finally:
            deps.DEPENDENCY_GROUPS["convert"] = original


if __name__ == "__main__":
    unittest.main()

