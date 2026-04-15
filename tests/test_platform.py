"""Host-platform helpers — exercises the three ``sys.platform`` branches
without actually switching OS."""

from __future__ import annotations

import sys
from unittest.mock import patch

from lunavox.core import platform as plat


def test_shared_lib_name_windows():
    with patch.object(sys, "platform", "win32"):
        assert plat.shared_lib_name("lunavox") == "lunavox.dll"


def test_shared_lib_name_macos():
    with patch.object(sys, "platform", "darwin"):
        assert plat.shared_lib_name("lunavox") == "liblunavox.dylib"


def test_shared_lib_name_linux():
    with patch.object(sys, "platform", "linux"):
        assert plat.shared_lib_name("lunavox") == "liblunavox.so"


def test_executable_suffix_windows():
    with patch.object(sys, "platform", "win32"):
        assert plat.executable_suffix() == ".exe"


def test_executable_suffix_posix():
    with patch.object(sys, "platform", "linux"):
        assert plat.executable_suffix() == ""
    with patch.object(sys, "platform", "darwin"):
        assert plat.executable_suffix() == ""


def test_is_helpers_mutually_exclusive():
    """Exactly one of is_windows / is_macos / is_linux is true for the
    host running the test."""
    flags = [plat.is_windows(), plat.is_macos(), plat.is_linux()]
    assert sum(flags) == 1, f"Platform flags not mutually exclusive: {flags}"


def test_is_linux_under_patched_platform():
    with patch.object(sys, "platform", "linux2"):
        assert plat.is_linux() is True
        assert plat.is_windows() is False
        assert plat.is_macos() is False
