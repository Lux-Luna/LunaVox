from __future__ import annotations
import platform
from typing import Type
from .base import ToolchainResolver
from .windows import WindowsResolver
from .macos import MacOSResolver
from .linux import LinuxResolver

def get_resolver_class() -> Type[ToolchainResolver]:
    sys_name = platform.system()
    if sys_name == "Windows":
        return WindowsResolver
    elif sys_name == "Darwin":
        return MacOSResolver
    else:
        return LinuxResolver
