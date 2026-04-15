from __future__ import annotations
import platform
from typing import Type
from .base import Builder, ToolchainResolver
from .windows import WindowsBuilder, WindowsResolver
from .macos import MacOSBuilder, MacOSResolver
from .linux import LinuxBuilder, LinuxResolver


def get_resolver_class() -> Type[ToolchainResolver]:
    sys_name = platform.system()
    if sys_name == "Windows":
        return WindowsResolver
    if sys_name == "Darwin":
        return MacOSResolver
    if sys_name == "Linux":
        return LinuxResolver
    raise NotImplementedError(f"Unsupported host platform: {sys_name}")


def get_builder_class() -> Type[Builder]:
    sys_name = platform.system()
    if sys_name == "Windows":
        return WindowsBuilder
    if sys_name == "Darwin":
        return MacOSBuilder
    if sys_name == "Linux":
        return LinuxBuilder
    raise NotImplementedError(f"Unsupported host platform: {sys_name}")
