from __future__ import annotations
from .base import ToolchainResolver, Builder


class LinuxBuilder(Builder):
    """Linux builder — inherits the generic post_build (metadata.json copy)
    from the base class. Kept as its own subclass so any future Linux-only
    work (e.g. strip, patchelf) has a home that stays symmetric with
    WindowsBuilder / MacOSBuilder."""
    pass


class LinuxResolver(ToolchainResolver):
    """Linux-specific toolchain resolution."""
    def resolve(self) -> tuple[str, list[str]]:
        tools = {
            "cmake": self._which("cmake"),
            "g++": self._which("g++") or self._which("clang++"),
            "ninja": self._which("ninja"),
            "make": self._which("make"),
        }
        if not tools["cmake"]:
            raise RuntimeError("cmake not found. Please install CMake via your package manager.")
        if not tools["g++"]:
            raise RuntimeError("C++ compiler (gcc/clang) not found.")
        
        args = [f"-DCMAKE_CXX_COMPILER={self._cmake_path(tools['g++'])}"]
        if tools["ninja"]: 
            args += ["-G", "Ninja"]
        elif tools["make"]:
            args += ["-G", "Unix Makefiles"]
        
        return "linux-gcc", args
