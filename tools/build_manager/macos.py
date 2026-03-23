from __future__ import annotations
import subprocess
from pathlib import Path
from .base import ToolchainResolver

class MacOSResolver(ToolchainResolver):
    """MacOS-specific toolchain resolution."""
    def resolve(self) -> tuple[str, list[str]]:
        # Common homebrew paths
        brew_prefix = "/opt/homebrew" if Path("/opt/homebrew").exists() else "/usr/local"
        os.environ["PATH"] = f"{brew_prefix}/bin:{os.environ.get('PATH', '')}"
        
        tools = {
            "cmake": self._which("cmake") or f"{brew_prefix}/bin/cmake",
            "clang++": self._which("clang++"),
            "ninja": self._which("ninja"),
            "make": self._which("make"),
        }
        
        if not tools["cmake"]: raise RuntimeError("cmake not found. Try 'brew install cmake'")
        if not tools["clang++"]: raise RuntimeError("clang++ not found. Please install Xcode Command Line Tools.")
        
        args = [f"-DCMAKE_CXX_COMPILER={self._cmake_path(tools['clang++'])}"]
        if tools["ninja"]: 
            args += ["-G", "Ninja"]
        elif tools["make"]:
            args += ["-G", "Unix Makefiles"]
        
        return "macos-clang", args
