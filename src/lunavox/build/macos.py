from __future__ import annotations

import os
import platform
from pathlib import Path

from .base import ToolchainResolver, Builder


class MacOSBuilder(Builder):
    """macOS builder — inherits the generic post_build (metadata.json copy).
    Exists as its own subclass so future macOS-only fixups (rpath rewriting,
    codesigning) have a home without bifurcating the build factory."""
    pass


class MacOSResolver(ToolchainResolver):
    """macOS toolchain resolver.

    Responsibilities beyond the generic path-probing done by the base class:

    - Detect host architecture (``arm64`` vs ``x86_64``) and pass
      ``CMAKE_OSX_ARCHITECTURES`` explicitly. Letting CMake auto-pick is
      unreliable once Xcode cross-targets come into play, and llama.cpp /
      ONNX Runtime prebuilt binaries are arch-specific so a mismatch will
      silently load a wrong-arch dylib at runtime.

    - Pick the correct Homebrew prefix (``/opt/homebrew`` on Apple Silicon,
      ``/usr/local`` on Intel) and prepend it to ``PATH`` so tools installed
      via ``brew`` are discoverable even when launched from a minimal login
      shell.

    - Respect ``LUNAVOX_OSX_ARCH`` / ``MACOSX_DEPLOYMENT_TARGET`` env overrides
      so users can force an arch or deployment target without editing code.
    """

    def resolve(self) -> tuple[str, list[str]]:
        host_arch = (os.environ.get("LUNAVOX_OSX_ARCH") or platform.machine() or "").lower()
        if host_arch in ("arm64", "aarch64"):
            arch = "arm64"
            brew_prefix = "/opt/homebrew"
        elif host_arch in ("x86_64", "amd64"):
            arch = "x86_64"
            brew_prefix = "/usr/local"
        else:
            arch = platform.machine() or "arm64"
            brew_prefix = "/opt/homebrew" if Path("/opt/homebrew").exists() else "/usr/local"

        if Path(brew_prefix).exists():
            self.ctx.env["PATH"] = f"{brew_prefix}/bin:{self.ctx.env.get('PATH', '')}"
            os.environ["PATH"] = self.ctx.env["PATH"]

        tools = {
            "cmake": self._which("cmake") or f"{brew_prefix}/bin/cmake",
            "clang++": self._which("clang++"),
            "ninja": self._which("ninja"),
            "make": self._which("make"),
        }

        if not Path(tools["cmake"]).exists() and not self._which("cmake"):
            raise RuntimeError("cmake not found. Install via 'brew install cmake'.")
        if not tools["clang++"]:
            raise RuntimeError(
                "clang++ not found. Install Xcode Command Line Tools via 'xcode-select --install'."
            )

        deployment_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "11.0")

        args = [
            f"-DCMAKE_CXX_COMPILER={self._cmake_path(tools['clang++'])}",
            f"-DCMAKE_OSX_ARCHITECTURES={arch}",
            f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}",
        ]
        if tools["ninja"]:
            args += ["-G", "Ninja"]
        elif tools["make"]:
            args += ["-G", "Unix Makefiles"]

        return f"macos-clang-{arch}", args
