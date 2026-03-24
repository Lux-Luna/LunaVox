from __future__ import annotations
import os
import shutil
import subprocess
import platform
import re
import json
from pathlib import Path
from .base import ToolchainResolver, Builder

class WindowsResolver(ToolchainResolver):
    """Windows-specific toolchain resolution."""
    
    MSVC_VARS = [
        "INCLUDE", "LIB", "LIBPATH", "VSINSTALLDIR", "VCINSTALLDIR",
        "VCToolsInstallDir", "VisualStudioVersion", "WindowsSdkDir",
        "WindowsSDKLibVersion", "WindowsSDKVersion", "UniversalCRTSdkDir",
        "UCRTVersion", "VSCMD_ARG_TGT_ARCH", "VSCMD_ARG_HOST_ARCH",
        "DISTUTILS_USE_SDK", "MSSdk", "CMAKE_GENERATOR",
        "CMAKE_GENERATOR_PLATFORM", "CMAKE_GENERATOR_TOOLSET",
    ]
    
    COMPILER_ENV_VARS = [
        "CC", "CXX", "AR", "LD", "NM", "RANLIB",
        "CPPFLAGS", "CPPFLAGS_USED", "CFLAGS", "CXXFLAGS", "LDFLAGS"
    ]

    def _get_vs_installation_path(self) -> str | None:
        vswhere = self._which("vswhere") or str(Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"))
        if not Path(vswhere).exists(): return None

        try:
            cmd = [vswhere, "-nologo", "-latest", "-products", "*", "-version", "[17.0,19.0)", "-property", "installationPath"]
            out = subprocess.check_output(cmd, text=True, timeout=5).strip()
            if out and Path(out).exists():
                return out
        except Exception: pass
        return None

    def _activate_msvc_env(self, vs_path: str) -> bool:
        vcvars = Path(vs_path) / "VC/Auxiliary/Build/vcvars64.bat"
        if not vcvars.exists(): return False
        
        try:
            cmd = f'@echo off && chcp 65001 >nul && "{vcvars}" >nul 2>&1 && set'
            out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
            for line in out.splitlines():
                if "=" in line:
                    key, val = line.split("=", 1)
                    ukey = key.upper()
                    self.ctx.env[ukey] = val
                    os.environ[ukey] = val
            return True
        except Exception:
            return False

    def _get_msvc_compiler_path(self) -> str | None:
        vctools = self.ctx.env.get("VCTOOLSINSTALLDIR")
        if not vctools: return None
        cl_exe = Path(vctools) / "bin" / "HostX64" / "x64" / "cl.exe"
        if cl_exe.exists(): return str(cl_exe)
        return self._which("cl")

    def _get_compiler_major_version(self, compiler: str) -> int | None:
        for flag in ["-dumpversion", "--version"]:
            try:
                out = subprocess.check_output([compiler, flag], stderr=subprocess.STDOUT, timeout=5, text=True)
                m = re.search(r"(\d+)", out)
                if m: return int(m.group(1))
            except Exception: continue
        return None

    def sanitize_env(self, toolchain: str) -> None:
        self.ctx.env["PYTHONUTF8"] = "1"
        self.ctx.env["PYTHONIOENCODING"] = "utf-8"
        self.ctx.env["PROMPT"] = "$P$G" 
        for key in self.COMPILER_ENV_VARS:
            self.ctx.env.pop(key, None)
        if toolchain in {"clang", "mingw"}:
            for key in self.MSVC_VARS:
                self.ctx.env.pop(key, None)

    def resolve(self) -> tuple[str, list[str]]:
        forced = self.ctx.env.get("LUNAVOX_TOOLCHAIN", "auto").lower()
        order = ["msvc", "mingw", "clang"] if forced == "auto" else [forced]
        
        tools = {
            "cl": self._which("cl"),
            "cmake": self._which("cmake"),
            "g++": self._which("g++"),
            "gcc": self._which("gcc"),
            "clang++": self._which("clang++"),
            "ninja": self._which("ninja"),
            "make": self._which("mingw32-make") or self._which("make"),
        }

        if not tools["cmake"]:
            raise RuntimeError("cmake not found in PATH. Please install CMake.")

        vs_path = self._get_vs_installation_path()

        for ts in order:
            if ts == "msvc" and vs_path:
                self.sanitize_env("msvc")
                if self._activate_msvc_env(vs_path):
                    cl_path = self._get_msvc_compiler_path()
                    if not cl_path: continue
                    args = [f"-DCMAKE_CXX_COMPILER={self._cmake_path(cl_path)}", f"-DCMAKE_C_COMPILER={self._cmake_path(cl_path)}"]
                    if tools["ninja"]:
                        return "msvc", ["-G", "Ninja"] + args
                    return "msvc", ["-G", "Visual Studio 17 2022", "-A", "x64"] + args
            
            if ts == "mingw" and tools["g++"]:
                ver = self._get_compiler_major_version(tools["g++"])
                if ver and ver < 7: continue
                args = [f"-DCMAKE_CXX_COMPILER={self._cmake_path(tools['g++'])}"]
                if tools["make"]: args += ["-G", "MinGW Makefiles", f"-DCMAKE_MAKE_PROGRAM={self._cmake_path(tools['make'])}"]
                elif tools["ninja"]: args += ["-G", "Ninja"]
                self.sanitize_env("mingw")
                return "mingw", args

            if ts == "clang" and tools["clang++"]:
                args = [f"-DCMAKE_CXX_COMPILER={self._cmake_path(tools['clang++'])}"]
                if tools["ninja"]: args += ["-G", "Ninja"]
                elif tools["make"]: args += ["-G", "MinGW Makefiles", f"-DCMAKE_MAKE_PROGRAM={self._cmake_path(tools['make'])}"]
                self.sanitize_env("clang")
                return "clang", args

        raise RuntimeError(f"No supported Windows toolchain found (order: {order}).")

class WindowsBuilder(Builder):
    def post_build(self, portable: bool = False):
        """Windows-only runtime copying."""
        # 1. Bundle metadata.json (Explicit Intent for C++ Engine)
        lib_root = self.ctx.root / "lib"
        metadata_file = lib_root / "metadata.json"
        if metadata_file.exists():
            shutil.copy2(metadata_file, self.ctx.build_dir / "metadata.json")
            print(f"[build] Bundled metadata.json to {self.ctx.build_dir}")

        # 2. Compiler runtimes (always needed for MinGW)
        if self.ctx.toolchain_name == "mingw":
            dlls = ["libstdc++-6.dll", "libgcc_s_seh-1.dll", "libwinpthread-1.dll"]
            for d in self.ctx.env.get("PATH", "").split(";"):
                path = Path(d)
                if not path.exists(): continue
                for dll in dlls:
                    src = path / dll
                    if src.exists() and not (self.ctx.build_dir / dll).exists():
                        shutil.copy2(src, self.ctx.build_dir / dll)
                        print(f"[build] Bundled compiler runtime: {dll}")

        # 3. Portable system dependencies (CUDA, cuDNN)
        if portable:
            print("[build] Searching for system dependencies to bundle (--portable)...")
            patterns = [
                "cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll",
                "cufft64_*.dll", "curand64_*.dll", "cusolver64_*.dll",
                "cusparse64_*.dll", "cudnn*.dll", "zlibwapi.dll"
            ]
            bundle_count = 0
            paths = self.ctx.env.get("PATH", "").split(";")
            seen_dirs = set()
            for d in paths:
                p_dir = Path(d)
                if not p_dir.exists() or p_dir in seen_dirs: continue
                seen_dirs.add(p_dir)
                for pattern in patterns:
                    for src in p_dir.glob(pattern):
                        dest = self.ctx.build_dir / src.name
                        if not dest.exists():
                            try:
                                shutil.copy2(src, dest)
                                print(f"[build] Bundled system dependency: {src.name}")
                                bundle_count += 1
                            except Exception as e:
                                print(f"[build] Warning: failed to copy {src.name}: {e}")

            if bundle_count > 0:
                print(f"[build] Successfully bundled {bundle_count} system dependencies.")
            else:
                print("[build] Warning: --portable was requested but no CUDA/cuDNN DLLs were found in PATH.")
