#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BuildContext:
    root: Path
    env: dict[str, str]
    toolchain_name: str = "unknown"
    toolchain_args: list[str] = field(default_factory=list)
    lib_dir: Path = field(init=False)
    ort_root: Path = field(init=False)
    build_dir: Path = field(init=False)
    log_file: Path = field(init=False)

    def __post_init__(self):
        self.lib_dir = self.root / "lib" / "llama"
        self.ort_root = self.root / "lib" / "onnx"
        self.build_dir = self.root / "build"
        log_dir = self.root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / "latest.log"


class ToolchainResolver:
    """Handles the complex task of finding and setting up a C++ toolchain."""
    
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

    def __init__(self, context: BuildContext):
        self.ctx = context
        self._vs_path_cache: str | None = None

    def _which(self, program: str) -> str | None:
        return shutil.which(program, path=self.ctx.env.get("PATH"))

    def _cmake_path(self, path: Path | str) -> str:
        return Path(path).as_posix()

    def _get_vs_installation_path(self) -> str | None:
        if platform.system() != "Windows": return None
        if self._vs_path_cache is not None: return self._vs_path_cache
        
        vswhere = self._which("vswhere") or str(Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"))
        if not Path(vswhere).exists(): return None

        try:
            # Search for VS 2022 and VS 2025+
            cmd = [vswhere, "-nologo", "-latest", "-products", "*", "-version", "[17.0,19.0)", "-property", "installationPath"]
            out = subprocess.check_output(cmd, text=True, timeout=5).strip()
            if out and Path(out).exists():
                self._vs_path_cache = out
                return out
        except Exception: pass
        return None

    def _activate_msvc_env(self, vs_path: str) -> bool:
        """Sourcing vcvars64.bat to bring MSVC tools into the environment."""
        vcvars = Path(vs_path) / "VC/Auxiliary/Build/vcvars64.bat"
        if not vcvars.exists(): return False
        
        try:
            # Silent execution: @echo off and nul redirection to suppress noisy echoes and path errors from internal VS scripts
            cmd = f'@echo off && chcp 65001 >nul && "{vcvars}" >nul 2>&1 && set'
            out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
            for line in out.splitlines():
                if "=" in line:
                    key, val = line.split("=", 1)
                    # Windows env vars are case-insensitive, harmonizing to uppercase for consistency in our dict
                    ukey = key.upper()
                    self.ctx.env[ukey] = val
                    os.environ[ukey] = val
            return True
        except Exception:
            # We fail silently and let resolve() log the fallback or handle the error
            return False

    def _get_msvc_compiler_path(self) -> str | None:
        """After activation, find the absolute path to cl.exe."""
        vctools = self.ctx.env.get("VCTOOLSINSTALLDIR")
        if not vctools: return None
        
        # Standard layout: bin/HostX64/x64/cl.exe
        cl_exe = Path(vctools) / "bin" / "HostX64" / "x64" / "cl.exe"
        if cl_exe.exists(): return str(cl_exe)
        
        # Fallback to simple which if the above layout fails
        return self._which("cl")

    def _has_vs2022(self) -> bool:
        return self._get_vs_installation_path() is not None

    def _get_compiler_major_version(self, compiler: str) -> int | None:
        for flag in ["-dumpversion", "--version"]:
            try:
                out = subprocess.check_output([compiler, flag], stderr=subprocess.STDOUT, timeout=5, text=True)
                m = re.search(r"(\d+)", out)
                if m: return int(m.group(1))
            except Exception: continue
        return None

    def sanitize_env(self, toolchain: str) -> None:
        """Removes environment variables that might interfere with the selected toolchain."""
        # Fix encoding and suppress echoing in Windows subshells
        self.ctx.env["PYTHONUTF8"] = "1"
        self.ctx.env["PYTHONIOENCODING"] = "utf-8"
        self.ctx.env["PROMPT"] = "$P$G" 
        
        for key in self.COMPILER_ENV_VARS:
            self.ctx.env.pop(key, None)
        if toolchain in {"clang", "mingw"}:
            for key in self.MSVC_VARS:
                self.ctx.env.pop(key, None)

    def resolve(self) -> tuple[str, list[str]]:
        if platform.system() != "Windows":
            return "default", []

        forced = self.ctx.env.get("LUNAVOX_TOOLCHAIN", "auto").lower()
        order = ["msvc", "mingw", "clang"] if forced == "auto" else [forced]
        
        # Tools availability
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
                    if not cl_path:
                        self.ctx.env["_MSVC_ERROR"] = "cl.exe not found after activation"
                        continue
                        
                    # We MUST explicitly tell CMake to use the absolute cl.exe path, 
                    # otherwise a broken clang in PATH might hijack it or it might not be found.
                    args = [f"-DCMAKE_CXX_COMPILER={self._cmake_path(cl_path)}", f"-DCMAKE_C_COMPILER={self._cmake_path(cl_path)}"]
                    if tools["ninja"]:
                        return "msvc", ["-G", "Ninja"] + args
                    return "msvc", ["-G", "Visual Studio 17 2022", "-A", "x64"] + args
                else:
                    self.ctx.env["_MSVC_ERROR"] = "Activation failed"
            
            if ts == "mingw" and tools["g++"]:
                ver = self._get_compiler_major_version(tools["g++"])
                if ver and ver < 7:
                    # Explicitly skip old MinGW that doesn't support C++17
                    continue
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

        raise RuntimeError(f"No supported C++ toolchain found (order: {order}). Found VS Path: {vs_path or 'None'}")


class Builder:
    def __init__(self, root: Path, timeout_sec: int = 200):
        # Initial environment setup
        env = os.environ.copy()
        py_prefix = Path(sys.prefix).resolve()
        if (py_prefix / "python.exe").exists():
            # Injecting Conda/Python paths early
            candidates = [py_prefix, py_prefix / "Library/bin", py_prefix / "Library/mingw-w64/bin", py_prefix / "Scripts", py_prefix / "bin"]
            valid_dirs = [str(d) for d in candidates if d.exists()]
            if valid_dirs:
                env["PATH"] = ";".join(valid_dirs + [env.get("PATH", "")])
            env["TMP"] = env["TEMP"] = str(root / "_tmp_build")
            (root / "_tmp_build").mkdir(parents=True, exist_ok=True)

        self.ctx = BuildContext(root, env)
        self.timeout = max(10, timeout_sec)
        self.resolver = ToolchainResolver(self.ctx)

    def _run_step(self, cmd: list[str], stage: str, cwd: Path | None = None, timeout: int | None = None) -> None:
        cwd = cwd or self.ctx.root
        t = timeout or self.timeout
        print(f"[build:{stage}] {' '.join(cmd)}")
        
        header = f"\n{'='*80}\nSTAGE: {stage}\nTIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\nCMD: {' '.join(cmd)}\n{'='*80}\n"
        start = time.time()
        try:
            p = subprocess.run(cmd, cwd=str(cwd), env=self.ctx.env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=t)
            output, rc = p.stdout + p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            output, rc = (e.stdout.decode() if e.stdout else "") + (e.stderr.decode() if e.stderr else "") + f"\nTIMEOUT after {t}s", -1

        log_entry = f"{header}STATUS: {'ok' if rc==0 else 'failed'}\nELAPSED: {time.time()-start:.2f}s\nRC: {rc}\n\n{output}\n"
        with open(self.ctx.log_file, "a", encoding="utf-8") as f: f.write(log_entry)
        if rc != 0: raise RuntimeError(f"Stage '{stage}' failed. See {self.ctx.log_file}")

    def _copy_runtimes(self):
        """Windows-only: Copy compiler runtime DLLs (libstdc++, etc.) if using MinGW."""
        if platform.system() != "Windows" or self.ctx.toolchain_name != "mingw": return
        dlls = ["libstdc++-6.dll", "libgcc_s_seh-1.dll", "libwinpthread-1.dll"]
        found_any = False
        for d in self.ctx.env.get("PATH", "").split(";"):
            path = Path(d)
            if not path.exists(): continue
            for dll in dlls:
                src = path / dll
                if src.exists():
                    shutil.copy2(src, self.ctx.build_dir / dll)
                    found_any = True
        if found_any: print("[build] Copied compiler runtime DLLs to build directory.")

    def build(self, clean: bool = False, parallel: int = 4, verify: bool = False):
        # 1. Validation
        if not self.ctx.lib_dir.exists(): raise RuntimeError(f"Llama prebuilts missing at {self.ctx.lib_dir}")
        if not (self.ctx.ort_root / "include/onnxruntime_cxx_api.h").exists():
            raise RuntimeError(f"ONNX SDK missing at {self.ctx.ort_root}")

        # 2. Toolchain Resolution
        name, args = self.resolver.resolve()
        self.ctx.toolchain_name, self.ctx.toolchain_args = name, args
        print(f"[build] Toolchain: {name}")

        # 3. Preparation
        if clean and self.ctx.build_dir.exists(): shutil.rmtree(self.ctx.build_dir)
        self.ctx.build_dir.mkdir(exist_ok=True)

        # 4. Configure
        cmake_args = [
            "cmake", "-S", ".", "-B", str(self.ctx.build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DQWEN3_TTS_PREBUILT_LIB_DIR={self.ctx.lib_dir.as_posix()}",
            f"-DQWEN3_TTS_ORT_ROOT={self.ctx.ort_root.as_posix()}",
        ] + args
        self._run_step(cmake_args, "cmake_configure")

        # 5. Build
        self._run_step(["cmake", "--build", str(self.ctx.build_dir), "-j", str(parallel), "--config", "Release"], "cmake_build")
        
        # 6. Post-build
        self._copy_runtimes()
        
        # 7. Verify
        if verify:
            exe = self.ctx.build_dir / ("qwen3-tts-cli.exe" if platform.system() == "Windows" else "qwen3-tts-cli")
            if not exe.exists(): raise RuntimeError("Build finished but executable not found.")
            self._run_step([str(exe), "--help"], "verify_help", timeout=60)
            print("[build] Verification passed.")

        print(f"\n[build] Successfully completed. Artifacts in: {self.ctx.build_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LunaVox Build Manager")
    parser.add_argument("--clean", action="store_true", help="Clean build directory first")
    parser.add_argument("--j", type=int, default=4, help="Parallel build jobs")
    parser.add_argument("--timeout-sec", type=int, default=200, help="Stage timeout")
    parser.add_argument("--verify", action="store_true", help="Verify build with --help test")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    builder = Builder(root, timeout_sec=args.timeout_sec)
    try:
        builder.build(clean=args.clean, parallel=args.j, verify=args.verify)
    except Exception as e:
        print(f"\n[ERROR] Build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
