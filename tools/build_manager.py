#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


class Builder:
    def __init__(self, root: Path):
        self.root = root
        self.env = os.environ.copy()
        self.lib_dir = self.root / "lib"
        self.ort_root = self.lib_dir / "onnxruntime"
        self.tmp_dir = self.root / "_tmp_build"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        # Avoid writing temp files to user profile paths that may be inaccessible in sandboxed sessions.
        self.env["TMP"] = str(self.tmp_dir)
        self.env["TEMP"] = str(self.tmp_dir)
        self.conda_prefix = self._resolve_conda_prefix()
        self._vs2022_available_cache: bool | None = None
        self._inject_conda_paths()

    def run_cmake(self, args: list[str], cwd: Path) -> None:
        print(f"[build] cmake {' '.join(args)}")
        subprocess.run(["cmake"] + args, cwd=str(cwd), env=self.env, check=True)

    def run_cmd(self, cmd: list[str], cwd: Path, timeout_sec: int | None = None) -> None:
        print(f"[build] {' '.join(cmd)}")
        subprocess.run(
            cmd,
            cwd=str(cwd),
            env=self.env,
            check=True,
            timeout=timeout_sec,
        )

    def _resolve_conda_prefix(self) -> Path | None:
        # Prefer the currently running interpreter prefix.
        py_prefix = Path(sys.prefix).resolve()
        if py_prefix.exists() and (py_prefix / "python.exe").exists():
            return py_prefix

        env_prefix = self.env.get("CONDA_PREFIX", "").strip()
        if env_prefix:
            p = Path(env_prefix)
            if p.exists():
                return p

        py = Path(sys.executable).resolve()
        if py.name.lower().startswith("python") and py.parent.exists():
            return py.parent
        return None

    def _candidate_tool_dirs(self) -> list[Path]:
        dirs: list[Path] = []
        if self.conda_prefix:
            dirs.extend(
                [
                    self.conda_prefix,
                    self.conda_prefix / "Library" / "bin",
                    self.conda_prefix / "Library" / "mingw-w64" / "bin",
                    self.conda_prefix / "Scripts",
                    self.conda_prefix / "bin",
                ]
            )
        return [d for d in dirs if d.exists()]

    def _inject_conda_paths(self) -> None:
        prepend = [str(p) for p in self._candidate_tool_dirs()]
        if not prepend:
            return
        old_path = self.env.get("PATH", "")
        self.env["PATH"] = ";".join(prepend + [old_path])

    def _which(self, program: str) -> str | None:
        found = shutil.which(program, path=self.env.get("PATH"))
        if found:
            return found

        # Manual fallback for environments where shutil.which misses executable names with '+'.
        exts = [".exe", ".com", ".bat", ".cmd", ""]
        for d in self._candidate_tool_dirs():
            for ext in exts:
                p = d / f"{program}{ext}"
                if p.exists():
                    return str(p)
        return None

    @staticmethod
    def _cmake_path(path_str: str) -> str:
        return Path(path_str).as_posix()

    def _copy_windows_runtime_dlls(self, build_dir: Path) -> None:
        if platform.system() != "Windows":
            return

        candidates: list[Path] = []
        if self.conda_prefix:
            candidates.extend(
                [
                    self.conda_prefix / "Library" / "mingw-w64" / "bin",
                    self.conda_prefix / "Library" / "bin",
                ]
            )
        candidates.extend(self._candidate_tool_dirs())

        runtime_names = ("libstdc++-6.dll", "libgcc_s_seh-1.dll", "libwinpthread-1.dll")
        seen: set[str] = set()
        for dll_name in runtime_names:
            src: Path | None = None
            for d in candidates:
                if not d.exists():
                    continue
                p = d / dll_name
                if p.exists():
                    src = p
                    break
            if src is None:
                continue

            dst = build_dir / dll_name
            if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
                continue
            if str(dst) in seen:
                continue
            print(f"[build] Copying runtime DLL: {dll_name}")
            shutil.copy2(src, dst)
            seen.add(str(dst))

    def _toolchain_available(
        self,
        name: str,
        clangxx: str | None,
        clang: str | None,
        cl: str | None,
        gxx: str | None,
        gcc: str | None,
    ) -> bool:
        if name == "clang":
            return bool(clangxx and clang)
        if name == "msvc":
            return self._has_vs2022_generator()
        if name == "mingw":
            return bool(gxx and gcc)
        return False

    def _has_vs2022_generator(self) -> bool:
        if platform.system() != "Windows":
            return False
        if self._vs2022_available_cache is not None:
            return self._vs2022_available_cache

        vswhere = shutil.which("vswhere", path=self.env.get("PATH"))
        if not vswhere:
            default_vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
            vswhere = str(default_vswhere) if default_vswhere.exists() else None
        if not vswhere:
            self._vs2022_available_cache = False
            return False

        try:
            out = subprocess.check_output(
                [
                    vswhere,
                    "-nologo",
                    "-products",
                    "*",
                    "-version",
                    "[17.0,18.0)",
                    "-property",
                    "installationPath",
                ],
                stderr=subprocess.STDOUT,
                timeout=5,
                text=True,
            ).strip()
            self._vs2022_available_cache = bool(out)
        except Exception:
            self._vs2022_available_cache = False
        return self._vs2022_available_cache

    def _sanitize_toolchain_env(self, toolchain: str) -> None:
        # Conda activation may inject compiler/linker flags that force lld-link
        # or non-portable defaults; clear them before configuring any toolchain.
        for key in (
            "CC",
            "CXX",
            "AR",
            "LD",
            "NM",
            "RANLIB",
            "CPPFLAGS",
            "CPPFLAGS_USED",
            "CFLAGS",
            "CXXFLAGS",
            "LDFLAGS",
        ):
            self.env.pop(key, None)

        if toolchain in {"clang", "mingw"}:
            # `conda run` may export MSVC/CMake variables that force clang into
            # an MSVC-oriented link mode (lld-link + kernel32.lib lookup),
            # which breaks MinGW-style builds.
            for key in (
                "INCLUDE",
                "LIB",
                "LIBPATH",
                "VSINSTALLDIR",
                "VCINSTALLDIR",
                "VCToolsInstallDir",
                "VisualStudioVersion",
                "WindowsSdkDir",
                "WindowsSDKLibVersion",
                "WindowsSDKVersion",
                "UniversalCRTSdkDir",
                "UCRTVersion",
                "VSCMD_ARG_TGT_ARCH",
                "VSCMD_ARG_HOST_ARCH",
                "DISTUTILS_USE_SDK",
                "MSSdk",
                "CMAKE_GENERATOR",
                "CMAKE_GENERATOR_PLATFORM",
                "CMAKE_GENERATOR_TOOLSET",
            ):
                self.env.pop(key, None)

    def _compiler_major_version(self, compiler: str) -> int | None:
        try:
            out = subprocess.check_output(
                [compiler, "-dumpversion"],
                stderr=subprocess.STDOUT,
                timeout=5,
                text=True,
            ).strip()
            m = re.match(r"(\d+)", out)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        try:
            out = subprocess.check_output(
                [compiler, "--version"],
                stderr=subprocess.STDOUT,
                timeout=5,
                text=True,
            )
            m = re.search(r"\b(\d+)\.\d+(\.\d+)?\b", out)
            if m:
                return int(m.group(1))
        except Exception:
            return None
        return None

    def _resolve_toolchain(self) -> tuple[list[str], str]:
        if platform.system() != "Windows":
            return ([], "default")

        clangxx = self._which("clang++")
        clang = self._which("clang")
        cl = self._which("cl")
        gxx = self._which("g++")
        gcc = self._which("gcc")
        ninja = self._which("ninja")
        mingw_make = self._which("mingw32-make")
        windres = self._which("windres")

        forced = self.env.get("QWEN3_TTS_TOOLCHAIN", "").strip().lower()
        # Prefer MSVC first for C++17 headers (e.g. <variant>) and robust
        # Windows SDK linkage in conda environments.
        default_order = ["msvc", "mingw", "clang"]
        if forced:
            if forced not in {"mingw", "clang", "msvc", "auto"}:
                raise RuntimeError(
                    "Invalid QWEN3_TTS_TOOLCHAIN value. Expected one of: auto, mingw, clang, msvc."
                )
            order = default_order if forced == "auto" else [forced]
        else:
            order = default_order

        mingw_version_ok = True
        mingw_major = self._compiler_major_version(gxx) if gxx else None
        if mingw_major is not None and mingw_major < 7:
            mingw_version_ok = False

        for toolchain in order:
            if not self._toolchain_available(toolchain, clangxx, clang, cl, gxx, gcc):
                continue
            if toolchain == "mingw" and not mingw_version_ok:
                print(
                    f"[build] Skipping MinGW toolchain: g++ version {mingw_major} is too old for C++17 <variant>."
                )
                continue

            if toolchain == "mingw":
                args: list[str] = []
                if mingw_make:
                    args += ["-G", "MinGW Makefiles", f"-DCMAKE_MAKE_PROGRAM={self._cmake_path(mingw_make)}"]
                elif ninja:
                    args += ["-G", "Ninja"]
                args += [
                    f"-DCMAKE_CXX_COMPILER={self._cmake_path(gxx)}",
                ]
                if windres:
                    args += [f"-DCMAKE_RC_COMPILER={self._cmake_path(windres)}"]
                self._sanitize_toolchain_env(toolchain)
                return (args, toolchain)

            if toolchain == "clang":
                args: list[str] = []
                if mingw_make:
                    args += ["-G", "MinGW Makefiles", f"-DCMAKE_MAKE_PROGRAM={self._cmake_path(mingw_make)}"]
                elif ninja:
                    args += ["-G", "Ninja"]
                args += [
                    f"-DCMAKE_CXX_COMPILER={self._cmake_path(clangxx)}",
                ]
                if windres:
                    args += [f"-DCMAKE_RC_COMPILER={self._cmake_path(windres)}"]
                self._sanitize_toolchain_env(toolchain)
                return (args, toolchain)

            if toolchain == "msvc":
                # Prefer Visual Studio generator so CMake can discover MSVC toolchain
                # without requiring cl.exe to be pre-initialized in current shell.
                args: list[str] = ["-G", "Visual Studio 17 2022", "-A", "x64"]
                self._sanitize_toolchain_env(toolchain)
                return (args, toolchain)

        raise RuntimeError("No supported C++ toolchain found (need cl or gcc/g++ in PATH).")

    def verify(self, build_dir: Path) -> None:
        exe = build_dir / ("qwen3-tts-cli.exe" if platform.system() == "Windows" else "qwen3-tts-cli")
        if not exe.exists():
            raise RuntimeError(f"Verify failed: executable not found: {exe}")
        self.run_cmd([str(exe), "--help"], cwd=self.root, timeout_sec=60)
        print("[build] Verify passed: qwen3-tts-cli --help")

    def build(self, backend: str = "cpu", clean: bool = False, parallel: int = 4, verify: bool = False) -> None:
        if backend not in {"cpu", "auto"}:
            raise RuntimeError(
                f"Backend '{backend}' is no longer supported. "
                "This build now links prebuilt runtime from ./lib and only uses backend=cpu/auto."
            )
        backend = "cpu"

        if not self.lib_dir.exists():
            raise RuntimeError(f"Prebuilt runtime directory missing: {self.lib_dir}")
        if not self.ort_root.exists():
            raise RuntimeError(
                f"ONNX Runtime SDK missing: {self.ort_root}\n"
                "Expected include/ and lib/ under this directory."
            )
        if not (self.ort_root / "include" / "onnxruntime_cxx_api.h").exists():
            raise RuntimeError(f"ONNX Runtime headers missing under: {self.ort_root / 'include'}")
        toolchain_args, toolchain_name = self._resolve_toolchain()
        print(f"[build] Toolchain: {toolchain_name}")

        build_dir = self.root / "build-cpu"
        if clean and build_dir.exists():
            print(f"[build] Cleaning {build_dir}")
            shutil.rmtree(build_dir)
        build_dir.mkdir(exist_ok=True)

        desired_generator: str | None = None
        if "-G" in toolchain_args:
            g_idx = toolchain_args.index("-G")
            if g_idx + 1 < len(toolchain_args):
                desired_generator = toolchain_args[g_idx + 1]

        if desired_generator:
            cache_file = build_dir / "CMakeCache.txt"
            if cache_file.exists():
                cached_generator: str | None = None
                for line in cache_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.startswith("CMAKE_GENERATOR:INTERNAL="):
                        cached_generator = line.split("=", 1)[1].strip()
                        break
                if cached_generator and cached_generator != desired_generator:
                    print(
                        "[build] Generator changed "
                        f"({cached_generator} -> {desired_generator}), recreating {build_dir}"
                    )
                    shutil.rmtree(build_dir)
                    build_dir.mkdir(exist_ok=True)

        cmake_args = [
            "-S",
            ".",
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DQWEN3_TTS_PREBUILT_LIB_DIR={self._cmake_path(str(self.lib_dir))}",
            f"-DQWEN3_TTS_ORT_ROOT={self._cmake_path(str(self.ort_root))}",
        ] + toolchain_args
        self.run_cmake(cmake_args, self.root)

        build_cmd = ["--build", str(build_dir), "-j", str(parallel), "--config", "Release"]
        self.run_cmake(build_cmd, self.root)
        self._copy_windows_runtime_dlls(build_dir)
        if verify:
            self.verify(build_dir)
        print(f"\n[build] Successfully built in {build_dir}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cpu", "auto"], default="auto")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--j", type=int, default=4)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    builder = Builder(root)
    builder.build(backend=args.backend, clean=args.clean, parallel=args.j, verify=args.verify)


if __name__ == "__main__":
    main()
