#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


class Builder:
    def __init__(self, root: Path):
        self.root = root
        self.env = os.environ.copy()
        self.lib_dir = self.root / "lib"
        self.ggml_include_dir = self.root / "third_party" / "ggml" / "include"
        self.tmp_dir = self.root / "_tmp_build"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        # Avoid writing temp files to user profile paths that may be inaccessible in sandboxed sessions.
        self.env["TMP"] = str(self.tmp_dir)
        self.env["TEMP"] = str(self.tmp_dir)
        self.conda_prefix = self._resolve_conda_prefix()
        self._inject_conda_paths()

    def run_cmake(self, args: list[str], cwd: Path) -> None:
        print(f"[build] cmake {' '.join(args)}")
        subprocess.run(["cmake"] + args, cwd=str(cwd), env=self.env, check=True)

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

    def _ensure_import_lib(self, dll_name: str, out_lib_name: str) -> None:
        dll_path = self.lib_dir / dll_name
        if not dll_path.exists():
            raise RuntimeError(f"Required runtime DLL not found: {dll_path}")

        out_lib = self.lib_dir / out_lib_name
        if out_lib.exists() and out_lib.stat().st_mtime >= dll_path.stat().st_mtime:
            return

        gendef = self._which("gendef")
        dlltool = self._which("dlltool")
        if not gendef or not dlltool:
            raise RuntimeError(
                "Missing gendef/dlltool in PATH. Install MinGW toolchain first "
                "(e.g. conda install -n lunavox -c conda-forge m2w64-toolchain)."
            )

        def_file = self.lib_dir / f"{dll_path.stem}.def"
        print(f"[build] Generating import lib: {out_lib.name}")
        subprocess.run([gendef, str(dll_path)], cwd=str(self.lib_dir), env=self.env, check=True)
        if not def_file.exists():
            raise RuntimeError(f"gendef did not produce expected file: {def_file}")

        subprocess.run(
            [
                dlltool,
                "-d",
                str(def_file),
                "-D",
                dll_name,
                "-l",
                str(out_lib),
            ],
            cwd=str(self.lib_dir),
            env=self.env,
            check=True,
        )

    def _prepare_windows_import_libs(self) -> None:
        if platform.system() != "Windows":
            return
        self._ensure_import_lib("ggml-base.dll", "libggml-base.a")
        self._ensure_import_lib("ggml.dll", "libggml.a")

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
            return bool(cl)
        if name == "mingw":
            return bool(gxx and gcc)
        return False

    def _sanitize_toolchain_env(self, toolchain: str) -> None:
        # Conda clang toolchain activation injects linker flags that break MinGW/MSVC builds.
        # Clear those variables before invoking CMake when we are not using clang.
        if toolchain == "clang":
            return

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
        default_order = ["mingw", "clang", "msvc"]
        if forced:
            if forced not in {"mingw", "clang", "msvc", "auto"}:
                raise RuntimeError(
                    "Invalid QWEN3_TTS_TOOLCHAIN value. Expected one of: auto, mingw, clang, msvc."
                )
            order = default_order if forced == "auto" else [forced]
        else:
            order = default_order

        for toolchain in order:
            if not self._toolchain_available(toolchain, clangxx, clang, cl, gxx, gcc):
                continue

            if toolchain == "mingw":
                args: list[str] = []
                if mingw_make:
                    args += ["-G", "MinGW Makefiles", f"-DCMAKE_MAKE_PROGRAM={self._cmake_path(mingw_make)}"]
                elif ninja:
                    args += ["-G", "Ninja"]
                args += [
                    f"-DCMAKE_C_COMPILER={self._cmake_path(gcc)}",
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
                    f"-DCMAKE_C_COMPILER={self._cmake_path(clang)}",
                    f"-DCMAKE_CXX_COMPILER={self._cmake_path(clangxx)}",
                ]
                if windres:
                    args += [f"-DCMAKE_RC_COMPILER={self._cmake_path(windres)}"]
                return (args, toolchain)

            if toolchain == "msvc":
                args: list[str] = ["-G", "Ninja"] if ninja else []
                self._sanitize_toolchain_env(toolchain)
                return (args, toolchain)

        raise RuntimeError("No supported C++ toolchain found (need cl or gcc/g++ in PATH).")

    def build(self, backend: str = "cpu", clean: bool = False, parallel: int = 4) -> None:
        if backend not in {"cpu", "auto"}:
            raise RuntimeError(
                f"Backend '{backend}' is no longer supported. "
                "This build now links prebuilt runtime from ./lib and only uses backend=cpu/auto."
            )
        backend = "cpu"

        if not self.lib_dir.exists():
            raise RuntimeError(f"Prebuilt runtime directory missing: {self.lib_dir}")
        if not self.ggml_include_dir.exists():
            raise RuntimeError(f"GGML headers missing: {self.ggml_include_dir}")

        self._prepare_windows_import_libs()
        toolchain_args, toolchain_name = self._resolve_toolchain()
        print(f"[build] Toolchain: {toolchain_name}")

        build_dir = self.root / "build-cpu"
        if clean and build_dir.exists():
            print(f"[build] Cleaning {build_dir}")
            shutil.rmtree(build_dir)
        build_dir.mkdir(exist_ok=True)

        cmake_args = [
            "-S",
            ".",
            "-B",
            str(build_dir),
            f"-DQWEN3_TTS_PREBUILT_LIB_DIR={self._cmake_path(str(self.lib_dir))}",
            f"-DQWEN3_TTS_GGML_INCLUDE_DIR={self._cmake_path(str(self.ggml_include_dir))}",
        ] + toolchain_args
        self.run_cmake(cmake_args, self.root)

        build_cmd = ["--build", str(build_dir), "-j", str(parallel), "--config", "Release"]
        self.run_cmake(build_cmd, self.root)
        self._copy_windows_runtime_dlls(build_dir)
        print(f"\n[build] Successfully built in {build_dir}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cpu", "auto"], default="auto")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--j", type=int, default=4)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    builder = Builder(root)
    builder.build(backend=args.backend, clean=args.clean, parallel=args.j)


if __name__ == "__main__":
    main()
