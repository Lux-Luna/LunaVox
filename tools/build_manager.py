#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path
from typing import Dict, List, Optional

class MSVCEnv:
    """Detect and capture Visual Studio environment variables."""
    @staticmethod
    def get_vcvars_path() -> Optional[Path]:
        if platform.system() != "Windows":
            return None
        
        # Try using vswhere if available
        vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
        if vswhere.exists():
            try:
                res = subprocess.check_output([
                    str(vswhere), "-latest", "-products", "*", 
                    "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property", "installationPath"
                ], encoding="utf-8").strip()
                if res:
                    vcvars = Path(res) / "VC/Auxiliary/Build/vcvars64.bat"
                    if vcvars.exists():
                        return vcvars
            except Exception:
                pass
        
        # Fallback common paths
        common_paths = [
            "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat",
            "C:/Program Files/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat",
            "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/vcvars64.bat",
            "C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Auxiliary/Build/vcvars64.bat",
        ]
        for p in common_paths:
            path = Path(p)
            if path.exists():
                return path
        return None

    @staticmethod
    def capture_env() -> Dict[str, str]:
        vcvars = MSVCEnv.get_vcvars_path()
        if not vcvars:
            return os.environ.copy()
        
        print(f"[build] Using MSVC environment from: {vcvars}")
        # Run vcvars and then print the environment
        cmd = f'call "{vcvars}" && set'
        try:
            # vcvarsall.bat often outputs in system locale (e.g., GBK on Chinese Windows)
            # Use errors="ignore" or "replace" to avoid crashing on decode
            output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode(errors="ignore")
            new_env = {}
            for line in output.splitlines():
                if "=" in line:
                    try:
                        key, val = line.split("=", 1)
                        new_env[key] = val
                    except ValueError:
                        continue
            merged_env = os.environ.copy()
            merged_env.update(new_env)

            # Windows env keys are case-insensitive, but subprocess maps are case-sensitive.
            # Keep both PATH/Path synchronized so tool discovery (cl, cmake, ninja) is stable.
            if "PATH" in merged_env:
                merged_env["Path"] = merged_env["PATH"]
            elif "Path" in merged_env:
                merged_env["PATH"] = merged_env["Path"]

            return merged_env
        except Exception as e:
            print(f"[warn] Failed to capture MSVC env: {e}")
            return os.environ.copy()

class Builder:
    def __init__(self, root: Path):
        self.root = root
        self.env = os.environ.copy()
        if platform.system() == "Windows":
            self.env = MSVCEnv.capture_env()

    def run_cmake(self, args: List[str], cwd: Path):
        print(f"[build] cmake {' '.join(args)}", flush=True)
        subprocess.run(["cmake"] + args, cwd=str(cwd), env=self.env, check=True)

    @staticmethod
    def _is_multi_config_generator(generator: Optional[str]) -> bool:
        if not generator:
            return False
        generator_l = generator.lower()
        return (
            "visual studio" in generator_l or
            "xcode" in generator_l or
            "multi-config" in generator_l
        )

    @staticmethod
    def _read_cache_entry(cache_file: Path, key: str) -> Optional[str]:
        if not cache_file.exists():
            return None
        prefix = f"{key}:"
        with cache_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("//") or line.startswith("#"):
                    continue
                if line.startswith(prefix) and "=" in line:
                    return line.split("=", 1)[1]
        return None

    def _resolve_ggml_build_dir(self, ggml_build_dir: Optional[str]) -> Optional[Path]:
        if not ggml_build_dir:
            return None

        path = Path(ggml_build_dir).expanduser()
        if not path.is_absolute():
            path = (self.root / path).resolve()

        cache = path / "CMakeCache.txt"
        if not cache.exists():
            print(f"[error] --ggml-build-dir is not a configured CMake build tree: {path}")
            sys.exit(2)

        return path

    def build(self,
              backend: str = "cpu",
              clean: bool = False,
              parallel: int = 4,
              build_type: str = "Release",
              ggml_build_dir: Optional[str] = None,
              generator: Optional[str] = None):
        build_dir = self.root / f"build-{backend}"
        
        if clean and build_dir.exists():
            print(f"[build] Cleaning {build_dir}", flush=True)
            shutil.rmtree(build_dir)
        
        build_dir.mkdir(exist_ok=True)
        
        # 1. Configure
        cmake_args = ["-S", ".", "-B", str(build_dir)]
        chosen_generator: Optional[str] = generator

        resolved_ggml_build_dir = self._resolve_ggml_build_dir(ggml_build_dir)
        ggml_cache_generator = None
        if resolved_ggml_build_dir is not None:
            ggml_cache_generator = self._read_cache_entry(
                resolved_ggml_build_dir / "CMakeCache.txt",
                "CMAKE_GENERATOR",
            )
            if not chosen_generator and ggml_cache_generator:
                chosen_generator = ggml_cache_generator
                print(
                    f"[build] Using generator from GGML build tree: {chosen_generator}",
                    flush=True,
                )
        
        # Generator preference
        if not chosen_generator and platform.system() == "Windows":
            if shutil.which("ninja"):
                chosen_generator = "Ninja"
        if chosen_generator:
            cmake_args += ["-G", chosen_generator]
        use_multi_config = self._is_multi_config_generator(chosen_generator)
        if chosen_generator is None and platform.system() == "Windows":
            # Letting CMake choose on Windows typically means a Visual Studio multi-config generator.
            use_multi_config = True

        if build_type and not use_multi_config:
            cmake_args += [f"-DCMAKE_BUILD_TYPE={build_type}"]

        if resolved_ggml_build_dir is not None:
            cmake_args += [f"-DGGML_BUILD_DIR={resolved_ggml_build_dir}"]
            print(f"[build] Reusing GGML build tree: {resolved_ggml_build_dir}", flush=True)
        
        # Backend specific flags
        if backend == "cuda":
            cmake_args += ["-DGGML_CUDA=ON"]
        elif backend == "metal" and platform.system() == "Darwin":
            cmake_args += ["-DGGML_METAL=ON"]
        elif backend == "coreml" and platform.system() == "Darwin":
            cmake_args += ["-DGGML_METAL=ON", "-DQWEN3_TTS_COREML=ON"]
        elif backend == "coreml" or backend == "metal":
            if platform.system() != "Darwin":
                print(f"[error] {backend} is only supported on macOS/Apple platforms")
                sys.exit(1)
        
        self.run_cmake(cmake_args, self.root)
        
        # 2. Build
        build_cmd = ["--build", str(build_dir)]
        if parallel and not (chosen_generator and "nmake" in chosen_generator.lower()):
            build_cmd += ["-j", str(parallel)]
        if use_multi_config and build_type:
            build_cmd += ["--config", build_type]
        self.run_cmake(build_cmd, self.root)
        
        print(f"\n[build] Successfully built {backend} version in {build_dir}", flush=True)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cpu", "cuda", "metal", "coreml", "auto"], default="auto")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--j", type=int, default=4)
    parser.add_argument("--build-type",
                        choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
                        default="Release")
    parser.add_argument("--ggml-build-dir",
                        help="Reuse an existing configured GGML build tree (passes -DGGML_BUILD_DIR=...)")
    parser.add_argument("--generator",
                        help="Override CMake generator (e.g. Ninja, NMake Makefiles, Visual Studio 17 2022)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    
    backend = args.backend
    if backend == "auto":
        if platform.system() == "Darwin":
            backend = "metal"
        else:
            backend = "cpu" # Default to CPU for now
            
    builder = Builder(root)
    builder.build(
        backend=backend,
        clean=args.clean,
        parallel=args.j,
        build_type=args.build_type,
        ggml_build_dir=args.ggml_build_dir,
        generator=args.generator,
    )

if __name__ == "__main__":
    main()
