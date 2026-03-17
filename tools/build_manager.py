#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import platform
import uuid
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
        # Run vcvars and then print the environment.
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
            path_from_vcvars = new_env.get("PATH") or new_env.get("Path")
            if path_from_vcvars:
                merged_env["PATH"] = path_from_vcvars
                merged_env["Path"] = path_from_vcvars
            else:
                path_any = merged_env.get("PATH") or merged_env.get("Path")
                if path_any:
                    merged_env["PATH"] = path_any
                    merged_env["Path"] = path_any

            return merged_env
        except Exception as e:
            print(f"[warn] Failed to capture MSVC env: {e}")
            return os.environ.copy()

class Builder:
    def __init__(self, root: Path):
        self.root = root
        self.env = os.environ.copy()
        self._ninja_healthy_cache: Optional[bool] = None
        self._nmake_healthy_cache: Optional[bool] = None
        if platform.system() == "Windows":
            self.env = MSVCEnv.capture_env()

    def run_cmake(self, args: List[str], cwd: Path):
        print(f"[build] cmake {' '.join(args)}", flush=True)
        subprocess.run(["cmake"] + args, cwd=str(cwd), env=self.env, check=True)

    def _which(self, exe: str) -> Optional[str]:
        path = self.env.get("PATH") or self.env.get("Path")
        return shutil.which(exe, path=path)

    def _run_quick(self, cmd: List[str], timeout: int = 8) -> bool:
        try:
            cp = subprocess.run(
                cmd,
                env=self.env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
            return cp.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _nmake_healthy(self) -> bool:
        if self._nmake_healthy_cache is not None:
            return self._nmake_healthy_cache

        nmake = self._which("nmake")
        cl = self._which("cl")
        if not nmake or not cl:
            self._nmake_healthy_cache = False
            return False

        ok = self._run_quick([nmake, "-?"], timeout=8)
        self._nmake_healthy_cache = ok
        if not ok:
            print("[warn] NMake probe failed; will avoid NMake for this build session", flush=True)
        return ok

    def _ninja_healthy(self) -> bool:
        if self._ninja_healthy_cache is not None:
            return self._ninja_healthy_cache

        ninja = self._which("ninja")
        cmake = self._which("cmake")
        if not ninja or not cmake:
            self._ninja_healthy_cache = False
            return False

        # Probe with a tiny build to catch environments where Ninja hangs
        # when launching real build commands.
        probe_dir: Optional[Path] = None
        probe_base = self.root / ".build-probes"
        try:
            probe_base.mkdir(exist_ok=True)
            probe_dir = probe_base / f"ninja_probe_{uuid.uuid4().hex}"
            probe_dir.mkdir(parents=False, exist_ok=False)
            stamp = probe_dir / "stamp.txt"
            ninja_file = probe_dir / "build.ninja"
            ninja_file.write_text(
                "\n".join([
                    "rule touch",
                    f"  command = \"{cmake}\" -E touch stamp.txt",
                    "build stamp.txt: touch",
                    "default stamp.txt",
                    "",
                ]),
                encoding="utf-8",
            )

            cp = subprocess.run(
                [ninja, "-C", str(probe_dir)],
                cwd=str(self.root),
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=8,
                text=True,
                errors="ignore",
            )
            ok = cp.returncode == 0 and stamp.exists()
            self._ninja_healthy_cache = ok
            if not ok:
                print("[warn] Ninja probe failed; will avoid Ninja for this build session", flush=True)
            return ok
        except subprocess.TimeoutExpired:
            print("[warn] Ninja probe timed out; will avoid Ninja for this build session", flush=True)
            self._ninja_healthy_cache = False
            return False
        except Exception as e:
            print(f"[warn] Ninja probe error ({e}); will avoid Ninja for this build session", flush=True)
            self._ninja_healthy_cache = False
            return False
        finally:
            if probe_dir is not None:
                shutil.rmtree(probe_dir, ignore_errors=True)

    def _generator_healthy(self, generator: Optional[str]) -> bool:
        if not generator:
            return True
        if platform.system() != "Windows":
            return True

        g = generator.lower()
        if "nmake" in g:
            return self._nmake_healthy()
        if "ninja" in g:
            return self._ninja_healthy()
        return True

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
        user_forced_generator = generator is not None
        
        if clean and build_dir.exists():
            print(f"[build] Cleaning {build_dir}", flush=True)
            shutil.rmtree(build_dir)
        
        build_dir.mkdir(exist_ok=True)
        
        # 1. Configure
        cmake_args = ["-S", ".", "-B", str(build_dir)]
        chosen_generator: Optional[str] = generator
        existing_generator = self._read_cache_entry(build_dir / "CMakeCache.txt", "CMAKE_GENERATOR")
        if existing_generator and not user_forced_generator:
            chosen_generator = existing_generator
            print(f"[build] Using existing generator from build tree: {chosen_generator}", flush=True)

        if not ggml_build_dir:
            ggml_candidates = [
                self.root / "ggml" / f"build_{backend}",
                self.root / "ggml" / f"build-{backend}",
            ]
            for cand in ggml_candidates:
                if (cand / "CMakeCache.txt").exists():
                    ggml_build_dir = str(cand)
                    print(f"[build] Auto-detected GGML build tree: {cand}", flush=True)
                    break

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

        # Generator preference and health checks
        if platform.system() == "Windows":
            if user_forced_generator and chosen_generator and not self._generator_healthy(chosen_generator):
                print(f"[error] Requested generator is unavailable in current environment: {chosen_generator}", flush=True)
                print("[error] Ensure MSVC tools are installed and available, or choose another generator.", flush=True)
                sys.exit(2)

            if not user_forced_generator and chosen_generator and not self._generator_healthy(chosen_generator):
                print(f"[warn] Generator from cache/build tree is unavailable: {chosen_generator}", flush=True)
                chosen_generator = None

            if not chosen_generator:
                if self._nmake_healthy():
                    chosen_generator = "NMake Makefiles"
                    print("[build] Selected stable Windows generator: NMake Makefiles", flush=True)
                elif self._ninja_healthy():
                    chosen_generator = "Ninja"
                    print("[build] Selected generator: Ninja", flush=True)
                else:
                    print("[warn] Neither NMake nor Ninja passed health checks; delegating generator selection to CMake", flush=True)

        if existing_generator and chosen_generator and existing_generator != chosen_generator:
            print(
                f"[build] Existing build dir generator is '{existing_generator}', "
                f"switching to '{chosen_generator}' requires regenerating build dir",
                flush=True,
            )
            shutil.rmtree(build_dir)
            build_dir.mkdir(exist_ok=True)
            cmake_args = ["-S", ".", "-B", str(build_dir)]

        if chosen_generator:
            cmake_args += ["-G", chosen_generator]
        use_multi_config = self._is_multi_config_generator(chosen_generator)

        if build_type and (chosen_generator is None or not use_multi_config):
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

        # Refresh generator type from the configured build tree to avoid
        # wrong assumptions when CMake auto-selects a generator.
        configured_generator = self._read_cache_entry(build_dir / "CMakeCache.txt", "CMAKE_GENERATOR")
        if configured_generator:
            use_multi_config = self._is_multi_config_generator(configured_generator)
            chosen_generator = configured_generator
            print(f"[build] Configured generator: {configured_generator}", flush=True)
        
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
