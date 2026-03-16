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
            output = subprocess.check_output(cmd, shell=True, encoding="utf-8", stderr=subprocess.STDOUT)
            new_env = {}
            for line in output.splitlines():
                if "=" in line:
                    key, val = line.split("=", 1)
                    new_env[key] = val
            return new_env
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
        print(f"[build] cmake {' '.join(args)}")
        subprocess.run(["cmake"] + args, cwd=str(cwd), env=self.env, check=True)

    def build(self, backend: str = "cpu", clean: bool = False, parallel: int = 4):
        build_dir = self.root / f"build-{backend}"
        
        if clean and build_dir.exists():
            print(f"[build] Cleaning {build_dir}")
            shutil.rmtree(build_dir)
        
        build_dir.mkdir(exist_ok=True)
        
        # 1. Configure
        cmake_args = ["-S", ".", "-B", str(build_dir)]
        
        # Generator preference
        if platform.system() == "Windows":
            if shutil.which("ninja"):
                cmake_args += ["-G", "Ninja"]
        
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
        build_cmd = ["--build", str(build_dir), "-j", str(parallel), "--config", "Release"]
        self.run_cmake(build_cmd, self.root)
        
        print(f"\n[build] Successfully built {backend} version in {build_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cpu", "cuda", "metal", "coreml", "auto"], default="auto")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--j", type=int, default=4)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    
    backend = args.backend
    if backend == "auto":
        if platform.system() == "Darwin":
            backend = "metal"
        else:
            backend = "cpu" # Default to CPU for now
            
    builder = Builder(root)
    builder.build(backend=backend, clean=args.clean, parallel=args.j)

if __name__ == "__main__":
    main()
