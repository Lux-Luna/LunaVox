from __future__ import annotations
import os
import shutil
import subprocess
import time
from pathlib import Path
from .context import BuildContext

class ToolchainResolver:
    """Base class for finding and setting up a C++ toolchain."""
    def __init__(self, context: BuildContext):
        self.ctx = context

    def resolve(self) -> tuple[str, list[str]]:
        """Resolve a toolchain and return (name, cmake_args)."""
        raise NotImplementedError

    def _which(self, program: str) -> str | None:
        return shutil.which(program, path=self.ctx.env.get("PATH"))

    def _cmake_path(self, path: Path | str) -> str:
        return Path(path).as_posix()

class Builder:
    def __init__(self, context: BuildContext, timeout_sec: int = 200):
        self.ctx = context
        self.timeout = max(10, timeout_sec)

    def _run_step(self, cmd: list[str], stage: str, cwd: Path | None = None, timeout: int | None = None) -> None:
        cwd = cwd or self.ctx.root
        t = timeout or self.timeout
        print(f"[build:{stage}] {' '.join(cmd)}")
        
        header = f"\n{'='*80}\nSTAGE: {stage}\nTIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\nCMD: {' '.join(cmd)}\n{'='*80}\n"
        start = time.time()
        try:
            # Note: capturing output but not printing it to stdout unless it fails (legacy behavior of build_manager.py)
            p = subprocess.run(cmd, cwd=str(cwd), env=self.ctx.env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=t)
            output, rc = p.stdout + p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            output, rc = (e.stdout.decode() if e.stdout else "") + (e.stderr.decode() if e.stderr else "") + f"\nTIMEOUT after {t}s", -1

        log_entry = f"{header}STATUS: {'ok' if rc==0 else 'failed'}\nELAPSED: {time.time()-start:.2f}s\nRC: {rc}\n\n{output}\n"
        with open(self.ctx.log_file, "a", encoding="utf-8") as f: f.write(log_entry)
        if rc != 0: raise RuntimeError(f"Stage '{stage}' failed. See {self.ctx.log_file}")

    def post_build(self, portable: bool = False):
        """Platform-specific post-build tasks."""
        pass

    def build(self, resolver: ToolchainResolver, clean: bool = False, parallel: int = 4, verify: bool = False, portable: bool = False):
        # 1. Validation
        if not self.ctx.lib_dir.exists(): raise RuntimeError(f"Llama prebuilts missing at {self.ctx.lib_dir}")
        if not (self.ctx.ort_root / "include/onnxruntime_cxx_api.h").exists():
            raise RuntimeError(f"ONNX SDK missing at {self.ctx.ort_root}")

        # 2. Toolchain Resolution
        name, args = resolver.resolve()
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
        self.post_build(portable=portable)
        
        # 7. Verify
        if verify:
            import platform
            exe_ext = ".exe" if platform.system() == "Windows" else ""
            exe = self.ctx.build_dir / f"qwen3-tts-cli{exe_ext}"
            if not exe.exists(): raise RuntimeError(f"Build finished but executable not found: {exe}")
            self._run_step([str(exe), "--help"], "verify_help", timeout=60)
            print("[build] Verification passed.")

        print(f"\n[build] Successfully completed. Artifacts in: {self.ctx.build_dir}")
