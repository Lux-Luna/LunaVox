import os
import shutil
import subprocess
import time
import json
import difflib
import platform as py_platform
from pathlib import Path
from rich.panel import Panel
from rich.console import Console
from .context import BuildContext

console = Console()

VALID_PROVIDERS = [
    "CPUExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider",
    "ROCmExecutionProvider", "VulkanExecutionProvider", 
    "OpenVINOExecutionProvider", "CoreMLExecutionProvider"
]

VALID_BACKENDS = ["cpu", "cuda", "vulkan", "metal", "rocm", "sycl"]

# Handled by LunaVox C++ wrapper
SUPPORTED_PROVIDERS = [
    "CPUExecutionProvider", "CUDAExecutionProvider", 
    "DmlExecutionProvider", "ROCmExecutionProvider", "CoreMLExecutionProvider"
]

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
        self.verbose = self.ctx.env.get("LUNAVOX_BUILD_VERBOSE", "").strip() == "1"
        self.target_details = {"onnx": "", "llama": ""}

    def _run_step(self, cmd: list[str], stage: str, desc: str, cwd: Path | None = None, timeout: int | None = None) -> None:
        cwd = cwd or self.ctx.root
        t = timeout or self.timeout
        
        status_msg = f"[bold]{desc}[/]"
        header = f"\n{'='*80}\nSTAGE: {stage}\nTIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\nCMD: {' '.join(cmd)}\n{'='*80}\n"
        start = time.time()
        
        try:
            if self.verbose:
                console.print(f" [dim]DEBUG: {' '.join(cmd)}[/]")
                p = subprocess.run(
                    cmd, cwd=str(cwd), env=self.ctx.env, text=True,
                    encoding="utf-8", errors="replace", timeout=t
                )
                output, rc = "(streamed)", p.returncode
            else:
                with console.status(f" {status_msg} ...") as status:
                    p = subprocess.run(
                        cmd, cwd=str(cwd), env=self.ctx.env, capture_output=True,
                        text=True, encoding="utf-8", errors="replace", timeout=t
                    )
                    output, rc = p.stdout + p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            out = e.stdout or ""
            err = e.stderr or ""
            if isinstance(out, bytes):
                out = out.decode("utf-8", errors="replace")
            if isinstance(err, bytes):
                err = err.decode("utf-8", errors="replace")
            output, rc = out + err + f"\nTIMEOUT after {t}s", -1

        log_entry = f"{header}STATUS: {'ok' if rc==0 else 'failed'}\nELAPSED: {time.time()-start:.2f}s\nRC: {rc}\n\n{output}\n"
        with open(self.ctx.log_file, "a", encoding="utf-8") as f: f.write(log_entry)
        if rc != 0: raise RuntimeError(f"Stage '{stage}' failed. See {self.ctx.log_file}")

    def post_build(self, portable: bool = False):
        """Platform-specific post-build tasks."""
        pass

    def verify_dependencies(self, platform_key: str | None = None):
        """Report the intended hardware acceleration from metadata.json."""
        lib_root = self.ctx.root / "lib"
        metadata_file = lib_root / "metadata.json"
        
        if not metadata_file.exists():
            console.print(Panel(
                "[bold yellow]Warning: No metadata.json found.[/]\n"
                "[dim]Building without specific backend intent. Engine will default to CPU.[/]",
                title="[bold yellow]Dependency Status[/]", border_style="yellow"
            ))
            return True

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            rules_path = Path(__file__).parent / "consistency_rules.json"
            rules = {}
            if rules_path.exists():
                with open(rules_path, "r", encoding="utf-8") as f: rules = json.load(f)
        except Exception as e:
            console.print(f"[bold red]Error loading configuration: {e}[/]")
            return False

        sys_name = py_platform.system().lower()
        
        def check_field(engine_type: str, val: str, valid_list: list[str], lib_subdir: Path, supported_list: list[str] = None):
            status = "[bold green]Ready[/]"
            hint = ""
            
            # 1. Check validity
            if val not in valid_list:
                status = "[bold red]Unrecognized[/]"
                matches = difflib.get_close_matches(val, valid_list, n=1, cutoff=0.6)
                if matches:
                    hint = f"\n [yellow]└─ Did you mean: {matches[0]}?[/]"
                else:
                    hint = f"\n [yellow]└─ Not in whitelist.[/]"
                return status, hint

            # 2. Check existence of key files (Reality Check)
            engine_rules = rules.get(engine_type, {}).get(val, {})
            req_files = engine_rules.get(sys_name, [])
            
            missing_files = []
            search_dirs = [lib_subdir, lib_subdir / "lib", lib_subdir / "bin"]
            
            for rf in req_files:
                found = False
                for sd in search_dirs:
                    if (sd / rf).exists():
                        found = True; break
                if not found: missing_files.append(rf)
            
            if missing_files:
                status = "[bold red]Missing Binaries[/]"
                hint = f"\n [red]└─ Missing: {', '.join(missing_files)}[/]"
            elif supported_list and val not in supported_list:
                status = "[bold yellow]Experimental[/]"
                hint = f"\n [dim]└─ Limited wrapper support for this provider.[/]"
            
            return status, hint

        onnx_prov = meta.get("onnx", {}).get("provider", "CPUExecutionProvider")
        llama_back = meta.get("llama", {}).get("backend", "cpu")

        onnx_status, onnx_hint = check_field("onnx", onnx_prov, VALID_PROVIDERS, self.ctx.root / "lib" / "onnx", SUPPORTED_PROVIDERS)
        llama_status, llama_hint = check_field("llama", llama_back, VALID_BACKENDS, self.ctx.root / "lib" / "llama")

        self.target_details["onnx"] = f"[bold cyan]{onnx_prov}[/] ({onnx_status})"
        self.target_details["llama"] = f"[bold cyan]{llama_back}[/] ({llama_status})"

        onnx_panel = f" [bold]Provider[/] : [cyan]{onnx_prov}[/]\n [bold]Status[/]   : {onnx_status}{onnx_hint}"
        llama_panel = f" [bold]Backend[/]  : [cyan]{llama_back}[/]\n [bold]Status[/]   : {llama_status}{llama_hint}"

        console.print("\n")
        console.print(Panel(onnx_panel, title="[bold]Engine: ONNX Audio Runtime (Decoder)[/]", border_style="blue", width=70))
        console.print(Panel(llama_panel, title="[bold]Engine: GGML / Llama.cpp (LLM)[/]", border_style="magenta", width=70))
        console.print("\n")
        
        return True

    def build(self, resolver: ToolchainResolver, clean: bool = False, parallel: int = 4, verify: bool = False, portable: bool = False, platform_key: str | None = None):
        # 1. Validation
        self.verify_dependencies(platform_key)

        # 2. Toolchain Resolution
        name, args = resolver.resolve()
        self.ctx.toolchain_name, self.ctx.toolchain_args = name, args
        console.print(f" ✨ [bold]Toolchain[/]: {name}")

        # 3. Preparation
        if clean and self.ctx.build_dir.exists(): shutil.rmtree(self.ctx.build_dir)
        self.ctx.build_dir.mkdir(exist_ok=True, parents=True)

        # 4. Configure
        cmake_args = [
            "cmake", "-S", ".", "-B", str(self.ctx.build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DQWEN3_TTS_PREBUILT_LIB_DIR={self.ctx.lib_dir.as_posix()}",
            f"-DQWEN3_TTS_ORT_ROOT={self.ctx.ort_root.as_posix()}",
        ] + args
        self._run_step(cmake_args, "cmake_configure", "Configuring Project")

        # 5. Build
        self._run_step(["cmake", "--build", str(self.ctx.build_dir), "-j", str(parallel), "--config", "Release"], 
                       "cmake_build", "Compiling Source")
        
        # 6. Post-build
        self.post_build(portable=portable)
        console.print(" ✅ [bold]Post-build Bundling Completed[/]")
        
        # 7. Verify
        if verify:
            exe_ext = ".exe" if py_platform.system() == "Windows" else ""
            exe = self.ctx.build_dir / f"qwen3-tts-cli{exe_ext}"
            if not exe.exists():
                console.print(f" ⚠️ [yellow]Verification skipped (output not found)[/]")
            else:
                self._run_step([str(exe), "--help"], "verify_help", "Verifying Binary", timeout=60)
                console.print(" 🎉 [bold green]Verification Passed[/]")

        # 8. Summary
        console.print("\n" + "─"*70)
        console.print(" [bold green]✔ Build Successfully Completed[/]")
        console.print(f" 📂 Artifacts: [underline dim]{self.ctx.build_dir}[/]")
        
        onnx_info = self.target_details.get("onnx", "[yellow]Generic (CPU)[/]")
        llama_info = self.target_details.get("llama", "[yellow]cpu[/]")
        
        console.print(f" 🎯 Optimization: {onnx_info} / {llama_info}")
        if "Missing" in onnx_info or "Missing" in llama_info:
            console.print(" ⚠️ [dim]Note: Incomplete binaries detected. Inference may fallback to CPU.[/]")
        console.print("─"*70 + "\n")

        return True
