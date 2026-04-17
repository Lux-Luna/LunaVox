import os
import shutil
import subprocess
import time
import json
import difflib
import platform as py_platform
from pathlib import Path
from rich.panel import Panel

from lunavox.core.ui import console
from .context import BuildContext

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
    def __init__(self, context: BuildContext, timeout_sec: int = 175):
        self.ctx = context
        self.timeout = max(10, timeout_sec)
        self.verbose = self.ctx.env.get("LUNAVOX_BUILD_VERBOSE", "").strip() == "1"
        self.target_details = {"onnx": "", "llama": ""}

    def _run_step(
        self,
        cmd: list[str],
        stage: str,
        desc: str,
        cwd: Path | None = None,
        timeout: int | None = None,
    ) -> None:
        cwd = cwd or self.ctx.root
        t = timeout or self.timeout

        status_msg = f"[bold]{desc}[/]"
        header = (
            f"\n{'=' * 80}\n"
            f"STAGE: {stage}\n"
            f"TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"{'=' * 80}\n"
        )
        start = time.time()

        try:
            if self.verbose:
                console.print(f" [dim]DEBUG: {' '.join(cmd)}[/]")
                p = subprocess.run(
                    cmd,
                    cwd=str(cwd),
                    env=self.ctx.env,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=t,
                )
                output, rc = "(streamed)", p.returncode
            else:
                with console.status(f" {status_msg} ..."):
                    p = subprocess.run(
                        cmd,
                        cwd=str(cwd),
                        env=self.ctx.env,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=t,
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

        log_entry = (
            f"{header}"
            f"STATUS: {'ok' if rc == 0 else 'failed'}\n"
            f"ELAPSED: {time.time() - start:.2f}s\n"
            f"RC: {rc}\n\n"
            f"{output}\n"
        )
        from lunavox.core import logging as lvlog
        lvlog.append(log_entry)
        if rc != 0:
            raise RuntimeError(f"Stage '{stage}' failed. See {self.ctx.log_file}")

    def post_build(self):
        """Platform-agnostic post-build tasks. Subclasses that need extra
        work (DLL bundling, rpath fixup) should call ``super().post_build()``
        first so the shared metadata.json copy still happens."""
        lib_root = self.ctx.root / "lib"
        metadata_file = lib_root / "metadata.json"
        if metadata_file.exists():
            shutil.copy2(metadata_file, self.ctx.build_dir / "metadata.json")
            console.print(f"[dim][build] Bundled metadata.json to {self.ctx.build_dir}[/dim]")

    def verify_dependencies(self, platform_key: str | None = None):
        """Report the intended hardware acceleration from metadata.json."""
        lib_root = self.ctx.root / "lib"
        metadata_file = lib_root / "metadata.json"

        if not metadata_file.exists():
            console.print(
                Panel(
                    "[bold yellow]Warning: No metadata.json found.[/]\n"
                    "[dim]Building without specific backend intent. Engine will default to CPU.[/]",
                    title="[bold yellow]Target Intent[/]",
                    border_style="yellow",
                )
            )
            return True

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            console.print(f"[bold red]Error loading configuration: {e}[/]")
            return False

        onnx_meta = meta.get("onnx", {})
        # build_capabilities is the ordered list of EPs linked into this
        # build. The first entry is the preferred EP (what we show as the
        # "Target Provider"); later entries exist as runtime fallbacks.
        caps = onnx_meta.get("build_capabilities")
        if isinstance(caps, list) and caps:
            onnx_prov = str(caps[0])
            fallback_caps = [str(c) for c in caps[1:]]
        else:
            onnx_prov = str(onnx_meta.get("provider", "CPUExecutionProvider"))
            fallback_caps = []
        llama_back = meta.get("llama", {}).get("backend", "cpu")

        # 1. Store details for summary
        self.target_details["onnx"] = f"[bold cyan]{onnx_prov}[/]"
        self.target_details["llama"] = f"[bold cyan]{llama_back}[/]"

        # 2. Display Intent Panels
        fallback_line = ""
        if fallback_caps:
            fallback_line = (
                f"\n [bold]Runtime Fallbacks[/] : [cyan]{', '.join(fallback_caps)}[/]"
            )
        onnx_panel = (
            f" [bold]Target Provider[/] : [cyan]{onnx_prov}[/]"
            f"{fallback_line}\n"
            " [dim]- Intent from metadata.json[/]"
        )
        llama_panel = (
            f" [bold]Target Backend[/]  : [cyan]{llama_back}[/]\n"
            " [dim]- Intent from metadata.json[/]"
        )

        console.print("\n")
        console.print(
            Panel(
                onnx_panel,
                title="[bold blue]Hardware Intent: Audio Decoder[/]",
                border_style="blue",
                width=70,
            )
        )
        console.print(
            Panel(
                llama_panel,
                title="[bold magenta]Hardware Intent: LLM Engine[/]",
                border_style="magenta",
                width=70,
            )
        )
        console.print(
            "[dim] Note: Build manager will link against these providers. "
            "Use --stats-json at runtime to verify actual acceleration.[/]\n"
        )

        return True

    def build(
        self,
        resolver: ToolchainResolver,
        clean: bool = False,
        parallel: int = 4,
        platform_key: str | None = None,
    ):
        # 1. Validation
        self.verify_dependencies(platform_key)

        # 2. Toolchain Resolution
        name, args = resolver.resolve()
        self.ctx.toolchain_name, self.ctx.toolchain_args = name, args
        console.print(f" [bold]Toolchain[/]: {name}")

        # 3. Preparation
        if clean and self.ctx.build_dir.exists():
            shutil.rmtree(self.ctx.build_dir)
        self.ctx.build_dir.mkdir(exist_ok=True, parents=True)

        # 4. Configure
        cmake_args = [
            "cmake",
            "-S",
            ".",
            "-B",
            str(self.ctx.build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY",
            f"-DLUNAVOX_PREBUILT_LIB_DIR={self.ctx.lib_dir.as_posix()}",
            f"-DLUNAVOX_ORT_ROOT={self.ctx.ort_root.as_posix()}",
        ] + args
        if py_platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_C_COMPILER_WORKS=TRUE",
                "-DCMAKE_CXX_COMPILER_WORKS=TRUE",
            ]
        self._run_step(cmake_args, "cmake_configure", "Configuring Project")

        # 5. Build
        self._run_step(
            ["cmake", "--build", str(self.ctx.build_dir), "-j", str(parallel), "--config", "Release"],
            "cmake_build",
            "Compiling Source",
        )

        # 6. Post-build
        self.post_build()
        console.print(" [bold]Post-build Bundling Completed[/]")

        # 7. Summary
        console.print("\n" + "-" * 70)
        console.print(" [bold green]Build Successfully Completed[/]")
        console.print(f" Artifacts: [underline dim]{self.ctx.build_dir}[/]")

        onnx_info = self.target_details.get("onnx", "[yellow]Generic (CPU)[/]")
        llama_info = self.target_details.get("llama", "[yellow]cpu[/]")

        console.print(f" Optimization: {onnx_info} / {llama_info}")
        console.print(
            " [dim]Note: Hardware acceleration depends on your runtime environment "
            "and installed drivers.[/]"
        )
        console.print(
            " [dim]Check the 'Backend Configuration' table during inference to verify "
            "actual GPU usage.[/]"
        )
        console.print("-" * 70 + "\n")

        return True
