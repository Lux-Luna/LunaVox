"""``lunavox doctor`` — environment check and dependency triage."""

from __future__ import annotations

import shutil
import sys

import typer
from rich.table import Table

from lunavox.core.deps import DEPENDENCY_GROUPS, missing_modules
from lunavox.core.project import DEPLOYMENT_MARKER
from lunavox.core.ui import console

from ._common import state


def register(parent: typer.Typer) -> None:
    @parent.command("doctor")
    def doctor_cmd(ctx: typer.Context) -> None:
        """Print an environment health check."""
        st = state(ctx)

        table = Table(title="LunaVox Doctor", show_header=True, header_style="bold cyan")
        table.add_column("Check")
        table.add_column("Result")
        table.add_column("Details")

        dev_layout = (st.project_root / "CMakeLists.txt").exists() and (
            st.project_root / "src"
        ).exists()
        deployment_layout = (st.project_root / DEPLOYMENT_MARKER).exists()
        root_ok = (
            (dev_layout or deployment_layout)
            and (st.project_root / "lib").exists()
            and (st.project_root / "models").exists()
        )
        table.add_row("Project Root", "OK" if root_ok else "FAIL", str(st.project_root))

        cmake = shutil.which("cmake")
        table.add_row("CMake", "OK" if cmake else "WARN", cmake or "Not found in PATH")

        llama_dir = st.project_root / "lib" / "llama"
        onnx_header = st.project_root / "lib" / "onnx" / "include" / "onnxruntime_cxx_api.h"
        table.add_row("Llama Runtime", "OK" if llama_dir.exists() else "WARN", str(llama_dir))
        table.add_row("ONNX SDK Header", "OK" if onnx_header.exists() else "WARN", str(onnx_header))

        missing_convert = missing_modules(DEPENDENCY_GROUPS["convert"])
        if missing_convert:
            details = ", ".join(f"{m[0]}({m[1]})" for m in missing_convert)
            table.add_row("Convert Dependencies", "WARN", details)
        else:
            table.add_row("Convert Dependencies", "OK", "All required modules available")

        table.add_row("Log File", "INFO", str(st.latest_log))
        if st.config.profile_name:
            table.add_row("Active Profile", "INFO", st.config.profile_name)

        console.print(table)

        if missing_convert:
            install_cmd = f'{sys.executable} -m pip install "lunavox[convert]"'
            console.print(
                f"Missing convert deps. Install with: {install_cmd}",
                style="warning",
                markup=False,
            )
