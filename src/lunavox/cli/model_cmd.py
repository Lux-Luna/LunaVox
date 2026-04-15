"""``lunavox model {pull,convert,list}`` — the model lifecycle group."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel
from rich.table import Table

from lunavox.core.deps import DependencyPolicy, ensure_dependency_group
from lunavox.core.ui import console
from lunavox.model import (
    ModelConvertPipeline,
    ModelDownloader,
    Models,
    all_models,
    model_keys,
)

from ._common import RuntimeState, confirm_or_fail, pick_model, state

app = typer.Typer(no_args_is_help=True, help="Model catalog, pull, and convert.")


@app.command("list")
def list_cmd(ctx: typer.Context) -> None:
    """Show every model in the catalog and whether its artifacts are local."""
    st = state(ctx)
    table = Table(title="LunaVox Models", show_header=True, header_style="bold cyan")
    table.add_column("Key")
    table.add_column("Display")
    table.add_column("HF Repo")
    table.add_column("Installed")

    models_dir = st.project_root / "models"
    for spec in all_models():
        local = models_dir / spec.name
        installed = "✓" if local.exists() and any(local.iterdir()) else "—"
        table.add_row(spec.name, spec.display_name, spec.repo_id, installed)
    console.print(table)


@app.command("pull")
def pull_cmd(
    ctx: typer.Context,
    model: Optional[str] = typer.Option(None, "--model", help="Model variant"),
) -> None:
    """Download pre-converted runtime artifacts (GGUF/ONNX) from HuggingFace."""
    st = state(ctx)
    pull(st, model)


def pull(st: RuntimeState, model: Optional[str]) -> list[str]:
    """Internal entry used by both ``lunavox model pull`` and ``bootstrap``."""
    selected_keys = pick_model(
        header="Model Pull Selection",
        prompt="Select a model to pull",
        model=model,
    )
    if not selected_keys:
        return []

    for key in selected_keys:
        console.print(Panel(f"Pulling Model: [bold]{key}[/]", style="stage", expand=False))
        ModelDownloader.download_converted_model(key, st.project_root)

    console.print("\n[success]Model pull process completed successfully.[/success]")
    return selected_keys


@app.command("convert")
def convert_cmd(
    ctx: typer.Context,
    model: Optional[str] = typer.Option(None, "--model", help="Model variant (id)"),
    models_dir: str = typer.Option("", "--models-dir", help="Override output models directory"),
    force: bool = typer.Option(False, "--force", help="Force reconversion"),
    all_: bool = typer.Option(False, "--all", help="Convert every model in the catalog"),
) -> None:
    """Convert source model weights into LunaVox runtime artifacts.

    Requires the ``[convert]`` optional extra. ``lunavox doctor`` will
    tell you which modules are missing.
    """
    st = state(ctx)

    if all_:
        selected_keys = model_keys()
    else:
        selected_keys = pick_model(
            header="Model Conversion Selection",
            prompt="Select a model to convert",
            model=model,
        )
    if not selected_keys:
        return

    for key in selected_keys:
        console.print(Panel(f"Converting Model: [bold]{key}[/]", style="stage", expand=False))
        _convert_one(st, model=key, models_dir=models_dir, force=force)

    console.print("[success]Conversion process completed.[/success]")


def _convert_one(st: RuntimeState, model: str, models_dir: str, force: bool) -> None:
    ensure_dependency_group(
        "convert",
        DependencyPolicy(yes=st.config.yes, no_install=st.config.no_install),
        project_root=st.project_root,
    )
    models = Models(st.project_root)
    cfg = models.by_name(model)

    required = [cfg.source / "config.json", cfg.source / "model.safetensors"]
    missing = [p for p in required if not p.exists()]
    if missing:
        allow = confirm_or_fail(
            st,
            f"Model '{model}' source files are missing. Download from HuggingFace now?",
        )
        if not allow:
            raise RuntimeError("Model download cancelled by user.")
        cfg.source = ModelDownloader.download(model)

    target_dir = Path(models_dir).resolve() if models_dir else cfg.dest.resolve()
    pipeline = ModelConvertPipeline(st.project_root)
    pipeline.convert(cfg, target_dir, force=force)
