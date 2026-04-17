"""``lunavox synth`` — direct in-process synthesis via the runtime API.

Phase 6 makes this a thin adapter over :mod:`lunavox.core.synth`:
voice resolution, parameter defaulting, auto-split, and WAV encoding
all live in core. This command now only does what a CLI front-end
should: parse flags, report progress, write the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, cast

import typer

from lunavox.core.synth import (
    SynthesisPipeline,
    VoiceResolutionError,
    VoiceSpec,
    f32_to_wav,
    resolve_voice,
)

VoiceMode = Literal["base", "clone", "custom", "design"]
from lunavox.core.ui import console

from ._common import RuntimeState, state


def register(parent: typer.Typer) -> None:
    """Attach ``synth`` onto ``parent``.

    We register with a function rather than a fresh sub-Typer so the
    command shows up at the top level (``lunavox synth "…"``) rather
    than nested (``lunavox synth synth "…"``).
    """

    @parent.command("synth")
    def synth_cmd(
        ctx: typer.Context,
        text: str = typer.Argument(..., help="Text to synthesize"),
        output: Path = typer.Option(
            Path("output/synth.wav"), "--output", "-o", help="Output WAV path"
        ),
        model: Optional[str] = typer.Option(
            None, "--model", help="Model directory name under models/ (override config)"
        ),
        voice: str = typer.Option(
            "base",
            "--voice",
            help="Voice mode: base | clone | custom | design",
        ),
        reference: Optional[Path] = typer.Option(
            None, "--ref", help="Reference WAV or .json (voice=clone)"
        ),
        speaker: Optional[str] = typer.Option(
            None, "--speaker", help="Catalog speaker id (voice=custom)"
        ),
        instruct: Optional[str] = typer.Option(
            None, "--instruct", help="Style instruction (voice=custom or design)"
        ),
        temperature: Optional[float] = typer.Option(None, "--temperature"),
        top_p: Optional[float] = typer.Option(None, "--top-p"),
        top_k: Optional[int] = typer.Option(None, "--top-k"),
    ) -> None:
        """Synthesize ``text`` to a WAV file via the in-process engine."""
        st = state(ctx)
        _run_synth(
            st,
            text=text,
            output=output,
            model=model,
            voice=voice,
            reference=reference,
            speaker=speaker,
            instruct=instruct,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )


def _run_synth(
    st: RuntimeState,
    *,
    text: str,
    output: Path,
    model: Optional[str],
    voice: str,
    reference: Optional[Path],
    speaker: Optional[str],
    instruct: Optional[str],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
) -> None:
    # Import here so `lunavox --help` and `lunavox doctor` don't pay
    # the ctypes + liblunavox load cost.
    from lunavox.runtime import Engine, SynthesisParams

    resolved_model = model or st.config.model
    model_dir = st.project_root / "models" / resolved_model
    if not model_dir.exists():
        raise RuntimeError(
            f"Model directory {model_dir} not found. Run `lunavox model pull --model "
            f"{resolved_model}` first."
        )

    spec = VoiceSpec(
        mode=_coerce_voice_mode(voice),
        reference=str(reference) if reference is not None else None,
        speaker=speaker,
        instruct=instruct,
    )
    try:
        voice_obj = resolve_voice(spec)
    except VoiceResolutionError as err:
        raise RuntimeError(str(err)) from err

    params = SynthesisParams.from_overrides(
        temperature=temperature if temperature is not None else st.config.temperature,
        top_p=top_p if top_p is not None else st.config.top_p,
        top_k=top_k if top_k is not None else st.config.top_k,
        n_threads=st.config.n_threads,
        repetition_penalty=st.config.repetition_penalty,
        language_id=st.config.language_id,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    console.print(
        f"[stage]Synthesizing with model=[bold]{resolved_model}[/] voice=[bold]{voice}[/][/]"
    )

    with Engine(model_dir, n_threads=st.config.n_threads) as engine:
        pipeline = SynthesisPipeline(engine)
        result = pipeline.synthesize(text, voice=voice_obj, params=params)
        output.write_bytes(f32_to_wav(result.audio, result.sample_rate))

    console.print(
        f"[success]Synthesis complete: {output} "
        f"(duration={result.stats.audio_duration_ms / 1000:.2f}s, "
        f"rtf={result.stats.rtf:.3f})[/success]"
    )


def _coerce_voice_mode(mode: str) -> VoiceMode:
    """Normalise the ``--voice`` flag into one of the four canonical modes.

    Case-insensitive; unknown values fail fast with a typer-friendly
    error so the user sees a clean usage hint.
    """
    m = mode.lower()
    if m not in {"base", "clone", "custom", "design"}:
        raise RuntimeError(
            f"Unknown voice mode: {mode}. Expected base|clone|custom|design."
        )
    return cast(VoiceMode, m)
