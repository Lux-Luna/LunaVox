"""``lunavox synth`` — direct in-process synthesis via the runtime API.

Before this command existed, the only way to synthesize from the CLI
was to run the C++ ``lunavox-cli`` exe as a subprocess. ``lunavox
synth`` is the native path: it loads ``Engine`` in-process, picks a
``Voice``, and writes a WAV. Same code path the GUI and future serve
layer will use, so it's the canonical smoke test for every release.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Any, Optional

import typer

from lunavox.core.ui import console

from ._common import RuntimeState, state

app_command = typer.Typer  # only used for type hints below; app created in main.py


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

    v = _build_voice(voice, reference, speaker, instruct)
    params = SynthesisParams(
        temperature=(temperature if temperature is not None else st.config.temperature),
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
        result = engine.synthesize(text, voice=v, params=params)
        _write_wav(output, result.audio, result.sample_rate)

    console.print(
        f"[success]Synthesis complete: {output} "
        f"(duration={result.stats.audio_duration_ms / 1000:.2f}s, "
        f"rtf={result.stats.rtf:.3f})[/success]"
    )


def _build_voice(
    voice: str,
    reference: Optional[Path],
    speaker: Optional[str],
    instruct: Optional[str],
) -> Any:
    """Translate ``--voice`` + the related flags into a :class:`Voice`."""
    from lunavox.runtime import Voice

    v = voice.lower()
    if v == "base":
        return Voice.base()
    if v == "clone":
        if reference is None:
            raise RuntimeError("--voice clone requires --ref <path>")
        return Voice.clone_file(reference)
    if v == "custom":
        if not speaker:
            raise RuntimeError("--voice custom requires --speaker <id>")
        return Voice.custom(speaker, instruct=instruct or "")
    if v == "design":
        if not instruct:
            raise RuntimeError("--voice design requires --instruct <text>")
        return Voice.design(instruct)
    raise RuntimeError(f"Unknown voice mode: {voice}. Expected base|clone|custom|design.")


def _write_wav(path: Path, audio: Any, sample_rate: int) -> None:
    """Write a mono 16-bit PCM WAV. Takes any iterable of float samples
    in [-1, 1]; GUI and CLI share the same helper."""
    pcm16 = bytearray()
    for sample in audio:
        clipped = max(-1.0, min(1.0, float(sample)))
        pcm16 += struct.pack("<h", int(clipped * 32767.0))

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(bytes(pcm16))
