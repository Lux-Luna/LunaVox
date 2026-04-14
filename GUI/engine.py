"""GUI side of the LunaVox runtime.

Historically this module shelled out to ``lunavox-cli`` via subprocess and
regex-parsed the stdout to recover stats — brittle, slow, and duplicated
the work the C++ engine already reports. It now calls
:class:`lunavox.runtime.Engine` through the ctypes binding, so stats come
back as structured fields and there is no parser to keep in sync with the
CLI output format.

The engine handle is cached per model so repeated synthesize calls do not
reload the GGUF/ONNX files. Switching models closes the previous handle
before loading the next one.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from lunavox.runtime import (
    Engine,
    LunavoxSynthesisError,
    SynthesisParams,
    SynthesisResult,
)

log = logging.getLogger("lunavox.gui.engine")


def _bytes_to_human(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    val = float(n)
    idx = 0
    while val >= 1024.0 and idx < len(units) - 1:
        val /= 1024.0
        idx += 1
    return f"{val:.2f} {units[idx]}"


class LunaVoxEngine:
    """Thin GUI-side wrapper over ``lunavox.runtime.Engine``.

    Responsibilities:

    - Discover local models by scanning ``models/*/model_profile.json``.
    - Discover reference audio files under ``ref/``.
    - Lazily open a ``runtime.Engine`` per model directory and run
      synthesize on it, returning a ``SynthesisResult``.
    - Surface the same command-preview text the old subprocess
      version showed, for users copying commands out to a terminal.
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.models_dir = self.root_dir / "models"
        self.build_dir = self.root_dir / "build"
        self.ref_dir = self.root_dir / "ref"
        self.logs_dir = self.root_dir / "logs"

        self._active_model_dir: Optional[Path] = None
        self._active_engine: Optional[Engine] = None

    # --- introspection -------------------------------------------------

    def get_backend_info(self) -> Optional[dict]:
        metadata_path = self.build_dir / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            with open(metadata_path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception:
            return None

    def discover_models(self) -> list[dict]:
        models: list[dict] = []
        if not self.models_dir.exists():
            return models
        for d in self.models_dir.iterdir():
            if not d.is_dir():
                continue
            profile_path = d / "model_profile.json"
            if not profile_path.exists():
                continue
            try:
                with open(profile_path, "r", encoding="utf-8-sig") as f:
                    profile = json.load(f)
            except Exception:
                continue
            size = profile.get("model_size", "unknown").upper()
            mtype = profile.get("model_type", "unknown").capitalize()
            models.append({
                "id": d.name,
                "name": f"{d.name} (Qwen3-TTS-12Hz-{size}-{mtype})",
                "type": profile.get("model_type", "base"),
                "path": str(d.absolute()),
                "profile": profile,
                "instruct_support": profile.get("instruct_support", False),
            })
        return models

    def discover_references(self) -> list[str]:
        refs: list[str] = []
        if not self.ref_dir.exists():
            return refs
        for f in self.ref_dir.iterdir():
            if f.suffix in (".wav", ".json"):
                refs.append(str(f.relative_to(self.root_dir)))
        return refs

    # --- engine lifecycle ---------------------------------------------

    def _ensure_engine(self, model_dir: Path, n_threads: int) -> Engine:
        """Return an Engine bound to ``model_dir``, reloading if it changed.

        Keeps a single handle alive between synth runs to avoid re-paying
        load + warmup on every generate-click. The caller is responsible
        for calling ``shutdown()`` before the app exits; otherwise the C++
        engine leaks (ctypes does drop the handle via ``__del__`` but that
        runs outside our control).
        """
        if self._active_engine is not None and self._active_model_dir == model_dir:
            return self._active_engine

        self.shutdown()
        log.info("GUI loading engine for %s", model_dir)
        self._active_engine = Engine(model_dir, n_threads=n_threads)
        self._active_model_dir = model_dir
        return self._active_engine

    def shutdown(self) -> None:
        if self._active_engine is not None:
            try:
                self._active_engine.close()
            except Exception:
                pass
            self._active_engine = None
            self._active_model_dir = None

    # --- synthesis -----------------------------------------------------

    def run_synthesis(self, args: dict[str, Any]) -> SynthesisResult:
        """Synthesize audio for the chosen mode and return structured stats.

        ``args`` is the same dict the GUI has historically built from its
        form fields. Unlike the old implementation this is a blocking
        call — the caller must run it on a worker thread if it wants a
        responsive UI.
        """
        model_path = Path(args["model_path"]).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        params = SynthesisParams(
            max_audio_tokens=int(args.get("max_new_tokens", 400) or 0),
            temperature=float(args.get("temperature", 0.6)),
            top_p=float(args.get("top_p", 1.0)),
            top_k=int(args.get("top_k", 50)),
            n_threads=int(args.get("threads", 4)),
            repetition_penalty=float(args.get("repetition_penalty", 1.05)),
            language_id=-1,
            ref_text=args.get("ref_text") or None,
        )
        # Language override: the GUI speaks in names (e.g. "English") but
        # the C API wants a language_id. We keep -1 (Auto) if the user
        # picked "auto" and otherwise let the engine resolve the name
        # through the reference-text path. The legacy CLI plumbing is not
        # ported because nothing in the GUI currently sets it beyond auto.

        engine = self._ensure_engine(model_path, params.n_threads)

        mtype = args.get("model_type", "base")
        text = args["text"]

        if mtype == "base":
            reference = args.get("reference")
            if reference:
                ref_path = (self.root_dir / reference).resolve()
                result = engine.synthesize_with_voice_file(text, ref_path, params)
            else:
                result = engine.synthesize(text, params)
        elif mtype == "custom":
            speaker = args.get("speaker") or ""
            if not speaker:
                raise LunavoxSynthesisError("Custom mode requires a speaker")
            result = engine.synthesize_custom(
                text,
                speaker,
                args.get("instruct") or "",
                params,
            )
        elif mtype == "design":
            instruct = args.get("instruct") or ""
            if not instruct.strip():
                raise LunavoxSynthesisError("Design mode requires an instruct prompt")
            result = engine.synthesize_design(text, instruct, params)
        else:
            raise LunavoxSynthesisError(f"Unknown model_type: {mtype}")

        return result

    def save_audio(self, result: SynthesisResult, output_path: str) -> Path:
        """Write a synthesis result to a 16-bit PCM WAV file."""
        import wave
        import numpy as np

        out = (self.root_dir / output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        samples = np.clip(result.audio, -1.0, 1.0)
        pcm = (samples * 32767.0).astype(np.int16)
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(result.sample_rate)
            wf.writeframes(pcm.tobytes())
        return out

    # --- legacy helpers kept for GUI report view -----------------------

    def format_metrics(self, result: SynthesisResult) -> dict[str, str]:
        """Convert a SynthesisResult to the metric dict the GUI report
        panel was built around. The old implementation regex-parsed
        CLI stdout; this function just projects the already-structured
        fields so the report view needs no changes."""
        s = result.stats
        backend = self.get_backend_info() or {}
        return {
            "latency": f"{s.t_total_ms} ms",
            "ram": _bytes_to_human(s.rss_peak_bytes) if s.rss_peak_bytes else "N/A",
            "vram": "N/A",  # C API does not expose GPU VRAM yet
            "tokenization": f"{s.t_tokenize_ms} ms",
            "speaker_encoding": f"{s.t_encode_ms} ms",
            "code_generation": f"{s.t_generate_ms} ms",
            "audio_decoding": f"{s.t_decode_ms} ms",
            "rtf": f"{s.rtf:.3f}",
            "duration": f"{s.audio_duration_ms / 1000.0:.2f} s",
            "actual_backend_llama": str(backend.get("llama", {}).get("backend", "N/A")),
            "actual_backend_onnx": str(backend.get("onnx", {}).get("provider", "N/A")),
        }

    def get_command_string(self, args: dict[str, Any]) -> str:
        """Return the CLI command a user would run to reproduce this
        synthesize call. Still useful for the command-preview panel even
        though the GUI itself no longer spawns a subprocess."""
        platform = args.get("platform", "Windows")
        ext = ".exe" if platform == "Windows" else ""
        sep = "`" if platform == "Windows" else "\\"

        lines = [f".\\build\\lunavox-cli{ext} {sep}"]
        lines.append(f"  -m models/{args['model_id']} {sep}")
        lines.append(f"  -t \"{args['text']}\" {sep}")

        output_path = args.get("output", "output/out.wav")
        lines.append(f"  -o {output_path} {sep}")

        if args.get("language") and args["language"] != "auto":
            lines.append(f"  -l {args['language']} {sep}")

        mtype = args["model_type"]
        if mtype == "base":
            if args.get("reference"):
                lines.append(f"  -r \"{args['reference']}\" {sep}")
                if args.get("ref_text"):
                    lines.append(f"  --ref-text \"{args['ref_text']}\" {sep}")
        elif mtype == "custom":
            lines.append(f"  --mode custom {sep}")
            lines.append(f"  --speaker {args['speaker']} {sep}")
            if args.get("instruct") and args["instruct"].strip():
                lines.append(f"  --instruct \"{args['instruct']}\" {sep}")
        elif mtype == "design":
            lines.append(f"  --mode design {sep}")
            if args.get("instruct") and args["instruct"].strip():
                lines.append(f"  --instruct \"{args['instruct']}\" {sep}")

        if round(args.get("temperature", 0.6), 2) != 0.6:
            lines.append(f"  --temperature {args['temperature']} {sep}")
        if round(args.get("predictor_temp", 0.6), 2) != 0.6:
            lines.append(f"  --predictor-temperature {args['predictor_temp']} {sep}")
        if args.get("max_new_tokens", 400) != 400:
            lines.append(f"  --max-tokens {args['max_new_tokens']} {sep}")
        if args.get("seed", 42) != 42:
            lines.append(f"  --seed {args['seed']} {sep}")
        if args.get("top_k", 50) != 50:
            lines.append(f"  --predictor-top-k {args['top_k']} {sep}")
        if round(args.get("top_p", 1.0), 2) != 1.0:
            lines.append(f"  --predictor-top-p {args['top_p']} {sep}")
        if round(args.get("repetition_penalty", 1.05), 2) != 1.05:
            lines.append(f"  --repetition-penalty {args['repetition_penalty']} {sep}")
        if args.get("threads", 4) != 4:
            lines.append(f"  -j {args['threads']} {sep}")

        if lines[-1].endswith(f" {sep}"):
            lines[-1] = lines[-1][: -(len(sep) + 1)]
        return "\n".join(lines)
