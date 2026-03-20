#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import wave
from pathlib import Path
from typing import Any

QWEN_CORE_FILES = (
    "qwen3_tts_talker.q5_k.gguf",
    "qwen3_tts_predictor.q8_0.gguf",
    "qwen3_tts_decoder.fp16.onnx",
    "tokenizer.json",
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_stat(stats, key: str, default: float = 0.0) -> float:
    return float(getattr(stats, key, default)) if stats is not None else default


def force_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def missing_core_files(model_dir: Path) -> list[str]:
    missing: list[str] = []
    for name in QWEN_CORE_FILES:
        if not (model_dir / name).exists():
            missing.append(name)
    return missing


def maybe_clip_wav_anchor(reference_audio: str, anchor_seconds: float, out_json: Path) -> tuple[str, str]:
    if anchor_seconds <= 0:
        return reference_audio, "disabled"
    ref_path = Path(reference_audio)
    if ref_path.suffix.lower() != ".wav":
        return reference_audio, "not_wav"
    if not ref_path.exists():
        return reference_audio, "missing"

    try:
        with wave.open(str(ref_path), "rb") as reader:
            sr = int(reader.getframerate())
            total_frames = int(reader.getnframes())
            keep_frames = min(total_frames, max(1, int(sr * anchor_seconds)))
            if keep_frames >= total_frames:
                return reference_audio, "unchanged"
            payload = reader.readframes(keep_frames)
            nch = reader.getnchannels()
            sw = reader.getsampwidth()

        tmp_path = out_json.parent / f"{ref_path.stem}.anchor_{keep_frames}f.wav"
        with wave.open(str(tmp_path), "wb") as writer:
            writer.setnchannels(nch)
            writer.setsampwidth(sw)
            writer.setframerate(sr)
            writer.writeframes(payload)
        return str(tmp_path), "clipped"
    except Exception:
        return reference_audio, "clip_failed"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run local Qwen3-TTS-GGUF benchmark and dump JSON stats")
    ap.add_argument("--qwen-repo", required=True, help="Path to Qwen3-TTS-GGUF repository")
    ap.add_argument("--model-dir", required=True, help="Model directory relative to qwen repo or absolute")
    ap.add_argument("--mode", choices=["base", "clone"], required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--language", default="chinese")
    ap.add_argument("--speaker", default="Vivian")
    ap.add_argument("--reference-audio", default="")
    ap.add_argument("--reference-text", default="")
    ap.add_argument("--clone-anchor-seconds", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--sub-temperature", type=float, default=0.6)
    ap.add_argument("--sub-top-p", type=float, default=1.0)
    ap.add_argument("--sub-top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--sub-seed", type=int, default=12345)
    ap.add_argument("--single-process", action="store_true", default=True,
                    help="Use local in-process decoder proxy to avoid multiprocessing restrictions.")
    ap.add_argument("--no-single-process", action="store_true",
                    help="Disable local in-process decoder proxy patch.")
    ap.add_argument("--out-json", required=True)
    return ap.parse_args()


def install_local_decoder_proxy_patch() -> None:
    import numpy as np
    from qwen3_tts_gguf.inference import engine as engine_mod  # type: ignore
    from qwen3_tts_gguf.inference.decoder import StatefulDecoder  # type: ignore
    from qwen3_tts_gguf.inference.schema.protocol import DecoderResponse  # type: ignore
    from qwen3_tts_gguf.inference.schema.result import DecodeResult, TTSResult  # type: ignore

    if getattr(engine_mod, "_lunavox_local_decoder_proxy", False):
        return

    class LocalDecoderProxy:
        def __init__(self, onnx_path: str, onnx_provider: str = "CPU", chunk_size: int = 12):
            self.decoder = StatefulDecoder(onnx_path, onnx_provider=onnx_provider, chunk_size=chunk_size)
            self.ready_states = {"decoder": True, "speaker": True}
            self._states: dict[Any, Any] = {}
            self._responses: dict[Any, list[Any]] = {}

        def wait_until_ready(self, timeout=10):
            return True

        def join_speaker(self, timeout=None):
            return None

        def join_decoder(self, timeout=None):
            return None

        def pause(self):
            return None

        def resume(self):
            return None

        def raw_play(self, pcm):
            return None

        def stop(self, task_id="default"):
            self._states.pop(task_id, None)
            self._responses.pop(task_id, None)
            return np.array([], dtype=np.float32)

        def get_decode_result(self, task_id):
            return DecodeResult(responses=self._responses.get(task_id, []))

        def shutdown(self):
            self._states.clear()
            self._responses.clear()

        def decode(self, input, task_id="default", is_final: bool = False, stream: bool = False, state=None):
            if isinstance(input, TTSResult):
                if input.ref_codes is not None and len(input.ref_codes) > 0 and input.final_state is None:
                    ref_init = self.decode(input.ref_codes, task_id=f"{task_id}_ref_init", is_final=True)
                    input.final_state = ref_init.final_state
                state = state or input.final_state
                codes = input.codes
                is_final = True
            else:
                codes = input

            codes_arr = np.array(codes, dtype=np.int64)
            current_state = state if state is not None else self._states.get(task_id)
            idx = len(self._responses.get(task_id, []))

            t0 = time.time()
            audio, next_state = self.decoder.decode(codes_arr, state=current_state, is_final=(is_final or not stream))
            dt = time.time() - t0
            self._states[task_id] = next_state

            responses = self._responses.setdefault(task_id, [])
            responses.append(
                DecoderResponse(
                    task_id=task_id,
                    msg_type="AUDIO",
                    index=idx,
                    audio=np.asarray(audio, dtype=np.float32),
                    compute_time=dt,
                    recv_time=time.time(),
                )
            )

            if stream and not is_final:
                return np.array([], dtype=np.float32)

            responses.append(
                DecoderResponse(
                    task_id=task_id,
                    msg_type="FINISH",
                    index=idx + 1,
                    state=next_state,
                    recv_time=time.time(),
                )
            )
            result = DecodeResult(responses=responses.copy())
            self._responses.pop(task_id, None)

            if isinstance(input, TTSResult):
                input.audio = result.audio
                input.final_state = result.final_state
                if input.stats:
                    input.stats.decoder_compute_times = result.chunk_compute_times
            return result

    engine_mod.DecoderProxy = LocalDecoderProxy
    engine_mod._lunavox_local_decoder_proxy = True


def main() -> int:
    force_utf8_stdio()
    args = parse_args()
    out_json = Path(args.out_json).resolve()
    qwen_repo = Path(args.qwen_repo).resolve()

    payload: dict = {
        "ok": False,
        "mode": args.mode,
        "framework": "qwen3-tts-gguf",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        if not qwen_repo.exists():
            raise RuntimeError(f"Qwen repo not found: {qwen_repo}")
        os.chdir(str(qwen_repo))
        if str(qwen_repo) not in sys.path:
            sys.path.insert(0, str(qwen_repo))

        from qwen3_tts_gguf.inference import TTSConfig  # type: ignore
        if args.single_process and not args.no_single_process:
            install_local_decoder_proxy_patch()
        from qwen3_tts_gguf.inference import TTSEngine  # type: ignore

        cfg = TTSConfig(
            max_steps=int(args.max_steps),
            streaming=False,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            sub_temperature=float(args.sub_temperature),
            sub_top_p=float(args.sub_top_p),
            sub_top_k=int(args.sub_top_k),
            seed=int(args.seed),
            sub_seed=int(args.sub_seed),
        )
        model_dir_arg = args.model_dir
        resolved_model_dir: Path
        if Path(model_dir_arg).is_absolute():
            # TTSEngine internally requires model path to be relative to qwen repo root.
            resolved_model_dir = Path(model_dir_arg).resolve()
            try:
                model_dir_arg = os.path.relpath(model_dir_arg, start=str(qwen_repo))
            except ValueError:
                # Different drive: keep original absolute path and let engine surface a clear error.
                model_dir_arg = str(Path(model_dir_arg))
        else:
            resolved_model_dir = (qwen_repo / model_dir_arg).resolve()
            model_dir_arg = str(model_dir_arg)
        payload["resolved_model_dir"] = str(resolved_model_dir)
        payload["model_dir_arg"] = model_dir_arg
        payload["missing_core_files"] = missing_core_files(resolved_model_dir)

        t0 = time.time()
        engine = TTSEngine(model_dir=model_dir_arg, onnx_provider="CPU", chunk_size=12, verbose=False)
        if not engine:
            missing = payload.get("missing_core_files", [])
            if missing:
                raise RuntimeError(f"TTSEngine init failed: missing core files in model_dir: {missing}")
            raise RuntimeError("TTSEngine init failed (engine not ready)")
        stream = engine.create_stream()
        if stream is None:
            raise RuntimeError("create_stream() returned None")
        payload["init_sec"] = time.time() - t0

        t1 = time.time()
        if args.mode == "base":
            result = stream.custom(
                text=args.text,
                speaker=args.speaker,
                language=args.language,
                config=cfg,
            )
        else:
            if not args.reference_audio:
                raise RuntimeError("clone mode requires --reference-audio")
            prepared_ref_audio, anchor_status = maybe_clip_wav_anchor(
                args.reference_audio, float(args.clone_anchor_seconds), out_json
            )
            payload["reference_audio_input"] = args.reference_audio
            payload["reference_audio_used"] = prepared_ref_audio
            payload["clone_anchor_status"] = anchor_status
            if not stream.set_voice(prepared_ref_audio, args.reference_text):
                raise RuntimeError("set_voice(reference_audio) failed")
            result = stream.clone(
                text=args.text,
                language=args.language,
                config=cfg,
            )
        wall_sec = time.time() - t1

        if result is None:
            raise RuntimeError("inference returned None")

        stats = getattr(result, "stats", None)
        audio = getattr(result, "audio", None)
        codes = getattr(result, "codes", None)
        sr = 24000
        audio_samples = int(len(audio)) if audio is not None else 0
        audio_sec = float(audio_samples) / float(sr) if audio_samples > 0 else 0.0
        rtf = wall_sec / audio_sec if audio_sec > 0 else 0.0

        payload.update(
            {
                "ok": True,
                "sample_rate": sr,
                "audio_samples": audio_samples,
                "audio_sec": audio_sec,
                "wall_sec": wall_sec,
                "rtf": rtf,
                "codes_frames": int(len(codes)) if codes is not None else 0,
                "timing_sec": {
                    "prompt": get_stat(stats, "prompt_time", 0.0),
                    "prefill": get_stat(stats, "prefill_time", 0.0),
                    "talker": get_stat(stats, "total_talker_time", 0.0),
                    "predictor": get_stat(stats, "total_predictor_time", 0.0),
                    "decoder": get_stat(stats, "total_decoder_time", 0.0),
                    "inference_only": get_stat(stats, "inference_only_time", 0.0),
                    "total_inference": get_stat(stats, "total_inference_time", 0.0),
                },
            }
        )

        try:
            stream.shutdown()
        except Exception:
            pass
        try:
            engine.shutdown()
        except Exception:
            pass
        write_json(out_json, payload)
        return 0

    except Exception as err:
        payload["error"] = str(err)
        payload["traceback"] = traceback.format_exc()
        write_json(out_json, payload)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
