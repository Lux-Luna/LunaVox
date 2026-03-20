#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LUNAVOX_CLI_NAME = "qwen3-tts-cli.exe" if sys.platform.startswith("win") else "qwen3-tts-cli"
QWEN_CORE_FILES = (
    "qwen3_tts_talker.q5_k.gguf",
    "qwen3_tts_predictor.q8_0.gguf",
    "qwen3_tts_decoder.fp16.onnx",
    "tokenizer.json",
)


def run_cmd(cmd: list[str], cwd: Path, timeout_sec: int, log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1, int(timeout_sec)),
            check=False,
        )
        output = proc.stdout or ""
        elapsed = time.time() - start
        log_path.write_text(
            f"[rc={proc.returncode}] {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"cwd: {cwd}\n"
            f"timeout_sec: {timeout_sec}\n"
            f"elapsed_sec: {elapsed:.3f}\n\n"
            f"{output}",
            encoding="utf-8",
        )
        return int(proc.returncode), output
    except subprocess.TimeoutExpired as err:
        output = ""
        if err.stdout:
            output += err.stdout if isinstance(err.stdout, str) else err.stdout.decode("utf-8", errors="ignore")
        if err.stderr:
            output += err.stderr if isinstance(err.stderr, str) else err.stderr.decode("utf-8", errors="ignore")
        elapsed = time.time() - start
        log_path.write_text(
            f"[timeout] {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"cwd: {cwd}\n"
            f"timeout_sec: {timeout_sec}\n"
            f"elapsed_sec: {elapsed:.3f}\n\n"
            f"{output}",
            encoding="utf-8",
        )
        return 124, output


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def missing_qwen_core_files(model_dir: Path) -> list[str]:
    missing: list[str] = []
    for name in QWEN_CORE_FILES:
        if not (model_dir / name).exists():
            missing.append(name)
    return missing


def run_lunavox_case(
    mode: str,
    *,
    model_dir: Path,
    build_dir: Path,
    text: str,
    reference_audio: str,
    max_steps: int,
    temperature: float,
    top_k: int,
    top_p: float,
    predictor_temperature: float,
    predictor_top_k: int,
    predictor_top_p: float,
    seed: int,
    predictor_seed: int,
    timeout_sec: int,
    out_dir: Path,
) -> dict[str, Any]:
    cli = build_dir / LUNAVOX_CLI_NAME
    if not cli.exists():
        return {"ok": False, "error": f"lunavox cli not found: {cli}"}
    stats_json = out_dir / f"lunavox_{mode}.json"
    wav_out = out_dir / f"lunavox_{mode}.wav"
    cmd = [
        str(cli),
        "-m",
        str(model_dir),
        "-t",
        text,
        "-o",
        str(wav_out),
        "--max-tokens",
        str(max_steps),
        "--temperature",
        str(temperature),
        "--top-k",
        str(top_k),
        "--top-p",
        str(top_p),
        "--predictor-temperature",
        str(predictor_temperature),
        "--predictor-top-k",
        str(predictor_top_k),
        "--predictor-top-p",
        str(predictor_top_p),
        "--seed",
        str(seed),
        "--predictor-seed",
        str(predictor_seed),
        "--stats-json",
        str(stats_json),
    ]
    if predictor_temperature <= 0.0 and predictor_top_k <= 0:
        cmd.append("--predictor-greedy")
    if mode == "clone":
        cmd += ["-r", reference_audio]

    rc, _ = run_cmd(cmd, cwd=ROOT, timeout_sec=timeout_sec, log_path=out_dir / f"lunavox_{mode}.log")
    if rc != 0:
        return {"ok": False, "returncode": rc, "error": "lunavox command failed"}
    if not stats_json.exists():
        return {"ok": False, "error": f"stats json missing: {stats_json}"}
    data = load_json(stats_json)
    data["framework"] = "lunavox"
    data["mode"] = mode
    data["ok"] = bool(data.get("success", False))
    return data


def run_qwen_case(
    mode: str,
    *,
    qwen_repo: Path,
    qwen_model_dir: str,
    qwen_python: str,
    text: str,
    reference_audio: str,
    reference_text: str,
    max_steps: int,
    temperature: float,
    top_k: int,
    top_p: float,
    predictor_temperature: float,
    predictor_top_k: int,
    predictor_top_p: float,
    seed: int,
    predictor_seed: int,
    timeout_sec: int,
    out_dir: Path,
) -> dict[str, Any]:
    stats_json = out_dir / f"qwen_{mode}.json"
    script = ROOT / "tools" / "qwen_local_benchmark.py"
    python_exec = qwen_python.strip() or sys.executable
    cmd = [
        python_exec,
        str(script),
        "--qwen-repo",
        str(qwen_repo),
        "--model-dir",
        qwen_model_dir,
        "--mode",
        mode,
        "--text",
        text,
        "--language",
        "chinese",
        "--max-steps",
        str(max_steps),
        "--temperature",
        str(temperature),
        "--top-k",
        str(top_k),
        "--top-p",
        str(top_p),
        "--sub-temperature",
        str(predictor_temperature),
        "--sub-top-k",
        str(predictor_top_k),
        "--sub-top-p",
        str(predictor_top_p),
        "--seed",
        str(seed),
        "--sub-seed",
        str(predictor_seed),
        "--out-json",
        str(stats_json),
    ]
    if mode == "clone":
        cmd += ["--reference-audio", reference_audio]
        if reference_text:
            cmd += ["--reference-text", reference_text]
    rc, _ = run_cmd(cmd, cwd=ROOT, timeout_sec=timeout_sec, log_path=out_dir / f"qwen_{mode}.log")
    if not stats_json.exists():
        return {"ok": False, "returncode": rc, "error": "qwen benchmark did not write json"}
    data = load_json(stats_json)
    data["returncode"] = rc
    data.setdefault("framework", "qwen3-tts-gguf")
    data.setdefault("mode", mode)
    return data


def diff_ratio(lhs: float, rhs: float) -> float | None:
    if rhs == 0:
        return None
    return (lhs - rhs) / rhs


def summarize_pair(lv: dict[str, Any], qw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"lunavox_ok": bool(lv.get("ok")), "qwen_ok": bool(qw.get("ok"))}
    if not (out["lunavox_ok"] and out["qwen_ok"]):
        return out

    lv_total_ms = float(lv.get("timing_ms", {}).get("total", 0.0))
    qw_total_sec = float(qw.get("timing_sec", {}).get("total_inference", 0.0))
    lv_rtf = float(lv.get("rtf", 0.0))
    qw_rtf = float(qw.get("rtf", 0.0))
    out["latency_ratio_vs_qwen"] = diff_ratio(lv_total_ms / 1000.0, qw_total_sec)
    out["rtf_ratio_vs_qwen"] = diff_ratio(lv_rtf, qw_rtf)

    out["breakdown"] = {
        "lunavox_sec": {
            "tokenize": float(lv.get("timing_ms", {}).get("tokenize", 0.0)) / 1000.0,
            "encode": float(lv.get("timing_ms", {}).get("encode", 0.0)) / 1000.0,
            "generate": float(lv.get("timing_ms", {}).get("generate", 0.0)) / 1000.0,
            "decode": float(lv.get("timing_ms", {}).get("decode", 0.0)) / 1000.0,
            "total": lv_total_ms / 1000.0,
        },
        "qwen_sec": {
            "prefill": float(qw.get("timing_sec", {}).get("prefill", 0.0)),
            "talker": float(qw.get("timing_sec", {}).get("talker", 0.0)),
            "predictor": float(qw.get("timing_sec", {}).get("predictor", 0.0)),
            "decoder": float(qw.get("timing_sec", {}).get("decoder", 0.0)),
            "total": qw_total_sec,
        },
    }
    out["delta_sec"] = {
        "total_lunavox_minus_qwen": (lv_total_ms / 1000.0) - qw_total_sec,
        "decode_lunavox_minus_qwen": float(lv.get("timing_ms", {}).get("decode", 0.0)) / 1000.0
        - float(qw.get("timing_sec", {}).get("decoder", 0.0)),
        "gen_lunavox_minus_qwen_talker_predictor": float(lv.get("timing_ms", {}).get("generate", 0.0)) / 1000.0
        - (
            float(qw.get("timing_sec", {}).get("talker", 0.0))
            + float(qw.get("timing_sec", {}).get("predictor", 0.0))
        ),
    }
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare lunavox and Qwen3-TTS-GGUF inference chains")
    ap.add_argument("--mode", choices=["base", "clone", "both"], default="both")
    ap.add_argument("--timeout-sec", type=int, default=170)
    ap.add_argument("--report-out", required=True)
    ap.add_argument("--lunavox-model-dir", required=True)
    ap.add_argument("--lunavox-build-dir", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--lunavox-reference-audio", default=str(ROOT / "ref" / "ref.wav"))
    ap.add_argument("--reference-audio", dest="lunavox_reference_audio", help="Alias of --lunavox-reference-audio")
    ap.add_argument("--qwen-reference-audio", default="")
    ap.add_argument("--qwen-reference-text", default="")
    ap.add_argument("--qwen-repo", default=str(ROOT.parent / "Qwen3-TTS-GGUF"))
    ap.add_argument("--qwen-model-dir", default=str(ROOT / "models" / "base_small"))
    ap.add_argument("--qwen-python", default="")
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--predictor-temperature", type=float, default=0.6)
    ap.add_argument("--predictor-top-k", type=int, default=50)
    ap.add_argument("--predictor-top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--predictor-seed", type=int, default=12345)
    ap.add_argument("--keep-artifacts", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    timeout_sec = max(1, int(args.timeout_sec))
    report_out = Path(args.report_out).resolve()
    artifact_dir = report_out.parent / "_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    modes = ["base", "clone"] if args.mode == "both" else [args.mode]
    report: dict[str, Any] = {
        "ok": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timeout_sec": timeout_sec,
        "sampling": {
            "max_steps": int(args.max_steps),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
            "predictor_temperature": float(args.predictor_temperature),
            "predictor_top_k": int(args.predictor_top_k),
            "predictor_top_p": float(args.predictor_top_p),
            "seed": int(args.seed),
            "predictor_seed": int(args.predictor_seed),
        },
        "notes": [],
        "modes": {},
    }

    model_dir = Path(args.lunavox_model_dir).resolve()
    build_dir = Path(args.lunavox_build_dir).resolve()
    qwen_repo = Path(args.qwen_repo).resolve()
    qwen_model_dir_resolved = Path(args.qwen_model_dir).resolve()
    qwen_model_missing = missing_qwen_core_files(qwen_model_dir_resolved)
    if qwen_model_missing:
        fallback_missing = missing_qwen_core_files(model_dir)
        if not fallback_missing:
            report["notes"].append(
                {
                    "type": "qwen_model_fallback",
                    "message": "qwen-model-dir missing required files, fallback to lunavox-model-dir",
                    "requested": str(qwen_model_dir_resolved),
                    "fallback": str(model_dir),
                    "missing": qwen_model_missing,
                }
            )
            qwen_model_dir_resolved = model_dir
        else:
            report["notes"].append(
                {
                    "type": "qwen_model_missing",
                    "message": "qwen-model-dir missing required files and fallback is also incomplete",
                    "requested": str(qwen_model_dir_resolved),
                    "missing": qwen_model_missing,
                    "fallback": str(model_dir),
                    "fallback_missing": fallback_missing,
                }
            )
    report["resolved_dirs"] = {
        "lunavox_model_dir": str(model_dir),
        "lunavox_build_dir": str(build_dir),
        "qwen_repo": str(qwen_repo),
        "qwen_model_dir": str(qwen_model_dir_resolved),
    }

    for mode in modes:
        mode_dir = artifact_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        qwen_reference_audio = args.qwen_reference_audio or args.lunavox_reference_audio
        lv = run_lunavox_case(
            mode,
            model_dir=model_dir,
            build_dir=build_dir,
            text=args.text,
            reference_audio=args.lunavox_reference_audio,
            max_steps=int(args.max_steps),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            predictor_temperature=float(args.predictor_temperature),
            predictor_top_k=int(args.predictor_top_k),
            predictor_top_p=float(args.predictor_top_p),
            seed=int(args.seed),
            predictor_seed=int(args.predictor_seed),
            timeout_sec=timeout_sec,
            out_dir=mode_dir,
        )
        qw = run_qwen_case(
            mode,
            qwen_repo=qwen_repo,
            qwen_model_dir=str(qwen_model_dir_resolved),
            qwen_python=args.qwen_python,
            text=args.text,
            reference_audio=qwen_reference_audio,
            reference_text=args.qwen_reference_text,
            max_steps=int(args.max_steps),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            predictor_temperature=float(args.predictor_temperature),
            predictor_top_k=int(args.predictor_top_k),
            predictor_top_p=float(args.predictor_top_p),
            seed=int(args.seed),
            predictor_seed=int(args.predictor_seed),
            timeout_sec=timeout_sec,
            out_dir=mode_dir,
        )
        summary = summarize_pair(lv, qw)
        report["modes"][mode] = {"lunavox": lv, "qwen": qw, "summary": summary}
        if not (lv.get("ok") and qw.get("ok")):
            report["ok"] = False

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[compare] report saved: {report_out}")

    if not args.keep_artifacts:
        for mode in modes:
            wav = artifact_dir / mode / f"lunavox_{mode}.wav"
            if wav.exists():
                try:
                    wav.unlink()
                except OSError:
                    pass
    if not args.keep_artifacts and artifact_dir.exists():
        for mode in modes:
            mode_dir = artifact_dir / mode
            if mode_dir.exists() and not any(mode_dir.iterdir()):
                shutil.rmtree(mode_dir, ignore_errors=True)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
