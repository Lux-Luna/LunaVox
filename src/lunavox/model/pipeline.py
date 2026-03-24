from __future__ import annotations
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

class ModelSetupPipeline:
    def __init__(self, root: Path):
        self.root = root
        self.logs_dir = root / "logs"
        self.latest_log = self.logs_dir / "latest.log"

    def eprint(self, msg: str) -> None:
        print(msg, file=sys.stderr)

    def run_cmd(
        self,
        cmd: list[str],
        cwd: Path,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        self.eprint(f"[run] {' '.join(cmd)}")
        start = time.time()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        header = (
            f"\n{'='*80}\n"
            f"TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"CWD: {cwd}\n"
            f"{'='*80}\n"
        )
        
        run_env = dict(os.environ if env is None else env)
        run_env["LUNAVOX_PROJECT_ROOT"] = str(self.root)

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                check=True,
                env=run_env,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output_text = proc.stdout or ""
            elapsed = time.time() - start
            log_entry = (
                f"{header}"
                f"STATUS: ok\n"
                f"ELAPSED: {elapsed:.3f}s\n\n"
                f"{output_text}\n"
            )
            with open(self.latest_log, "a", encoding="utf-8") as f:
                f.write(log_entry)
                    
        except subprocess.CalledProcessError as err:
            output_text = err.stdout or ""
            elapsed = time.time() - start
            log_entry = (
                f"{header}"
                f"STATUS: failed\n"
                f"RETURNCODE: {err.returncode}\n"
                f"ELAPSED: {elapsed:.3f}s\n\n"
                f"{output_text}\n"
            )
            with open(self.latest_log, "a", encoding="utf-8") as f:
                f.write(log_entry)
            raise

    def ensure_source_exists(self, cfg) -> None:
        """Verify the HF source model exists on disk."""
        required = [cfg.source / "config.json", cfg.source / "model.safetensors"]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Model source missing for '{cfg.name}' at {cfg.source}. Missing: {', '.join(missing)}"
            )
        self.eprint(f"[ok] model source found: {cfg.source}")

    def setup(self, cfg, models_dir: Path, force: bool = False):
        self.eprint(f"[pipeline] Setting up model: {cfg.name}")
        self.ensure_source_exists(cfg)
        
        base_dir = cfg.source.resolve()
        
        # Determine ONNX source
        onnx_source_dir = models_dir
        if cfg.name in ("custom", "design"):
            onnx_source_dir = cfg.dest.parent / "base"
        elif cfg.name == "custom_small":
            onnx_source_dir = cfg.dest.parent / "base_small"

        out_talker = models_dir / "qwen3_tts_talker.q5_k.gguf"
        out_predictor = models_dir / "qwen3_tts_predictor.q8_0.gguf"
        out_codec_encoder = models_dir / "qwen3_tts_codec_encoder.fp16.onnx"
        out_speaker_encoder = models_dir / "qwen3_tts_speaker_encoder.fp16.onnx"
        out_decoder = models_dir / "qwen3_tts_decoder.fp16.onnx"
        out_embeddings_dir = models_dir / "embeddings"
        out_tokenizer_json = models_dir / "tokenizer.json"

        models_dir.mkdir(parents=True, exist_ok=True)

        if force:
            for p in [out_talker, out_predictor, out_codec_encoder, out_speaker_encoder, out_decoder, out_tokenizer_json]:
                if p.exists(): p.unlink()
            if out_embeddings_dir.exists(): shutil.rmtree(out_embeddings_dir)

        self._ensure_talker_predictor(base_dir, out_talker, out_predictor, out_embeddings_dir)
        self._ensure_embeddings(base_dir, out_embeddings_dir)
        self._ensure_tokenizer_json(base_dir, out_tokenizer_json)

        if onnx_source_dir.resolve() != models_dir.resolve():
            self._copy_onnx(onnx_source_dir, models_dir)
        else:
            self._ensure_onnx_artifacts(base_dir, models_dir, out_codec_encoder, out_speaker_encoder, out_decoder)

        self.eprint(f"\n[done] Setup complete for: {cfg.name}")
        return 0

    def _ensure_talker_predictor(self, base_dir: Path, out_talker: Path, out_predictor: Path, out_embeddings_dir: Path) -> None:
        if out_talker.exists() and out_predictor.exists():
            return
        cmd = [
            sys.executable,
            "-m",
            "lunavox.model.conversion.convert_talker_predictor_llama",
            "--input",
            str(base_dir),
            "--out-talker",
            str(out_talker),
            "--out-predictor",
            str(out_predictor),
            "--embeddings-dir",
            str(out_embeddings_dir),
        ]
        self.run_cmd(cmd, cwd=self.root)

    def _ensure_embeddings(self, base_dir: Path, out_embeddings_dir: Path) -> None:
        if (out_embeddings_dir / "text_embedding_projected.npy").exists():
            return
        cmd = [
            sys.executable,
            "-m",
            "lunavox.model.conversion.export_embeddings",
            "--input",
            str(base_dir),
            "--output",
            str(out_embeddings_dir),
        ]
        self.run_cmd(cmd, cwd=self.root)

    def _ensure_tokenizer_json(self, base_dir: Path, out_tokenizer_json: Path) -> None:
        if out_tokenizer_json.exists(): return
        src = base_dir / "tokenizer.json"
        out_tokenizer_json.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, out_tokenizer_json)
        else:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True, fix_mistral_regex=True)
            tok.backend_tokenizer.save(str(out_tokenizer_json))

    def _copy_onnx(self, onnx_base_dir: Path, models_dir: Path) -> None:
        onnx_base_dir = onnx_base_dir.resolve()
        for fname in ["qwen3_tts_codec_encoder.fp16.onnx", "qwen3_tts_speaker_encoder.fp16.onnx", "qwen3_tts_decoder.fp16.onnx"]:
            src = onnx_base_dir / fname
            dst = models_dir / fname
            if dst.exists(): continue
            if not src.exists(): raise RuntimeError(f"ONNX source missing: {src}")
            shutil.copy2(src, dst)

    def _ensure_onnx_artifacts(self, base_dir, models_dir, out_codec, out_speaker, out_decoder) -> None:
        stage_to_output = {"codec_encoder": out_codec, "speaker_encoder": out_speaker, "decoder": out_decoder}
        for stage, artifact in stage_to_output.items():
            if not artifact.exists():
                self._run_onnx_stage(stage, base_dir, models_dir)

        self._run_onnx_stage("quantize", base_dir, models_dir)

        self.run_cmd(
            [
                sys.executable,
                "-m",
                "lunavox.model.conversion.validate_onnx_models",
                "--models-dir",
                str(models_dir),
            ],
            cwd=self.root,
        )

        # Cleanup fp32
        for name in ["qwen3_tts_codec_encoder.fp32.onnx", "qwen3_tts_speaker_encoder.fp32.onnx", "qwen3_tts_decoder.fp32.onnx"]:
            p = models_dir / name
            if p.exists():
                p.unlink()
            pd = models_dir / (name + ".data")
            if pd.exists():
                pd.unlink()

    def _run_onnx_stage(self, stage, base_dir, models_dir):
        cmd = [
            sys.executable,
            "-m",
            "lunavox.model.conversion.export_onnx_models",
            "--base-dir",
            str(base_dir),
            "--output-dir",
            str(models_dir),
            "--stage",
            stage,
            "--enable-quant",
        ]
        self.run_cmd(cmd, cwd=self.root)
