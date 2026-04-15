from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from lunavox.core import logging as lvlog


class ModelConvertPipeline:
    def __init__(self, root: Path):
        self.root = root
        self.logs_dir = root / "logs"

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
        header = (
            f"\n{'=' * 80}\n"
            f"TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"CWD: {cwd}\n"
            f"{'=' * 80}\n"
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
            lvlog.append(f"{header}STATUS: ok\nELAPSED: {elapsed:.3f}s\n\n{output_text}\n")

        except subprocess.CalledProcessError as err:
            output_text = err.stdout or ""
            elapsed = time.time() - start
            lvlog.append(
                f"{header}"
                f"STATUS: failed\n"
                f"RETURNCODE: {err.returncode}\n"
                f"ELAPSED: {elapsed:.3f}s\n\n"
                f"{output_text}\n"
            )
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

    def convert(self, cfg, models_dir: Path, force: bool = False):
        self.eprint(f"[pipeline] Converting model: {cfg.name}")
        self.ensure_source_exists(cfg)

        base_dir = cfg.source.resolve()

        out_talker = models_dir / "qwen3_tts_talker.q5_k.gguf"
        out_predictor = models_dir / "qwen3_tts_predictor.q8_0.gguf"
        out_codec_encoder = models_dir / "qwen3_tts_codec_encoder.fp16.onnx"
        out_speaker_encoder = models_dir / "qwen3_tts_speaker_encoder.fp16.onnx"
        out_decoder = models_dir / "qwen3_tts_decoder.fp16.onnx"
        out_embeddings_dir = models_dir / "embeddings"
        out_tokenizer_json = models_dir / "tokenizer.json"
        out_model_profile = models_dir / "model_profile.json"

        models_dir.mkdir(parents=True, exist_ok=True)

        if force:
            for p in [
                out_talker,
                out_predictor,
                out_codec_encoder,
                out_speaker_encoder,
                out_decoder,
                out_tokenizer_json,
            ]:
                if p.exists():
                    p.unlink()
            if out_embeddings_dir.exists():
                shutil.rmtree(out_embeddings_dir)

        self._ensure_talker_predictor(base_dir, out_talker, out_predictor, out_embeddings_dir)
        self._ensure_embeddings(base_dir, out_embeddings_dir)
        self._ensure_tokenizer_json(base_dir, out_tokenizer_json)
        self._write_model_profile(cfg.name, base_dir, out_model_profile)

        # For custom/design models, we only convert talker, predictor and codec_decoder.
        # Codec encoder and speaker encoder are not converted or reused.
        if cfg.name in ("custom", "design", "custom_small"):
            # Only ensure decoder
            self._ensure_onnx_artifacts(
                base_dir,
                models_dir,
                out_codec_encoder,
                out_speaker_encoder,
                out_decoder,
                skip_encoders=True,
            )
        else:
            self._ensure_onnx_artifacts(
                base_dir, models_dir, out_codec_encoder, out_speaker_encoder, out_decoder
            )

        self.eprint(f"\n[done] Conversion complete for: {cfg.name}")
        return 0

    def _load_json(self, path: Path) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _normalize_model_type(self, cfg_name: str, raw_type: str) -> str:
        r = (raw_type or "").strip().lower()
        if r in {"base"}:
            return "base"
        if r in {"custom", "customvoice", "custom_voice"}:
            return "custom"
        if r in {"design", "voice_design", "voicedesign"}:
            return "design"
        if cfg_name.startswith("base"):
            return "base"
        if cfg_name.startswith("custom"):
            return "custom"
        if cfg_name.startswith("design"):
            return "design"
        raise RuntimeError(
            f"Unable to derive model_type from cfg_name='{cfg_name}' raw_type='{raw_type}'"
        )

    def _normalize_model_size(self, cfg_name: str, raw_size: str) -> str:
        s = (raw_size or "").strip().lower()
        if s:
            compact = s.replace("_", "").replace("-", "")
            if compact in {"1.7b", "1b7", "17b"}:
                return "1.7b"
            if compact in {"0.6b", "0b6", "06b"}:
                return "0.6b"
            return s
        return "0.6b" if cfg_name.endswith("_small") else "1.7b"

    def _instruct_support(self, model_type: str, model_size: str) -> bool:
        if model_type == "base":
            return False
        return not model_size.startswith("0.6")

    def _language_aliases(self, language_name: str) -> list[str]:
        key = str(language_name).strip().lower()
        alias_table = {
            "english": ["en"],
            "russian": ["ru"],
            "chinese": ["zh"],
            "japanese": ["ja"],
            "korean": ["ko"],
            "german": ["de"],
            "french": ["fr"],
            "spanish": ["es"],
            "italian": ["it"],
            "portuguese": ["pt"],
        }
        out = [key]
        out.extend(alias_table.get(key, []))
        return out

    def _write_model_profile(self, cfg_name: str, base_dir: Path, out_path: Path) -> None:
        config_path = base_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found for profile generation: {config_path}")
        cfg = self._load_json(config_path)
        talker_cfg = dict(cfg.get("talker_config", {}))
        code_pred_cfg = dict(talker_cfg.get("code_predictor_config", {}))
        gen_cfg = {}
        for gen_cfg_path in (
            base_dir / "generation_config.json",
            base_dir / "generate_config.json",
        ):
            if gen_cfg_path.exists():
                gen_cfg = self._load_json(gen_cfg_path)
                break

        model_type = self._normalize_model_type(cfg_name, str(cfg.get("tts_model_type", "")))
        model_size = self._normalize_model_size(cfg_name, str(cfg.get("tts_model_size", "")))

        raw_speaker_map_src = dict(talker_cfg.get("spk_id", {}) or {})
        raw_speaker_map = {str(k).lower(): int(v) for k, v in raw_speaker_map_src.items()}
        raw_speaker_dialect_src = dict(talker_cfg.get("spk_is_dialect", {}) or {})
        raw_speaker_dialect = {str(k).lower(): v for k, v in raw_speaker_dialect_src.items()}
        raw_language_map = dict(talker_cfg.get("codec_language_id", {}) or {})

        speaker_names = sorted(raw_speaker_map.keys())
        speaker_ids = [raw_speaker_map[name] for name in speaker_names]
        speaker_dialects = []
        for name in speaker_names:
            dialect = raw_speaker_dialect.get(name, False)
            if isinstance(dialect, str):
                speaker_dialects.append(dialect.lower())
            else:
                speaker_dialects.append("false")

        lang_pairs: list[tuple[str, int]] = []
        for raw_name, raw_id in raw_language_map.items():
            for alias in self._language_aliases(str(raw_name)):
                lang_pairs.append((alias, int(raw_id)))
        lang_pairs = sorted(set(lang_pairs), key=lambda x: x[0])
        language_names = [name for name, _ in lang_pairs]
        language_ids = [lid for _, lid in lang_pairs]

        codec_vocab_size = int(code_pred_cfg.get("vocab_size", 2048))
        codec_num_codebooks = int(talker_cfg.get("num_code_groups", 16))
        if codec_num_codebooks != 16:
            raise RuntimeError(
                f"Unsupported talker num_code_groups={codec_num_codebooks}; current LunaVox runtime requires 16"
            )
        predictor_vocab_size = codec_vocab_size * max(1, codec_num_codebooks - 1)
        talker_vocab_size = int(talker_cfg.get("vocab_size", 3072))

        talker_train_ctx = int(talker_cfg.get("max_position_embeddings", 32768))
        # Runtime cap is intentionally much smaller than training context to reduce KV memory.
        talker_runtime_cap = min(2048, max(1, talker_train_ctx))
        # Predictor generation uses <= codec_num_codebooks tokens per frame; keep fixed runtime ctx.
        predictor_runtime_ctx = 256

        profile = {
            "version": 1,
            "model_type": model_type,
            "model_size": model_size,
            "instruct_support": self._instruct_support(model_type, model_size),
            "talker_n_ctx": talker_runtime_cap,
            "talker_n_ctx_train": talker_train_ctx,
            "predictor_n_ctx": predictor_runtime_ctx,
            "codec_id_start": 0,
            "codec_id_end": codec_vocab_size,
            "codec_num_codebooks": codec_num_codebooks,
            "predictor_vocab_size": predictor_vocab_size,
            "codec_pad_id": int(talker_cfg.get("codec_pad_id", 2148)),
            "codec_bos_id": int(talker_cfg.get("codec_bos_id", 2149)),
            "codec_eos_id": int(talker_cfg.get("codec_eos_token_id", 2150)),
            "codec_think_id": int(talker_cfg.get("codec_think_id", 2154)),
            "codec_nothink_id": int(talker_cfg.get("codec_nothink_id", 2155)),
            "codec_think_bos_id": int(talker_cfg.get("codec_think_bos_id", 2156)),
            "codec_think_eos_id": int(talker_cfg.get("codec_think_eos_id", 2157)),
            "tts_pad_id": int(cfg.get("tts_pad_token_id", 151671)),
            "tts_bos_id": int(cfg.get("tts_bos_token_id", 151672)),
            "tts_eos_id": int(cfg.get("tts_eos_token_id", 151673)),
            "min_new_tokens": int(gen_cfg.get("min_new_tokens", 2)),
            "suppress_from": max(0, talker_vocab_size - 1024),
            "suppress_to": talker_vocab_size,
            "speaker_names": speaker_names,
            "speaker_ids": speaker_ids,
            "speaker_dialects": speaker_dialects,
            "language_names": language_names,
            "language_ids": language_ids,
            # Keep deterministic quality defaults aligned with LunaVox runtime policy.
            "default_max_new_tokens": 400,
            "default_temperature": 0.6,
            "default_top_p": float(gen_cfg.get("top_p", 1.0)),
            "default_top_k": int(gen_cfg.get("top_k", 50)),
            "default_repetition_penalty": float(gen_cfg.get("repetition_penalty", 1.05)),
            "default_predictor_do_sample": bool(gen_cfg.get("subtalker_dosample", True)),
            "default_predictor_temperature": 0.6,
            "default_predictor_top_p": float(gen_cfg.get("subtalker_top_p", 1.0)),
            "default_predictor_top_k": int(gen_cfg.get("subtalker_top_k", 50)),
            "default_seed": 42,
            "default_predictor_seed": 45,
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

    def _ensure_talker_predictor(
        self, base_dir: Path, out_talker: Path, out_predictor: Path, out_embeddings_dir: Path
    ) -> None:
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
        cfg = self._load_json(base_dir / "config.json")
        talker_cfg = cfg.get("talker_config", {}) if isinstance(cfg, dict) else {}
        codec_num_codebooks = int(talker_cfg.get("num_code_groups", 16))
        required = [out_embeddings_dir / "text_embedding_projected.npy"]
        required.extend(
            out_embeddings_dir / f"codec_embedding_{i}.npy" for i in range(codec_num_codebooks)
        )

        if self._model_requires_projection(base_dir):
            required.extend(
                [
                    out_embeddings_dir / "proj_weight.npy",
                    out_embeddings_dir / "proj_bias.npy",
                ]
            )

        require_projection = self._model_requires_projection(base_dir)
        if all(p.exists() for p in required):
            if self._embeddings_dtype_contract_ok(
                out_embeddings_dir=out_embeddings_dir,
                codec_num_codebooks=codec_num_codebooks,
                require_projection=require_projection,
            ):
                return
            self.eprint("[fix] embeddings dtype mismatch detected; regenerating assets")

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

        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Embedding export did not produce required files: " + ", ".join(missing)
            )
        if not self._embeddings_dtype_contract_ok(
            out_embeddings_dir=out_embeddings_dir,
            codec_num_codebooks=codec_num_codebooks,
            require_projection=require_projection,
        ):
            raise RuntimeError(
                "Embedding export produced unsupported dtypes; "
                "expected codec/proj in float32 and text table in float16/float32"
            )

    def _model_requires_projection(self, base_dir: Path) -> bool:
        cfg_path = base_dir / "config.json"
        if not cfg_path.exists():
            return False
        cfg = self._load_json(cfg_path)
        talker_cfg = dict(cfg.get("talker_config", {}))
        code_pred_cfg = dict(talker_cfg.get("code_predictor_config", {}))
        talker_hidden = int(talker_cfg.get("hidden_size", 0))
        predictor_hidden = int(code_pred_cfg.get("hidden_size", talker_hidden))
        return talker_hidden > 0 and predictor_hidden > 0 and talker_hidden != predictor_hidden

    def _embeddings_dtype_contract_ok(
        self,
        out_embeddings_dir: Path,
        codec_num_codebooks: int,
        require_projection: bool,
    ) -> bool:
        def _dtype(path: Path) -> str:
            arr = np.load(path, mmap_mode="r")
            return str(arr.dtype)

        text_path = out_embeddings_dir / "text_embedding_projected.npy"
        if _dtype(text_path) not in {"float16", "float32"}:
            return False

        for i in range(codec_num_codebooks):
            codec_path = out_embeddings_dir / f"codec_embedding_{i}.npy"
            if _dtype(codec_path) != "float32":
                return False

        if require_projection:
            if _dtype(out_embeddings_dir / "proj_weight.npy") != "float32":
                return False
            if _dtype(out_embeddings_dir / "proj_bias.npy") != "float32":
                return False

        return True

    def _ensure_tokenizer_json(self, base_dir: Path, out_tokenizer_json: Path) -> None:
        if out_tokenizer_json.exists():
            return
        src = base_dir / "tokenizer.json"
        out_tokenizer_json.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, out_tokenizer_json)
        else:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(
                str(base_dir), trust_remote_code=True, fix_mistral_regex=True
            )
            tok.backend_tokenizer.save(str(out_tokenizer_json))

    def _ensure_onnx_artifacts(
        self, base_dir, models_dir, out_codec, out_speaker, out_decoder, skip_encoders: bool = False
    ) -> None:
        stage_to_output = {
            "codec_encoder": out_codec,
            "speaker_encoder": out_speaker,
            "decoder": out_decoder,
        }

        stages_to_run = list(stage_to_output.keys())
        if skip_encoders:
            stages_to_run = ["decoder"]

        # Record existence for final validation
        for stage in stages_to_run:
            artifact = stage_to_output[stage]
            if not artifact.exists():
                self._run_onnx_stage(stage, base_dir, models_dir)

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
        for name in [
            "qwen3_tts_codec_encoder.fp32.onnx",
            "qwen3_tts_speaker_encoder.fp32.onnx",
            "qwen3_tts_decoder.fp32.onnx",
        ]:
            p = models_dir / name
            if p.exists():
                p.unlink()
            pd = models_dir / (name + ".data")
            if pd.exists():
                pd.unlink()

            # Cleanup unwanted int8 residues
            pi8 = models_dir / name.replace(".fp32.onnx", ".fp32.int8.onnx")
            if pi8.exists():
                pi8.unlink()
            pi8_alt = models_dir / name.replace(".fp32.onnx", ".int8.onnx")
            if pi8_alt.exists():
                pi8_alt.unlink()

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
        ]
        self.run_cmd(cmd, cwd=self.root)
