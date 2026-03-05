"""
LunaVox C++ Accelerator Benchmark
Compares Python (original) vs C++ (accelerator) T2S inference loop performance.

Usage:
    conda run -n lunavox python benchmark_accelerator.py
"""

import os
import sys
import time
import json
import logging
import statistics
import wave
import platform
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("AcceleratorBenchmark")

# Paths
BENCHMARK_DIR = Path(__file__).parent.resolve()
REPO_ROOT = BENCHMARK_DIR.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(BENCHMARK_DIR))  # For lunavox_accelerator .pyd

# ======= CONFIG =======
WARMUP_ROUNDS = 3
TEST_ROUNDS = 10
LANGUAGE = "en"
RUN_MODE = "cpu"
# =======================

# Initialize environment
from lunavox_tts.Utils.EnvManager import env_manager
env_manager.set_mode(RUN_MODE)

import lunavox_tts as lunavox
from lunavox_tts.Core.Inference import tts_client, LunaVoxEngine
from lunavox_tts.ModelManager import model_manager

# Reference audio config
REFERENCE_CONFIG = {
    "en": {
        "audio_dir": REPO_ROOT / "CharacterData" / "audio_resources" / "English",
        "target_text": "Hi, this is lunavox speaking English",
        "specific_file": "First get into position like this, then move like that. Yep, thats it..wav"
    },
    "ja": {
        "audio_dir": REPO_ROOT / "CharacterData" / "audio_resources" / "Japanese",
        "target_text": "こんにちは、ルナヴォックスです。",
        "specific_file": None
    },
}

MODEL_DIR = str(REPO_ROOT / "CharacterData" / "character_model" / "v2" / "pretrained")
MODEL_NAME = "benchmark_acc"


def get_audio_duration(path):
    try:
        with wave.open(str(path), 'rb') as f:
            return f.getnframes() / float(f.getframerate())
    except:
        return 0


def get_cpu_name():
    cpu = platform.processor()
    if sys.platform == "win32":
        try:
            import subprocess
            output = subprocess.check_output(["wmic", "cpu", "get", "name"]).decode().split('\n')
            if len(output) > 1:
                cpu = output[1].strip()
        except:
            pass
    return cpu


def benchmark_python_path(target_text, lang, rounds):
    """Benchmark using the original Python T2S loop."""
    os.environ["LUNAVOX_USE_CPP_LOOP"] = "0"
    
    results = []
    for i in range(rounds):
        output_path = BENCHMARK_DIR / "output" / f"python_round_{i+1}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        t0 = time.perf_counter()
        lunavox.tts(MODEL_NAME, target_text, play=False, language=lang, save_path=str(output_path))
        t1 = time.perf_counter()
        
        elapsed_ms = (t1 - t0) * 1000
        audio_dur = get_audio_duration(output_path)
        rtf = (elapsed_ms / 1000) / audio_dur if audio_dur > 0 else 0
        
        results.append({
            "round": i + 1,
            "time_ms": round(elapsed_ms, 2),
            "audio_duration_s": round(audio_dur, 3),
            "rtf": round(rtf, 4),
        })
        logger.info(f"  [Python] Round {i+1}: {elapsed_ms:.2f} ms | RTF: {rtf:.4f}")
    
    return results


def benchmark_cpp_path(target_text, lang, rounds):
    """Benchmark using the C++ accelerated T2S loop."""
    # Check if accelerator is available
    try:
        import lunavox_accelerator
        logger.info(f"  C++ accelerator loaded: v{lunavox_accelerator.__version__}")
    except ImportError as e:
        logger.error(f"  C++ accelerator not found: {e}")
        return None
    
    # We need to export full ONNX models (with embedded weights) for C++ to load.
    # The Python ModelManager uses in-memory FP16->FP32 patching, so we need to
    # serialize the in-memory models to temp files.
    import tempfile
    import onnx
    
    gsv_model = model_manager.get(MODEL_NAME)
    if gsv_model is None:
        logger.error("  Model not loaded for C++ benchmark")
        return None
    
    temp_dir = tempfile.mkdtemp(prefix="lunavox_acc_")
    logger.info(f"  Exporting full ONNX models to: {temp_dir}")
    
    sessions = {
        "encoder": gsv_model.T2S_ENCODER,
        "fsd": gsv_model.T2S_FIRST_STAGE_DECODER,
        "sd": gsv_model.T2S_STAGE_DECODER,
    }
    
    # Export by serializing from the ONNX Runtime sessions via onnx
    # Since we can't extract from sessions directly, we need another approach:
    # Use the model_manager's weight patching to produce full models
    from lunavox_tts.ModelManager import load_session_with_fp16_conversion, _GSVModelFile
    
    # Re-create full models in memory and save them
    model_files = {
        "encoder": ("t2s_encoder_fp32.onnx", "t2s_encoder_fp32.bin"),
        "fsd": ("t2s_first_stage_decoder_fp32.onnx", "t2s_shared_fp16.bin"),
        "sd": ("t2s_stage_decoder_fp32.onnx", "t2s_shared_fp16.bin"),
    }
    
    exported_paths = {}
    for key, (onnx_name, bin_name) in model_files.items():
        onnx_path = os.path.join(MODEL_DIR, onnx_name)
        bin_path = os.path.join(MODEL_DIR, bin_name)
        
        if os.path.exists(bin_path):
            # Load and patch the model to get full ONNX
            import numpy as np
            model_proto = onnx.load(onnx_path, load_external_data=False)
            
            fp16_data = np.fromfile(bin_path, dtype=np.float16)
            fp32_data = fp16_data.astype(np.float32)
            del fp16_data
            fp32_bytes = fp32_data.tobytes()
            del fp32_data
            
            for tensor in model_proto.graph.initializer:
                if tensor.data_location == onnx.TensorProto.EXTERNAL:
                    offset = 0
                    length = 0
                    for entry in tensor.external_data:
                        if entry.key == 'offset':
                            offset = int(entry.value)
                        elif entry.key == 'length':
                            length = int(entry.value)
                    
                    if offset + length <= len(fp32_bytes):
                        tensor.raw_data = fp32_bytes[offset:offset+length]
                        del tensor.external_data[:]
                        tensor.data_location = onnx.TensorProto.DEFAULT
            
            del fp32_bytes
            
            out_path = os.path.join(temp_dir, f"{key}.onnx")
            onnx.save(model_proto, out_path)
            del model_proto
            exported_paths[key] = out_path
            logger.info(f"    Exported {key}: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")
        else:
            # No bin file, try loading the ONNX directly
            exported_paths[key] = onnx_path
    
    # Initialize C++ engine with exported models
    engine = lunavox_accelerator.T2SEngine()
    providers = ["CPUExecutionProvider"]
    
    logger.info(f"  Loading exported models into C++ engine...")
    t_load_start = time.perf_counter()
    engine.load_models(
        exported_paths["encoder"],
        exported_paths["fsd"],
        exported_paths["sd"],
        providers
    )
    t_load_end = time.perf_counter()
    logger.info(f"  Models loaded in {(t_load_end - t_load_start)*1000:.1f} ms")
    
    # Prepare inputs (same as Python path)
    import numpy as np
    from lunavox_tts.Core.TextFrontend import get_text_frontend
    from lunavox_tts.Utils.Constants import BERT_FEATURE_DIM
    from lunavox_tts.Utils.Shared import context
    
    prompt_audio = context.current_prompt_audio
    if prompt_audio is None:
        logger.error("  No reference audio set. Cannot run C++ benchmark.")
        return None
    
    frontend = get_text_frontend()
    text = "。" + target_text
    
    if lang == "en":
        ids = frontend.process_en(text)
        text_seq = np.array([ids], dtype=np.int64)
        text_bert = np.zeros((text_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)
    elif lang == "ja":
        text_seq = np.array([frontend.process_ja(text)], dtype=np.int64)
        text_bert = np.zeros((text_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)
    else:
        logger.error(f"  Language {lang} not supported in C++ benchmark yet")
        return None
    
    ref_seq = prompt_audio.phonemes_seq
    if ref_seq is None:
        logger.error("  Reference audio phonemes not available")
        return None
    ref_bert = prompt_audio.text_bert
    if ref_bert is None or ref_bert.shape[0] != ref_seq.shape[1]:
        ref_bert = np.zeros((ref_seq.shape[1], BERT_FEATURE_DIM), dtype=np.float32)
    
    ssl_content = prompt_audio.ssl_content
    
    results = []
    for i in range(rounds):
        t0 = time.perf_counter()
        semantic_tokens = engine.run_t2s(ref_seq, ref_bert, text_seq, text_bert, ssl_content)
        t1 = time.perf_counter()
        
        elapsed_ms = (t1 - t0) * 1000
        n_tokens = semantic_tokens.shape[1] if semantic_tokens is not None else 0
        
        results.append({
            "round": i + 1,
            "time_ms": round(elapsed_ms, 2),
            "n_tokens": n_tokens,
        })
        logger.info(f"  [C++] Round {i+1}: {elapsed_ms:.2f} ms | Tokens: {n_tokens}")
    
    # Cleanup temp files
    import shutil
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except:
        pass
    
    return results


def compute_stats(results, key="time_ms"):
    vals = [r[key] for r in results]
    return {
        "avg": round(statistics.mean(vals), 2),
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
        "std": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0,
    }


def run_benchmark():
    logger.info("=" * 60)
    logger.info("LunaVox C++ Accelerator Benchmark")
    logger.info(f"Mode: {RUN_MODE.upper()} | Language: {LANGUAGE.upper()}")
    logger.info(f"CPU: {get_cpu_name()}")
    logger.info(f"Warmup: {WARMUP_ROUNDS} | Test: {TEST_ROUNDS}")
    logger.info("=" * 60)
    
    # Load model
    logger.info("\n[1/5] Loading character model...")
    lunavox.load_character(MODEL_NAME, MODEL_DIR)
    
    # Set reference audio
    logger.info("[2/5] Setting reference audio...")
    config = REFERENCE_CONFIG[LANGUAGE]
    if config["specific_file"]:
        audio_path = config["audio_dir"] / config["specific_file"]
    else:
        wavs = list(config["audio_dir"].glob("*.wav"))
        audio_path = wavs[0] if wavs else None
    
    if not audio_path or not audio_path.exists():
        logger.error(f"Reference audio not found: {audio_path}")
        return
    
    lunavox.set_reference_audio(MODEL_NAME, str(audio_path), audio_path.stem, audio_language=LANGUAGE)
    
    target_text = config["target_text"]
    
    # Warmup
    logger.info(f"\n[3/5] Warmup ({WARMUP_ROUNDS} rounds)...")
    for i in range(WARMUP_ROUNDS):
        lunavox.tts(MODEL_NAME, target_text, play=False, language=LANGUAGE)
        logger.info(f"  Warmup {i+1} done")
    
    # Benchmark Python path
    logger.info(f"\n[4/5] Python path benchmark ({TEST_ROUNDS} rounds)...")
    py_results = benchmark_python_path(target_text, LANGUAGE, TEST_ROUNDS)
    py_stats = compute_stats(py_results)
    
    # Benchmark C++ path
    logger.info(f"\n[5/5] C++ path benchmark ({TEST_ROUNDS} rounds)...")
    cpp_results = benchmark_cpp_path(target_text, LANGUAGE, TEST_ROUNDS)
    
    # Generate report
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("-" * 60)
    
    logger.info(f"\n  Python Path (T2S + VITS + Audio):")
    logger.info(f"    Avg: {py_stats['avg']:.2f} ms")
    logger.info(f"    Min: {py_stats['min']:.2f} ms")
    logger.info(f"    Max: {py_stats['max']:.2f} ms")
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": get_cpu_name(),
        "mode": RUN_MODE,
        "language": LANGUAGE,
        "warmup_rounds": WARMUP_ROUNDS,
        "test_rounds": TEST_ROUNDS,
        "target_text": target_text,
        "python_path": {
            "rounds": py_results,
            "statistics": py_stats,
        },
    }
    
    if cpp_results:
        cpp_stats = compute_stats(cpp_results)
        logger.info(f"\n  C++ Path (T2S only, no VITS):")
        logger.info(f"    Avg: {cpp_stats['avg']:.2f} ms")
        logger.info(f"    Min: {cpp_stats['min']:.2f} ms")
        logger.info(f"    Max: {cpp_stats['max']:.2f} ms")
        
        # The Python path includes VITS + audio processing, C++ only does T2S
        # So we compare T2S-component only for fairness
        logger.info(f"\n  Note: C++ path measures T2S loop only (Encoder + FSD + AR loop)")
        logger.info(f"  Python path measures full TTS (T2S + VITS + Audio I/O)")
        
        report["cpp_path"] = {
            "rounds": cpp_results,
            "statistics": cpp_stats,
        }
    else:
        logger.info("\n  C++ Path: SKIPPED (plugin not available)")
        report["cpp_path"] = None
    
    logger.info("=" * 60)
    
    # Save JSON
    result_path = BENCHMARK_DIR / "benchmark_accelerator_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    logger.info(f"\nResults saved to: {result_path}")
    
    lunavox.unload_character(MODEL_NAME)


if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted.")
    except Exception as e:
        logger.error(f"\nError: {e}", exc_info=True)
