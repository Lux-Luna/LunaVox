"""
LunaVox Benchmark Runner
Configurable multi-mode, multi-environment, multi-version, multi-language testing.

Default: All environments, modes, versions, languages with 1 warmup and 1 test round.

GPU Auto-Recovery: Each environment runs in a subprocess for process isolation.
This enables automatic recovery after environment switching (e.g., CPU -> GPU).

Usage:
    python benchmark.py
    python benchmark.py --env cpu gpu --mode persona --version v2 --lang zh en --warmup 2 --rounds 5
"""

import os
import sys
import time
import json
import argparse
import logging
import statistics
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import defaultdict

try:
    import pynvml
except ImportError:
    pynvml = None

# Setup Benchmarking Environment
BENCHMARK_DIR = Path(__file__).parent.resolve()
REPO_ROOT = BENCHMARK_DIR.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))

# Local Imports
import benchmark_utils as utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Benchmark")


# ==========================================
# CONFIGURATION
# ==========================================
@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    environments: List[str] = field(default_factory=lambda: ["cpu"]) # "cpu"、"gpu"
    modes: List[str] = field(default_factory=lambda: ["persona", "reference"]) # "reference"、"persona"
    versions: List[str] = field(default_factory=lambda: ["v2", "v2pp"]) # "v2"、"v2pp"
    languages: List[str] = field(default_factory=lambda: ["zh", "en", "ja"]) # "zh"、"en"、"ja"
    warmup_rounds: int = 1
    test_rounds: int = 1
    _internal_env: Optional[str] = None
    _internal_lang: Optional[str] = None
    _result_file: Optional[str] = None


# Language-specific configuration
LANGUAGE_CONFIG = {
    "zh": {
        "persona_dir": "luna_zh",
        "audio_dir": "Chinese",
        "target_text": "你好，这是一次中文语音合成测试。"
    },
    "en": {
        "persona_dir": "luna_en",
        "audio_dir": "English",
        "target_text": "Hi, this is lunavox speaking English"
    },
    "ja": {
        "persona_dir": "luna_ja",
        "audio_dir": "Japanese",
        "target_text": "こんにちは、ルナヴォックスです。"
    }
}

# Version-specific configuration
VERSION_CONFIG = {
    "v2": {
        "model_path": "v2/pretrained",
        "char_suffix": ""
    },
    "v2pp": {
        "model_path": "v2_pro_plus/pretrained",
        "char_suffix": "_v2pp"
    }
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LunaVox Multi-Dimension Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full benchmark (default):
    python benchmark.py

  CPU-only, persona mode, all languages:
    python benchmark.py --env cpu --mode persona

  GPU v2pp English only, 3 warmup + 5 test rounds:
    python benchmark.py --env gpu --version v2pp --lang en --warmup 3 --rounds 5
        """
    )
    parser.add_argument("--env", nargs="+", choices=["cpu", "gpu"], default=None,
                        help="Environments to test (default: defined in BenchmarkConfig)")
    parser.add_argument("--mode", nargs="+", choices=["persona", "reference"], default=None,
                        help="Inference modes to test (default: defined in BenchmarkConfig)")
    parser.add_argument("--version", nargs="+", choices=["v2", "v2pp"], default=None,
                        help="Model versions to test (default: defined in BenchmarkConfig)")
    parser.add_argument("--lang", nargs="+", choices=["zh", "en", "ja"], default=None,
                        help="Languages to test (default: defined in BenchmarkConfig)")
    parser.add_argument("--warmup", type=int, default=None,
                        help="Number of warmup rounds (default: defined in BenchmarkConfig)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Number of test rounds (default: defined in BenchmarkConfig)")
    
    # Internal flag for subprocess execution
    parser.add_argument("--_internal_env", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_internal_lang", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_result_file", type=str, default=None,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Only pass arguments that were explicitly provided (not None)
    # This allows BenchmarkConfig to use its own defaults/factories
    config_kwargs = {}
    if args.env is not None:
        config_kwargs['environments'] = args.env
    if args.mode is not None:
        config_kwargs['modes'] = args.mode
    if args.version is not None:
        config_kwargs['versions'] = args.version
    if args.lang is not None:
        config_kwargs['languages'] = args.lang
    if args.warmup is not None:
        config_kwargs['warmup_rounds'] = args.warmup
    if args.rounds is not None:
        config_kwargs['test_rounds'] = args.rounds

    # Pass internal flags
    config_kwargs['_internal_env'] = args._internal_env
    config_kwargs['_internal_lang'] = args._internal_lang
    config_kwargs['_result_file'] = args._result_file

    return BenchmarkConfig(**config_kwargs)


# Fixed character name pool for cache efficiency
# Using language-based index allows cache reuse within same language
CHAR_NAME_POOL = ["bench_0", "bench_1", "bench_2"]
LANG_TO_INDEX = {"zh": 0, "en": 1, "ja": 2}


def get_character_name(version: str, lang: str, mode: str) -> str:
    """Get character name from fixed pool based on language.
    
    Using fixed pool enables LRU cache hits and prevents memory bloat
    from unique character names across test combinations.
    """
    lang_idx = LANG_TO_INDEX.get(lang, 0)
    return CHAR_NAME_POOL[lang_idx]


def resolve_reference_audio(lang: str) -> tuple:
    """Find reference audio file for a language."""
    lang_conf = LANGUAGE_CONFIG[lang]
    audio_dir = REPO_ROOT / "lunavoxData" / "CharacterData" / "audio" / lang_conf["audio_dir"]
    wav_files = list(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}")
    wav_file = wav_files[0]
    return str(wav_file), wav_file.stem


def run_single_test(
    lunavox,
    monitor,
    env: str,
    mode: str,
    version: str,
    lang: str,
    config: BenchmarkConfig,
    device_info: str
) -> Optional[Dict[str, Any]]:
    """Run a single test combination and return results."""
    lang_conf = LANGUAGE_CONFIG[lang]
    ver_conf = VERSION_CONFIG[version]

    char_name = get_character_name(version, lang, mode)
    persona_dir = str(REPO_ROOT / "lunavoxData" / "CharacterData" / "character" / lang_conf["persona_dir"])
    model_dir = str(REPO_ROOT / "lunavoxData" / "CharacterData" / "model" / ver_conf["model_path"])

    logger.info(f"\n{'='*60}")
    logger.info(f"[TEST] Env:{env.upper()} | Mode:{mode} | Version:{version} | Lang:{lang.upper()}")
    logger.info(f"{'='*60}")
    
    # --- CLEANUP: Ensure fresh state before each test ---
    from lunavox_tts.Utils.RuntimeManager import runtime_manager
    runtime_manager.cleanup_all()
    
    # Reset memory baselines for accurate measurement
    monitor.reset_baselines()

    try:
        # Load model based on mode
        if mode == "persona":
            lunavox.load_persona(char_name, persona_dir)
            lunavox.load_character(char_name, model_dir)
        else:  # reference mode
            lunavox.load_character(char_name, model_dir)
            audio_path, ref_text = resolve_reference_audio(lang)
            lunavox.set_reference_audio(char_name, audio_path, ref_text, audio_language=lang)

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

    target_text = lang_conf["target_text"]

    # Create temp directory for test outputs
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Warmup Phase
        warmup_audio_saved = False
        if config.warmup_rounds > 0:
            logger.info(f"Warmup ({config.warmup_rounds} rounds)...")
            for i in range(config.warmup_rounds):
                warmup_path = tmp_path / f"warmup_{i}.wav"
                lunavox.tts(char_name, target_text, play=False, language=lang, save_path=str(warmup_path))

                # Save first warmup audio
                if i == 0 and not warmup_audio_saved:
                    output_dir = BENCHMARK_DIR / "audio_output" / version / lang / mode / env
                    utils.save_warmup_audio(warmup_path, output_dir)
                    warmup_audio_saved = True
                    logger.info(f"  Warmup audio saved to {output_dir / 'warmup.wav'}")

        # Benchmark Phase
        logger.info(f"Benchmarking ({config.test_rounds} rounds)...")
        round_data = []

        for i in range(config.test_rounds):
            test_path = tmp_path / f"test_{i}.wav"

            # Clear buffer for isolated measurement
            monitor.get_buffer()
            monitor.set_buffering(True)

            # Run inference
            lunavox.tts(char_name, target_text, play=False, language=lang, save_path=str(test_path))

            # Retrieve metrics
            metrics = monitor.get_buffer()
            monitor.set_buffering(False)

            # Process metrics
            perf_entries = [m for m in metrics if m["type"] == "perf"]
            
            # Aggregate all "Total TTS Latency" entries (supports multi-sentence)
            total_latency_entries = [m for m in perf_entries if m["task"] == "Total TTS Latency"]

            if not total_latency_entries:
                logger.warning(f"  Round {i+1}: No metrics captured")
                continue

            # Sum latency of all chunks (sentences)
            total_latency_ms = sum(m["duration_ms"] for m in total_latency_entries)
            
            # Get Peak RAM/VRAM across all captured perf events in this round
            peak_ram = max((m["mem_rss_mb"] for m in perf_entries), default=0)
            peak_vram = max((m.get("vram_mb", 0) for m in perf_entries), default=0)

            audio_dur = utils.get_audio_duration(test_path)
            rtf = (total_latency_ms / 1000.0) / audio_dur if audio_dur > 0 else 0

            # Aggregate components by summing durations (correctly handles overwrites)
            components_map = defaultdict(float)
            for m in perf_entries:
                if m.get("category") == "LINK_DETAIL":
                    components_map[m["task"]] += m["duration_ms"]

            entry_data = {
                "latency": total_latency_ms,
                "rtf": rtf,
                "ram": peak_ram,
                "vram": peak_vram,
                "components": dict(components_map)
            }
            round_data.append(entry_data)

            vram_str = f"| VRAM: {entry_data['vram']:.1f}MB" if env == "gpu" else ""
            logger.info(
                f"  Round {i+1:02d}: {entry_data['latency']:.1f}ms | "
                f"RTF: {rtf:.4f} | RAM: {entry_data['ram']:.1f}MB {vram_str}"
            )

    # Unload model and cleanup global resources
    try:
        lunavox.unload_character(char_name)
    except Exception:
        pass
    
    # Cleanup global resources for next test isolation
    from lunavox_tts.Utils.RuntimeManager import runtime_manager
    runtime_manager.cleanup_all()

    # Aggregate statistics
    if not round_data:
        logger.warning("No valid test data collected")
        return None

    comp_avgs = defaultdict(list)
    for rd in round_data:
        for k, v in rd["components"].items():
            comp_avgs[k].append(v)

    final_stats = {
        "latency": utils.format_stats([rd["latency"] for rd in round_data]),
        "rtf": utils.format_stats([rd["rtf"] for rd in round_data], 4),
        "ram": utils.format_stats([rd["ram"] for rd in round_data]),
        "vram": utils.format_stats([rd["vram"] for rd in round_data]),
        "components": {k: round(statistics.mean(v), 2) for k, v in comp_avgs.items()}
    }

    result = {
        "environment": env,
        "mode": mode,
        "version": version,
        "language": lang,
        "warmup_rounds": config.warmup_rounds,
        "test_rounds": config.test_rounds,
        "device_info": device_info,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "statistics": final_stats
    }

    # Save individual JSON result
    result_dir = BENCHMARK_DIR / "result" / version / lang / mode / env
    utils.save_json_result(result, result_dir)
    logger.info(f"  Result saved to {result_dir / 'benchmark_result.json'}")

    return result


def run_benchmark_internal(env: str, lang: str, args: BenchmarkConfig, result_file: str) -> bool:
    """
    Internal function that runs in a subprocess for a single environment and language.
    """
    from lunavox_tts.Utils.EnvManager import env_manager

    # Set environment mode
    env_manager.set_mode(env)
    env_manager.set_developer_mode(True)

    if not env_manager.ensure_environment():
        # Environment changed, need restart
        # Write empty result to signal parent to retry
        with open(result_file, "w") as f:
            json.dump({"status": "env_changed", "env": env}, f)
        return False

    # Import after environment is set
    import lunavox_tts as lunavox
    from lunavox_tts.Utils.PerformanceMonitor import monitor

    # Set HuBERT path
    os.environ['HUBERT_MODEL_PATH'] = str(REPO_ROOT / 'lunavoxData' / 'TTSData' / 'chinese-hubert-base' / 'chinese-hubert-base.onnx')

    # Get device info
    device_info = utils.get_device_info(pynvml if (pynvml and env == "gpu") else None, env)

    # Initialize baselines BEFORE loading any models to capture model weight VRAM usage
    if env == "gpu":
        logger.info("[Monitor] Initializing VRAM baseline to capture model weights...")
        monitor._ensure_baselines()

    # Build config from args (which is actually a BenchmarkConfig object from parent)
    config = BenchmarkConfig(
        environments=[env],
        modes=args.modes,
        versions=args.versions,
        languages=args.languages,
        warmup_rounds=args.warmup_rounds,
        test_rounds=args.test_rounds
    )

    logger.info("\n" + "=" * 70)
    logger.info(f"LunaVox TTS Benchmark - {env.upper()} | {lang.upper()}")
    logger.info(f"Device: {device_info}")
    logger.info(f"Warmup: {config.warmup_rounds} | Rounds: {config.test_rounds}")
    logger.info("=" * 70)
    
    # Only run the specified language
    config.languages = [lang]

    env_results = []

    for version in config.versions:
        if version not in VERSION_CONFIG:
            continue

        for lang in config.languages:
            if lang not in LANGUAGE_CONFIG:
                continue

            for mode in config.modes:
                result = run_single_test(
                    lunavox, monitor, env, mode, version, lang, config, device_info
                )
                if result:
                    env_results.append(result)

    # Write results to file
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({"status": "success", "results": env_results}, f, ensure_ascii=False, indent=2)

    return True


def run_lang_env_in_subprocess(env: str, lang: str, config: BenchmarkConfig, max_retries: int = 3) -> List[Dict[str, Any]]:
    """
    Run benchmark for an (environment, language) pair in a subprocess.
    This ensures C-level memory (pyopenjtalk, jieba) is cleared between languages.
    """
    for attempt in range(max_retries):
        # Create temp file for results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            result_file = f.name

        try:
            # Build subprocess command
            cmd = [
                sys.executable,
                str(BENCHMARK_DIR / "benchmark.py"),
                "--_internal_env", env,
                "--_internal_lang", lang,
                "--_result_file", result_file,
                "--mode", *config.modes,
                "--version", *config.versions,
                "--warmup", str(config.warmup_rounds),
                "--rounds", str(config.test_rounds)
            ]

            logger.info(f"\n{'#'*70}")
            logger.info(f"# Starting {env.upper()} - {lang.upper()} (subprocess, attempt {attempt + 1}/{max_retries})")
            logger.info(f"{'#'*70}")

            # Run subprocess
            process = subprocess.run(
                cmd,
                cwd=str(BENCHMARK_DIR),
                capture_output=False,  # Let output show in terminal
                text=True
            )

            # Read results
            if Path(result_file).exists():
                with open(result_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if data.get("status") == "success":
                    return data.get("results", [])
                elif data.get("status") == "env_changed":
                    logger.info(f"\n[AUTO-RECOVERY] Environment changed, retrying ({attempt + 2}/{max_retries})...")
                    continue
            else:
                logger.warning(f"No result file found for {env} environment")

        except Exception as e:
            logger.error(f"Subprocess error for {env}: {e}")

        finally:
            # Clean up temp file
            try:
                Path(result_file).unlink(missing_ok=True)
            except Exception:
                pass

    logger.error(f"Failed to run {env} benchmark after {max_retries} attempts")
    return []


def run_benchmark():
    """Main benchmark runner with multi-environment support via subprocess isolation."""
    config = parse_args()

    # Check if we're running as subprocess for a specific job
    if config._internal_env and config._internal_lang:
        run_benchmark_internal(config._internal_env, config._internal_lang, config, config._result_file)
        return

    # Main orchestrator mode
    logger.info("\n" + "#" * 70)
    logger.info("#" + " " * 68 + "#")
    logger.info("#" + "LUNAVOX MULTI-DIMENSION BENCHMARK".center(68) + "#")
    logger.info("#" + " " * 68 + "#")
    logger.info("#" * 70)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Environments: {config.environments}")
    logger.info(f"  Modes: {config.modes}")
    logger.info(f"  Versions: {config.versions}")
    logger.info(f"  Languages: {config.languages}")
    logger.info(f"  Warmup Rounds: {config.warmup_rounds}")
    logger.info(f"  Test Rounds: {config.test_rounds}")

    total_tests = (
        len(config.environments) *
        len(config.modes) *
        len(config.versions) *
        len(config.languages)
    )
    logger.info(f"\nTotal test combinations: {total_tests}")

    all_results = []
    timestamp = utils.get_timestamp_str()

    # Run each [env x lang] combination in a separate subprocess for total isolation
    for env in config.environments:
        for lang in config.languages:
            job_results = run_lang_env_in_subprocess(env, lang, config)
            all_results.extend(job_results)

    # Generate final reports
    if all_results:
        # Terminal summary
        utils.print_summary_terminal(all_results)

        # Markdown report
        device_info = all_results[0].get("device_info", "Unknown")
        report_path = BENCHMARK_DIR / f"REPORT_{timestamp}.md"
        utils.generate_markdown_report(
            all_results, report_path, device_info, time.strftime("%Y-%m-%d %H:%M:%S")
        )
        logger.info(f"\nBenchmark completed!")
        logger.info(f"  Report: {report_path}")
        logger.info(f"  Results: {BENCHMARK_DIR / 'result'}")
        logger.info(f"  Audio: {BENCHMARK_DIR / 'audio_output'}")
    else:
        logger.warning("\nNo benchmark results collected.")


if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user.")
    except Exception as e:
        logger.error(f"\nBenchmark error: {e}", exc_info=True)
