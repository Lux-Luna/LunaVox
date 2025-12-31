import os
import sys
import time
import json
import logging
import statistics
import wave
try:
    import pynvml
except ImportError:
    pynvml = None
from pathlib import Path
from collections import defaultdict

# Setup Benchmarking Environment
BENCHMARK_DIR = Path(__file__).parent.resolve()
REPO_ROOT = BENCHMARK_DIR.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))

# Local Imports
import benchmark_utils as utils
from lunavox_tts.Utils.EnvManager import env_manager
from lunavox_tts.Utils.PerformanceMonitor import monitor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Benchmark")

# ==========================================
# TEST CONFIGURATION
# ==========================================
# Run single or multiple languages (Options: "zh", "en", "ja")
TEST_LANGUAGES = ["zh", "en", "ja"]

# Run single or multiple model versions (Options: "v2", "v2pp")
TEST_VERSIONS = ["v2", "v2pp"]

# Warmup rounds and actual test rounds
WARMUP_ROUNDS = 1
TEST_ROUNDS = 1

# Project running mode ("cpu" or "gpu")
RUN_MODE = "cpu"
# ==========================================

# Reference audio configuration
REFERENCE_CONFIG = {
    "zh": {
        "audio_dir": REPO_ROOT / "CharacterData" / "audio" / "Chinese",
        "target_text": "你好，这是一次中文语音合成测试。",
        "specific_file": "不过我相信，拯救世界树的关键就在其中，所以我一直没有放弃对它的解读。.wav"
    },
    "en": {
        "audio_dir": REPO_ROOT / "CharacterData" / "audio" / "English",
        "target_text": "Hi, this is lunavox speaking English",
        "specific_file": "First get into position like this, then move like that. Yep, thats it..wav"
    },
    "ja": {
        "audio_dir": REPO_ROOT / "CharacterData" / "audio" / "Japanese",
        "target_text": "こんにちは、ルナヴォックスです。",
        "specific_file": "私は天使なんかじゃないわ。病院なんてないわよ。誰も病まないから。みんな死んでるから。.wav"
    }
}

VERSION_MAP = {
    "v2": {"path": REPO_ROOT / "CharacterData" / "model" / "v2" / "pretrained", "name": "benchmark_v2"},
    "v2pp": {"path": REPO_ROOT / "CharacterData" / "model" / "v2_pro_plus" / "pretrained", "name": "benchmark_v2pp"}
}

def get_audio_duration(path):
    try:
        with wave.open(str(path), 'rb') as f:
            return f.getnframes() / float(f.getframerate())
    except: return 0

def run_benchmark():
    # 1. Environment Check
    env_manager.set_mode(RUN_MODE)
    env_manager.set_developer_mode(True)
    if not env_manager.ensure_environment():
        logger.info(f"\n[INFO] Environment updated. Please restart script.")
        return

    import lunavox_tts as lunavox
    os.environ['HUBERT_MODEL_PATH'] = str(REPO_ROOT / 'TTSData' / 'chinese-hubert-base' / 'chinese-hubert-base.onnx')
    
    device_info = utils.get_device_info(pynvml if (pynvml and RUN_MODE == "gpu") else None, RUN_MODE)
    all_final_reports = []

    logger.info("=" * 60)
    logger.info(f"LunaVox TTS Benchmark & Resource Monitoring (Optimized)")
    logger.info(f"Mode: {RUN_MODE.upper()} | Device: {device_info}")
    logger.info(f"Warmup: {WARMUP_ROUNDS} | Rounds: {TEST_ROUNDS}")
    logger.info("NOTE: 'Total Latency' isolates inference from Disk I/O speed.")
    logger.info("=" * 60)

    for version in TEST_VERSIONS:
        if version not in VERSION_MAP: continue
        ver_info = VERSION_MAP[version]
        
        logger.info(f"\n[STARTING VERSION] {version}")
        try:
            lunavox.load_character(ver_info["name"], str(ver_info["path"]))
        except Exception as e:
            logger.error(f"Failed loading {version}: {e}")
            continue

        for lang in TEST_LANGUAGES:
            if lang not in REFERENCE_CONFIG: continue
            conf = REFERENCE_CONFIG[lang]
            logger.info(f"--- Language: {lang.upper()} ---")
            
            # Setup path
            audio_path = conf["audio_dir"] / conf["specific_file"]
            if not audio_path.exists():
                audio_path = next(conf["audio_dir"].glob("*.wav"), None)
            
            try:
                lunavox.set_reference_audio(ver_info["name"], str(audio_path), audio_path.stem, audio_language=lang)
            except Exception as e:
                logger.error(f"Failed setting reference: {e}")
                continue

            # Warmup
            if WARMUP_ROUNDS > 0:
                logger.info(f"Warmup ({WARMUP_ROUNDS} rounds)...")
                for _ in range(WARMUP_ROUNDS):
                    lunavox.tts(ver_info["name"], conf["target_text"], play=False, language=lang)

            # Benchmark Loop
            logger.info(f"Benchmarking ({TEST_ROUNDS} rounds)...")
            round_data = []
            
            for i in range(TEST_ROUNDS):
                output_dir = BENCHMARK_DIR / "audio_output" / version / lang / RUN_MODE
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f"round_{i+1}.wav"
                
                # PRE-STEP: Clear buffer to ensure isolated measurement
                monitor.get_buffer()
                monitor.set_buffering(True)
                
                # INFERENCE
                lunavox.tts(ver_info["name"], conf["target_text"], play=False, language=lang, save_path=str(save_path))
                
                # POST-STEP: Retrieve metrics
                metrics = monitor.get_buffer()
                monitor.set_buffering(False)
                
                # Process metrics for this round
                perf_entries = [m for m in metrics if m["type"] == "perf"]
                total_entry = next((m for m in perf_entries if m["task"] == "Total TTS Latency"), None)
                
                if not total_entry: continue
                
                audio_dur = get_audio_duration(save_path)
                rtf = (total_entry["duration_ms"] / 1000.0) / audio_dur if audio_dur > 0 else 0
                
                entry_data = {
                    "latency": total_entry["duration_ms"],
                    "rtf": rtf,
                    "ram": total_entry["mem_rss_mb"],
                    "vram": total_entry["vram_mb"],
                    "components": {m["task"]: m["duration_ms"] for m in perf_entries if m["category"] == "LINK_DETAIL"}
                }
                round_data.append(entry_data)
                
                vram_str = f"| VRAM: {entry_data['vram']:.1f}MB" if RUN_MODE == "gpu" else ""
                logger.info(f" Round {i+1:02d}: {entry_data['latency']:.1f}ms | RTF: {rtf:.4f} | RAM: {entry_data['ram']:.1f}MB {vram_str}")

            # Aggregate statistics
            if round_data:
                comp_avgs = defaultdict(list)
                for rd in round_data:
                    for k, v in rd["components"].items(): comp_avgs[k].append(v)
                
                final_stats = {
                    "latency": utils.format_stats([rd["latency"] for rd in round_data]),
                    "rtf": utils.format_stats([rd["rtf"] for rd in round_data], 4),
                    "ram": utils.format_stats([rd["ram"] for rd in round_data]),
                    "vram": utils.format_stats([rd["vram"] for rd in round_data]),
                    "components": {k: statistics.mean(v) for k, v in comp_avgs.items()}
                }
                
                report = {
                    "version": version, "language": lang, "mode": RUN_MODE,
                    "statistics": final_stats, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                all_final_reports.append(report)
                
                # Save individual JSON
                json_path = BENCHMARK_DIR / "result" / version / lang / RUN_MODE / "benchmark_result.json"
                json_path.parent.mkdir(parents=True, exist_ok=True)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=4, ensure_ascii=False)

        lunavox.unload_character(ver_info["name"])

    # Reports
    utils.print_summary_terminal(all_final_reports, RUN_MODE)
    report_name = f"REPORT_{RUN_MODE.upper()}_{device_info}.md"
    utils.generate_markdown_report(all_final_reports, BENCHMARK_DIR / report_name, device_info, RUN_MODE, TEST_ROUNDS, WARMUP_ROUNDS)
    logger.info(f"Benchmark completed. Reports saved in {BENCHMARK_DIR}")

if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        logger.info("\nInterrupted.")
    except Exception as e:
        logger.error(f"\nError: {e}", exc_info=True)
