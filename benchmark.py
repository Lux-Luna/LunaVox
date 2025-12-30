import os
import sys
import time
import json
import logging
import statistics
import threading
import subprocess
from pathlib import Path

# Configure logging at the very beginning to ensure early messages are visible
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Benchmark")

# Attempt to import performance monitoring dependencies
try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
except (ImportError, Exception):
    pynvml = None

# ==========================================
# TEST CONFIGURATION (Set via code here)
# ==========================================
# Run single or multiple languages (Options: "zh", "en", "ja")
TEST_LANGUAGES = ["zh", "en", "ja"]

# Run single or multiple model versions (Options: "v2", "v2pp")
TEST_VERSIONS = ["v2", "v2pp"]

# Warmup rounds and actual test rounds
WARMUP_ROUNDS = 1
TEST_ROUNDS = 5

# Project running mode ("cpu" or "gpu")
RUN_MODE = "cpu"
# ==========================================

# Add src to sys.path for project module imports
REPO_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))

# Initialize environment manager before importing lunavox
from lunavox_tts.Utils.EnvManager import env_manager
env_manager.set_mode(RUN_MODE)
env_manager.set_developer_mode(True)  # Automatically enter developer mode

# Ensure runtime matches the selected mode
if not env_manager.ensure_environment():
    logger.info(f"\n[INFO] Environment updated to {RUN_MODE.upper()} mode. Please restart this script.")
    sys.exit(0)

import lunavox_tts as lunavox

# Environment variables
os.environ['HUBERT_MODEL_PATH'] = str(REPO_ROOT / 'TTSData' / 'chinese-hubert-base' / 'chinese-hubert-base.onnx')

# Reference audio and synthesis target configuration per language
REFERENCE_CONFIG = {
    "zh": {
        "audio_dir": REPO_ROOT / "CharacterData" / "audio_resources" / "Chinese",
        "target_text": "你好，这是一次中文语音合成测试。",
        "specific_file": "不过我相信，拯救世界树的关键就在其中，所以我一直没有放弃对它的解读。.wav"
    },
    "en": {
        "audio_dir": REPO_ROOT / "CharacterData" / "audio_resources" / "English",
        "target_text": "Hi, this is lunavox speaking English",
        "specific_file": None
    },
    "ja": {
        "audio_dir": REPO_ROOT / "CharacterData" / "audio_resources" / "Japanese",
        "target_text": "こんにちは、ルナヴォックスです。",
        "specific_file": "私は天使なんかじゃないわ。病院なんてないわよ。誰も病ま难いから。みんな死んでるから。.wav"
    }
}

# Model version mapping
VERSION_MAP = {
    "v2": {
        "path": REPO_ROOT / "CharacterData" / "character_model" / "v2" / "pretrained",
        "name": "benchmark_v2"
    },
    "v2pp": {
        "path": REPO_ROOT / "CharacterData" / "character_model" / "v2_pro_plus" / "pretrained",
        "name": "benchmark_v2pp"
    }
}

class ResourceTracker:
    """Resource monitoring class for sampling RAM and VRAM usage."""
    def __init__(self, interval=0.05):
        self.interval = interval
        self.keep_running = False
        self.thread = None
        self.results = {"ram": [], "vram": []}
        self.process = psutil.Process(os.getpid()) if psutil else None
        self.gpu_handle = None
        
        if pynvml and RUN_MODE == "gpu":
            try:
                # Always re-init NVML in case it was initialized late
                try:
                    pynvml.nvmlInit()
                except:
                    pass
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.gpu_handle = None

    def _sample_once(self):
        """Take a single sample of RAM and VRAM."""
        if self.process:
            try:
                ram = self.process.memory_info().rss / (1024 * 1024)
                self.results["ram"].append(ram)
            except Exception:
                pass
        
        if self.gpu_handle:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                vram = info.used / (1024 * 1024)
                self.results["vram"].append(vram)
            except Exception:
                pass

    def _sample(self):
        while self.keep_running:
            self._sample_once()
            time.sleep(self.interval)

    def start(self):
        self.results = {"ram": [], "vram": []}
        # Take an initial sample immediately
        self._sample_once()
        self.keep_running = True
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self):
        self.keep_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        # Take a final sample
        self._sample_once()
        
        stats = {}
        for key in ["ram", "vram"]:
            data = self.results[key]
            if data:
                stats[f"{key}_avg_mb"] = round(statistics.mean(data), 2)
                stats[f"{key}_peak_mb"] = round(max(data), 2)
            else:
                stats[f"{key}_avg_mb"] = 0
                stats[f"{key}_peak_mb"] = 0
        return stats

def get_dir_size(path):
    """Calculate total size of all files in a directory (MB)."""
    total = 0
    path = Path(path)
    if not path.exists():
        return 0
    for f in path.rglob('*'):
        if f.is_file():
            total += f.stat().st_size
    return round(total / (1024 * 1024), 2)

def get_dependency_size(version, lang):
    """
    Calculate total size of core dependencies for the current version and language.
    RoBERTa is only included for Chinese (zh) or Hybrid tests.
    """
    sizes = {}
    # 1. Base TTS Data (HuBERT, G2P, SV, etc.)
    sizes["tts_data"] = get_dir_size(REPO_ROOT / "TTSData")
    
    # 2. RoBERTa Model (Only for zh or hybrid)
    if lang in ["zh", "hybrid"]:
        sizes["roberta"] = get_dir_size(REPO_ROOT / "RoBERTa")
    
    # 3. Version-specific character model
    if version in VERSION_MAP:
        sizes["character_model"] = get_dir_size(VERSION_MAP[version]["path"])
    
    total = sum(sizes.values())
    return total, sizes

def get_reference_info(lang):
    """Get reference audio path and stem text."""
    config = REFERENCE_CONFIG[lang]
    audio_path = None
    if config["specific_file"]:
        temp_path = config["audio_dir"] / config["specific_file"]
        if temp_path.exists():
            audio_path = temp_path
    if audio_path is None:
        wavs = list(config["audio_dir"].glob("*.wav"))
        if not wavs:
            raise FileNotFoundError(f"No .wav files found in {config['audio_dir']}")
        audio_path = wavs[0]
    return str(audio_path), audio_path.stem

def print_summary_table(all_results):
    """Prints a formatted summary table of all benchmark results."""
    if not all_results:
        return

    logger.info("\n" + "=" * 110)
    logger.info(f"{'BENCHMARK SUMMARY TABLE':^110}")
    logger.info("-" * 110)
    header = f"{'Ver':<8} | {'Lang':<5} | {'Time(avg)':<12} | {'Time(min)':<12} | {'Time(max)':<12} | {'RAM(peak)':<12} | {'VRAM(peak)':<12}"
    logger.info(header)
    logger.info("-" * 110)

    for res in all_results:
        v = res['version']
        l = res['language'].upper()
        t_avg = f"{res['statistics']['time']['mean_ms']:.1f} ms"
        t_min = f"{res['statistics']['time']['min_ms']:.1f} ms"
        t_max = f"{res['statistics']['time']['max_ms']:.1f} ms"
        r_peak = f"{res['statistics']['memory']['ram_peak_max_mb']:.1f} MB"
        v_peak = f"{res['statistics']['memory']['vram_peak_max_mb']:.1f} MB" if RUN_MODE == "gpu" else "N/A"
        
        row = f"{v:<8} | {l:<5} | {t_avg:<12} | {t_min:<12} | {t_max:<12} | {r_peak:<12} | {v_peak:<12}"
        logger.info(row)
    logger.info("=" * 110 + "\n")

def run_benchmark():
    # Check for developer mode dependencies and provide installation instructions if missing
    # We explicitly check if developer_mode is enabled in env_manager
    if env_manager.get_developer_mode():
        global psutil, pynvml
        
        deps_to_check = []
        if psutil is None:
            deps_to_check.append(("psutil", "psutil"))
        if pynvml is None and RUN_MODE == "gpu":
            deps_to_check.append(("nvidia-ml-py", "pynvml"))
            
        for pkg_name, module_name in deps_to_check:
            logger.info(f"\n[DEVELOPER MODE] Optional dependency '{pkg_name}' is missing for full monitoring.")
            try:
                choice = input(f"Would you like to install '{pkg_name}' now? (y/n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = 'n'
                
            if choice == 'y':
                logger.info(f"Installing {pkg_name}...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
                    logger.info(f"Successfully installed {pkg_name}.")
                    # Re-import the module and update global reference
                    if module_name == "psutil":
                        import psutil as ps
                        psutil = ps
                    elif module_name == "pynvml":
                        import pynvml as nv
                        pynvml = nv
                        try:
                            pynvml.nvmlInit()
                        except:
                            pass
                except Exception as e:
                    logger.error(f"Failed to install {pkg_name}: {e}")
            else:
                logger.info(f"Skipping installation of {pkg_name}. Resource tracking for this component will be disabled.")

    # Non-NVIDIA GPU error check
    if RUN_MODE == "gpu":
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" not in available_providers:
                logger.error("\n" + "=" * 60)
                logger.error("[GPU MODE ERROR]")
                logger.error("You are attempting to run in GPU mode, but 'CUDAExecutionProvider' is not available.")
                logger.error("Reasons might include:")
                logger.error("1. You don't have an NVIDIA GPU.")
                logger.error("2. 'onnxruntime-gpu' is not installed correctly.")
                logger.error("3. CUDA/cuDNN drivers are missing or incompatible.")
                logger.error("\nFor non-NVIDIA users (e.g., AMD, Intel, or Apple Silicon), ")
                logger.error("please use 'cpu' mode as LunaVox currently optimizes GPU acceleration specifically for NVIDIA CUDA.")
                logger.error("=" * 60 + "\n")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Error checking GPU environment: {e}")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"LunaVox TTS Benchmark & Resource Monitoring Started")
    logger.info(f"Mode: {RUN_MODE.upper()} | Developer Mode: ENABLED")
    logger.info(f"Warmup: {WARMUP_ROUNDS} | Rounds: {TEST_ROUNDS}")
    logger.info("=" * 60)

    tracker = ResourceTracker()
    summary_results = []

    for version in TEST_VERSIONS:
        if version not in VERSION_MAP:
            continue
            
        ver_info = VERSION_MAP[version]
        model_name = ver_info["name"]
        model_path = str(ver_info["path"])

        logger.info(f"\n[STARTING VERSION TEST] Model Version: {version}")
        
        logger.info(f"Loading model...")
        try:
            lunavox.load_character(model_name, model_path)
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            continue

        for lang in TEST_LANGUAGES:
            if lang not in REFERENCE_CONFIG:
                continue
                
            # Calculate dependency size dynamically for current lang
            dep_total_mb, dep_details = get_dependency_size(version, lang)

            logger.info(f"\n--- Language Test: {lang.upper()} ---")
            logger.info(f"Dependency Size: {dep_total_mb:.2f} MB ({', '.join([f'{k}: {v}MB' for k,v in dep_details.items()])})")
            
            try:
                audio_path, ref_text = get_reference_info(lang)
                lunavox.set_reference_audio(model_name, audio_path, ref_text, audio_language=lang)
            except Exception as e:
                logger.error(f"Set reference audio failed: {e}")
                continue

            target_text = REFERENCE_CONFIG[lang]["target_text"]
            
            if WARMUP_ROUNDS > 0:
                logger.info(f"Executing {WARMUP_ROUNDS} warmup round(s)...")
                for _ in range(WARMUP_ROUNDS):
                    lunavox.tts(model_name, target_text, play=False, language=lang)
            
            logger.info(f"Executing {TEST_ROUNDS} benchmark rounds (monitoring resources)...")
            round_results = []
            
            for i in range(TEST_ROUNDS):
                save_audio_path = REPO_ROOT / "benchmark" / "audio_output" / version / lang / f"round_{i+1}.wav"
                
                tracker.start()
                start_ts = time.perf_counter()
                lunavox.tts(model_name, target_text, play=False, language=lang, save_path=str(save_audio_path))
                end_ts = time.perf_counter()
                res_stats = tracker.stop()
                
                duration_ms = (end_ts - start_ts) * 1000
                res_stats["time_ms"] = round(duration_ms, 2)
                res_stats["round"] = i + 1
                
                round_results.append(res_stats)
                
                mem_info = f"RAM: {res_stats['ram_peak_mb']:.1f}MB"
                if RUN_MODE == "gpu" and res_stats['vram_peak_mb'] > 0:
                    mem_info += f", VRAM: {res_stats['vram_peak_mb']:.1f}MB"
                logger.info(f"  Round {i+1}: {duration_ms:.2f} ms | {mem_info}")

            # Statistics
            times = [r["time_ms"] for r in round_results]
            ram_peaks = [r["ram_peak_mb"] for r in round_results]
            ram_avgs = [r["ram_avg_mb"] for r in round_results]
            vram_peaks = [r["vram_peak_mb"] for r in round_results]
            vram_avgs = [r["vram_avg_mb"] for r in round_results]

            stats = {
                "time": {
                    "mean_ms": round(statistics.mean(times), 2),
                    "min_ms": round(min(times), 2),
                    "max_ms": round(max(times), 2)
                },
                "memory": {
                    "ram_peak_max_mb": round(max(ram_peaks), 2),
                    "ram_avg_mean_mb": round(statistics.mean(ram_avgs), 2),
                    "vram_peak_max_mb": round(max(vram_peaks), 2),
                    "vram_avg_mean_mb": round(statistics.mean(vram_avgs), 2)
                }
            }
            
            final_report = {
                "version": version,
                "language": lang,
                "mode": RUN_MODE,
                "dependency_size_mb": dep_total_mb,
                "dependency_details": dep_details,
                "warmup_rounds": WARMUP_ROUNDS,
                "test_rounds": TEST_ROUNDS,
                "reference_audio": os.path.basename(audio_path),
                "target_text": target_text,
                "rounds": round_results,
                "statistics": stats,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            summary_results.append(final_report)

            result_json_path = REPO_ROOT / "benchmark" / "result" / version / lang / "benchmark_result.json"
            with open(result_json_path, "w", encoding="utf-8") as f:
                json.dump(final_report, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Average: {stats['time']['mean_ms']} ms | Peak RAM: {stats['memory']['ram_peak_max_mb']} MB")
            logger.info(f"Result saved to: {result_json_path}")

        lunavox.unload_character(model_name)

    # Print the final summary table
    print_summary_table(summary_results)

    logger.info("\n" + "=" * 60)
    logger.info("Benchmark and resource monitoring completed!")
    logger.info("=" * 60)

if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user.")
    except Exception as e:
        logger.error(f"\nError occurred during benchmark: {e}", exc_info=True)
