#include "lunavox_engine.h"
#include "logger.h"
#include "nvml_monitor.h"
#include "platform_utils.h"
#include "string_utils.h"
#include "audio_io.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace lunavox;

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned int) c);
                    out += buf;
                } else {
                    out.push_back((char) c);
                }
                break;
        }
    }
    return out;
}

void print_usage(const char * program) {
    fprintf(stderr, "Usage: %s [options] -m <model_dir> -t <text>\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <dir>      Model directory (required)\n");
    fprintf(stderr, "  -t, --text <text>      Text to synthesize (required)\n");
    fprintf(stderr, "  -o, --output <file>    Output WAV file (default: output.wav)\n");
    fprintf(stderr, "  -r, --reference <file> Reference audio for voice cloning\n");
    fprintf(stderr, "  --ref-text <text>      Reference text for voice cloning\n");
    fprintf(stderr, "  --mode <mode>          Optional mode override: base, clone, custom, design\n");
    fprintf(stderr, "  --instruct <text>      Instruct text for custom/design mode\n");
    fprintf(stderr, "  --speaker <name>       Speaker name for custom mode (Vivian,Ryan,Aiden,...)\n");
    fprintf(stderr, "  --temperature <val>    Sampling temperature (default: model_profile, 0=greedy)\n");
    fprintf(stderr, "  --top-k <n>            Top-k sampling (default: model_profile, 0=disabled)\n");
    fprintf(stderr, "  --top-p <val>          Top-p sampling (default: model_profile)\n");
    fprintf(stderr, "  --predictor-greedy     Use greedy decoding for predictor stage\n");
    fprintf(stderr, "  --predictor-temperature <val> Predictor stage temperature (default: model_profile)\n");
    fprintf(stderr, "  --predictor-top-k <n>  Predictor stage top-k (default: model_profile)\n");
    fprintf(stderr, "  --predictor-top-p <val> Predictor stage top-p (default: model_profile)\n");
    fprintf(stderr, "  --seed <n>            Talker sampler seed (default: model_profile)\n");
    fprintf(stderr, "  --predictor-seed <n>  Predictor sampler seed (default: model_profile)\n");
    fprintf(stderr, "  --max-tokens <n>       Maximum audio tokens (default: model_profile)\n");
    fprintf(stderr, "  --repetition-penalty <val> Repetition penalty (default: model_profile)\n");
    fprintf(stderr, "  --ort-debug-log        Enable ORT warning logs (default: error-only)\n");
    fprintf(stderr, "  --stats-json <file>    Write timing/runtime stats JSON report\n");
    fprintf(stderr, "  --warmup <n>           Warm-up runs after load (default: 0)\n");
    fprintf(stderr, "  --no-warmup            Skip the in-load decoder warmup\n");
    fprintf(stderr, "  --low-mem              Lazy-load decoder; unload after each synthesize\n");
    fprintf(stderr, "  --repeat <n>           Benchmark runs (default: 1)\n");
    fprintf(stderr, "  -l, --language <lang>  Explicit language from model profile aliases (or 'none')\n");
    fprintf(stderr, "  --no-auto-language     Deprecated no-op (default already Auto(None))\n");
    fprintf(stderr, "  -j, --threads <n>      Number of threads (default: 4)\n");
    fprintf(stderr, "  -v, --verbose          Show detailed debug logs\n");
    fprintf(stderr, "  -h, --help             Show this help\n");
    fprintf(stderr, "\n");
}

int main(int argc, char ** argv) {
    // Initialize logger first
    Logger::instance().init("logs/latest.log");

    uint64_t vram_start = 0;
    if (lunavox::NVMLMonitor::instance().init()) {
        vram_start = lunavox::NVMLMonitor::instance().get_used_vram();
    }

    std::vector<std::string> args = platform::utf8_args(argc, argv);

    const char * program = args.empty() ? "lunavox-cli" : args[0].c_str();
    std::string model_dir;
    std::string text;
    std::string output_file = "output.wav";
    std::string reference_audio;
    std::string stats_json_file;
    std::string mode;
    bool mode_explicit = false;
    std::string instruct_text;
    std::string speaker_name;
    std::string language_name = "none";
    bool language_explicit = false;
    int warmup = 0;
    int repeat = 1;
    bool verbose = false;
    bool disable_load_warmup = false;
    bool low_mem_mode_cli = false;

    lunavox::tts_params params;
    bool user_set_max_tokens = false;
    bool user_set_temperature = false;
    bool user_set_top_k = false;
    bool user_set_top_p = false;
    bool user_set_repetition_penalty = false;
    bool user_set_predictor_mode = false;
    bool user_set_predictor_temperature = false;
    bool user_set_predictor_top_k = false;
    bool user_set_predictor_top_p = false;
    bool user_set_seed = false;
    bool user_set_predictor_seed = false;
    
    // Parse arguments
    for (int i = 1; i < (int)args.size(); i++) {
        std::string arg = args[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(program);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing model directory");
                return 1;
            }
            model_dir = args[i];
        } else if (arg == "-t" || arg == "--text") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing text");
                return 1;
            }
            text = args[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing output file");
                return 1;
            }
            output_file = args[i];
        } else if (arg == "-r" || arg == "--reference") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing reference audio");
                return 1;
            }
            reference_audio = args[i];
        } else if (arg == "--ref-text") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing reference text");
                return 1;
            }
            params.ref_text = args[i];
        } else if (arg == "--temperature") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing temperature value");
                return 1;
            }
            params.temperature = std::stof(args[i]);
            user_set_temperature = true;
        } else if (arg == "--top-k") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing top-k value");
                return 1;
            }
            params.top_k = std::stoi(args[i]);
            user_set_top_k = true;
        } else if (arg == "--top-p") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing top-p value");
                return 1;
            }
            params.top_p = std::stof(args[i]);
            user_set_top_p = true;
        } else if (arg == "--predictor-greedy") {
            params.predictor_do_sample = false;
            user_set_predictor_mode = true;
        } else if (arg == "--predictor-temperature") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing predictor-temperature value");
                return 1;
            }
            params.predictor_temperature = std::stof(args[i]);
            params.predictor_do_sample = params.predictor_temperature > 0.0f;
            user_set_predictor_mode = true;
            user_set_predictor_temperature = true;
        } else if (arg == "--predictor-top-k") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing predictor-top-k value");
                return 1;
            }
            params.predictor_top_k = std::stoi(args[i]);
            user_set_predictor_top_k = true;
        } else if (arg == "--predictor-top-p") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing predictor-top-p value");
                return 1;
            }
            params.predictor_top_p = std::stof(args[i]);
            user_set_predictor_top_p = true;
        } else if (arg == "--seed") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing seed value");
                return 1;
            }
            params.seed = std::stoi(args[i]);
            user_set_seed = true;
        } else if (arg == "--predictor-seed") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing predictor-seed value");
                return 1;
            }
            params.predictor_seed = std::stoi(args[i]);
            user_set_predictor_seed = true;
        } else if (arg == "--max-tokens") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing max-tokens value");
                return 1;
            }
            params.max_audio_tokens = std::stoi(args[i]);
            user_set_max_tokens = true;
        } else if (arg == "--repetition-penalty") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing repetition-penalty value");
                return 1;
            }
            params.repetition_penalty = std::stof(args[i]);
            user_set_repetition_penalty = true;
        } else if (arg == "--ort-debug-log") {
            params.ort_debug_log = true;
        } else if (arg == "--stats-json") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing stats-json file path");
                return 1;
            }
            stats_json_file = args[i];
        } else if (arg == "--warmup") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing warmup count");
                return 1;
            }
            warmup = std::stoi(args[i]);
        } else if (arg == "--no-warmup") {
            disable_load_warmup = true;
        } else if (arg == "--low-mem") {
            low_mem_mode_cli = true;
        } else if (arg == "--repeat") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing repeat count");
                return 1;
            }
            repeat = std::stoi(args[i]);
        } else if (arg == "-l" || arg == "--language") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing language value");
                return 1;
            }
            language_name = to_lower_ascii(args[i]);
            language_explicit = true;
        } else if (arg == "--no-auto-language") {
            params.language_id = -1;
        } else if (arg == "-j" || arg == "--threads") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing threads value");
                return 1;
            }
            params.n_threads = std::stoi(args[i]);
        } else if (arg == "--mode") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing mode value");
                return 1;
            }
            mode = to_lower_ascii(args[i]);
            mode_explicit = true;
            if (mode != "base" && mode != "clone" && mode != "custom" && mode != "design") {
                LOG_ERROR("Error: unknown mode '%s'. Use: base, clone, custom, design", mode.c_str());
                return 1;
            }
        } else if (arg == "--instruct") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing instruct text");
                return 1;
            }
            instruct_text = args[i];
        } else if (arg == "--speaker") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing speaker name");
                return 1;
            }
            speaker_name = args[i];
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else {
            LOG_ERROR("Error: unknown argument: %s", arg.c_str());
            print_usage(program);
            return 1;
        }
    }

    if (verbose) {
        Logger::instance().set_level(LogLevel::DEBUG_LOG);
    }
    
    // Validate required arguments
    if (model_dir.empty()) {
        LOG_ERROR("Error: model directory is required");
        print_usage(program);
        return 1;
    }
    
    if (text.empty()) {
        LOG_ERROR("Error: text is required");
        print_usage(program);
        return 1;
    }

    LOG_USER("[ LunaVox TTS Engine ]");
    LOG_USER("--------------------------------------------------");
    
    // Initialize TTS
    lunavox::Engine tts;
    tts.set_warmup_enabled(!disable_load_warmup);

    lunavox::set_ort_debug_log(params.ort_debug_log);

    if (low_mem_mode_cli) {
        platform::set_env("LUNAVOX_LOW_MEM", "1");
        LOG_USER("[LOAD] Low-memory mode enabled via --low-mem");
    }

    LOG_USER("[LOAD] Loading Models...");
    const auto t_load_start = std::chrono::steady_clock::now();
    if (!tts.load_models(model_dir, params.n_threads)) {
        LOG_ERROR("\nError during model load: %s", tts.get_error().c_str());
        return 1;
    }
    const int64_t t_load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now() - t_load_start).count();
    const int64_t t_warmup_ms = tts.last_warmup_ms();
    LOG_USER("[LOAD] Model load completed in %lld ms (warmup %lld ms)",
             (long long) t_load_ms, (long long) t_warmup_ms);
    
    LOG_USER("[PARA] Text: \"%s\"", text.c_str());
    
    // Set progress callback
    tts.set_progress_callback([](int tokens, int max_tokens) {
        fprintf(stderr, "\rGenerating: %d/%d tokens   ", tokens, max_tokens);
    });
    
    const auto & profile = tts.profile();

    // Mode auto-routing when --mode is omitted:
    // base -> base (auto-upgrade to clone when reference audio is given)
    // custom -> custom
    // design -> design
    if (!mode_explicit) {
        if (profile.model_type == "base") {
            mode = "base";
        } else if (profile.model_type == "custom") {
            mode = "custom";
        } else if (profile.model_type == "design") {
            mode = "design";
        } else {
            LOG_ERROR("Error: unsupported model_type '%s' in model_profile.json", profile.model_type.c_str());
            return 1;
        }
        if (!reference_audio.empty() && mode == "base") {
            mode = "clone";
        }
    }

    // Use model profile defaults unless explicitly overridden by CLI args.
    if (!user_set_max_tokens) {
        params.max_audio_tokens = std::max(1, profile.default_max_new_tokens);
    }
    if (!user_set_temperature) {
        params.temperature = profile.default_temperature;
    }
    if (!user_set_top_k) {
        params.top_k = profile.default_top_k;
    }
    if (!user_set_top_p) {
        params.top_p = profile.default_top_p;
    }
    if (!user_set_repetition_penalty) {
        params.repetition_penalty = profile.default_repetition_penalty;
    }
    if (!user_set_predictor_mode) {
        params.predictor_do_sample = profile.default_predictor_do_sample;
    }
    if (!user_set_predictor_temperature) {
        params.predictor_temperature = profile.default_predictor_temperature;
    }
    if (!user_set_predictor_top_k) {
        params.predictor_top_k = profile.default_predictor_top_k;
    }
    if (!user_set_predictor_top_p) {
        params.predictor_top_p = profile.default_predictor_top_p;
    }
    if (!user_set_seed) {
        params.seed = profile.default_seed;
    }
    if (!user_set_predictor_seed) {
        params.predictor_seed = profile.default_predictor_seed;
    }

    if (language_explicit) {
        int32_t resolved_language_id = -1;
        if (!tts.resolve_language_id(language_name, resolved_language_id)) {
            const auto langs = tts.supported_languages();
            std::string supported;
            for (size_t i = 0; i < langs.size(); ++i) {
                if (i > 0) supported += ", ";
                supported += langs[i];
            }
            LOG_ERROR("Error: unknown language '%s'. Supported aliases from model_profile.json: %s, none",
                      language_name.c_str(),
                      supported.c_str());
            return 1;
        }
        params.language_id = resolved_language_id;
    } else {
        params.language_id = -1;
    }

    if (!tts.supports_mode(mode)) {
        LOG_ERROR("Error: mode '%s' is incompatible with model_type '%s' from model_profile.json",
                  mode.c_str(),
                  profile.model_type.c_str());
        return 1;
    }
    if (mode == "base" && !instruct_text.empty()) {
        LOG_ERROR("Error: --instruct is forbidden in base mode by runtime policy.");
        return 1;
    }
    if (!profile.instruct_support && !instruct_text.empty()) {
        LOG_ERROR("Error: this model does not support --instruct (model_profile.instruct_support=false).");
        return 1;
    }
    if (mode == "clone" && reference_audio.empty()) {
        LOG_ERROR("Error: clone mode requires --reference");
        return 1;
    }
    if (mode == "custom") {
        if (speaker_name.empty()) {
            LOG_ERROR("Error: custom mode requires --speaker");
            return 1;
        }
        if (profile.resolve_speaker_id(speaker_name) < 0) {
            const auto speakers = tts.supported_speakers();
            std::string supported;
            for (size_t i = 0; i < speakers.size(); ++i) {
                if (i > 0) supported += ", ";
                supported += speakers[i];
            }
            LOG_ERROR("Error: unknown speaker '%s'. Supported from model_profile.json: %s",
                      speaker_name.c_str(),
                      supported.c_str());
            return 1;
        }
    }
    if (mode == "design" && instruct_text.empty()) {
        LOG_ERROR("Error: design mode requires --instruct");
        return 1;
    }

    LOG_USER("[PARA] Mode: %s", mode.c_str());
    if (mode == "clone") LOG_USER("[PARA] Reference: %s", reference_audio.c_str());
    if (mode == "custom") LOG_USER("[PARA] Speaker: %s", speaker_name.c_str());
    if (language_explicit) LOG_USER("[PARA] Language: %s", language_name.c_str());
    else LOG_USER("[PARA] Language: Auto(None)");
    if (!instruct_text.empty()) LOG_USER("[PARA] Instruct: %s", instruct_text.c_str());

    // Warm-up
    if (warmup > 0) {
        LOG_USER("\n[BNCH] Starting %d warm-up runs...", warmup);
        for (int i = 0; i < warmup; ++i) {
            fprintf(stderr, "\rWarm-up: %d/%d   ", i + 1, warmup);
            if (mode == "clone") {
                tts.synthesize_with_voice(text, reference_audio, params);
            } else if (mode == "custom") {
                tts.synthesize_custom(text, speaker_name, instruct_text, params);
            } else if (mode == "design") {
                tts.synthesize_design(text, instruct_text, params);
            } else {
                tts.synthesize(text, params);
            }
        }
        fprintf(stderr, "\n");
    }

    // Benchmark Run
    std::vector<lunavox::tts_result> repeat_results;
    LOG_DEBUG("[BNCH] Starting %d benchmark runs...", repeat);
    
    for (int it = 0; it < repeat; ++it) {
        if (repeat > 1) {
            LOG_USER("[BNCH] Run %d/%d", it + 1, repeat);
        }
        
        // Generate speech
        lunavox::tts_result result;
        
        if (mode == "clone") {
            if (reference_audio.empty()) { // Added this check back for safety, though it should be caught earlier
                LOG_ERROR("Error: clone mode requires --reference");
                return 1;
            }
            result = tts.synthesize_with_voice(text, reference_audio, params);
        } else if (mode == "custom") {
            if (speaker_name.empty()) { // Added this check back for safety
                LOG_ERROR("Error: custom mode requires --speaker");
                return 1;
            }
            result = tts.synthesize_custom(text, speaker_name, instruct_text, params);
        } else if (mode == "design") {
            if (instruct_text.empty()) { // Added this check back for safety
                LOG_ERROR("Error: design mode requires --instruct");
                return 1;
            }
            result = tts.synthesize_design(text, instruct_text, params);
        } else {
            result = tts.synthesize(text, params);
        }
        
        if (!result.success) {
            LOG_ERROR("\nError: %s", result.error_msg.c_str());
            return 1;
        }

        repeat_results.push_back(result);
        
        // Save output
        std::string current_output = output_file;
        if (repeat > 1) {
            size_t dot_pos = output_file.find_last_of('.');
            if (dot_pos != std::string::npos) {
                current_output = output_file.substr(0, dot_pos) + "_" + std::to_string(it + 1) + output_file.substr(dot_pos);
            } else {
                current_output = output_file + "_" + std::to_string(it + 1);
            }
        }

        if (!lunavox::save_audio_file(current_output, result.audio, result.sample_rate)) {
            LOG_ERROR("Error: failed to save output file: %s", current_output.c_str());
            return 1;
        }
        
        if (repeat == 1) {
            LOG_USER("\rGenerating: [####################] 100%%          \n");
        } else {
            LOG_USER("  - Completed: %s", current_output.c_str());
        }
    }
    
    LOG_USER("");
    LOG_USER("[ Backend Configuration ]");
    {
        std::string binfo = lunavox::Logger::instance().get_llama_backend_info();
        if (binfo == "cpu") binfo = "CPU: Native Implementation";
        else if (binfo == "cuda" || binfo == "CUDA") binfo = "CUDA: NVIDIA Acceleration";
        else if (binfo == "vulkan" || binfo == "Vulkan") binfo = "Vulkan: Generic Acceleration";
        else if (binfo == "metal" || binfo == "Metal") binfo = "Metal: Apple Silicon Acceleration";
        else if (binfo == "rocm" || binfo == "ROCm") binfo = "ROCm: AMD Acceleration";
        else if (binfo == "sycl" || binfo == "SYCL") binfo = "SYCL: Intel Acceleration";
        
        LOG_USER("  - LLM Engine:     llama.cpp [%s]", binfo.c_str());
    }
    
    // Using the last result for provider info
    const auto & last_res = repeat_results.back();
    if (last_res.ort_provider_speaker_encoder == "not_loaded" && last_res.ort_provider_codec_encoder == "not_loaded") {
        LOG_USER("  - Audio Encoder:  ONNX Runtime [not loaded]");
    } else {
        LOG_USER("  - Audio Encoder:  ONNX Runtime [Codec: %s]", last_res.ort_provider_codec_encoder.c_str());
    }
    LOG_USER("  - Audio Decoder:  ONNX Runtime [%s]", last_res.ort_provider_decoder.c_str());

    LOG_USER("");
    LOG_USER("[ Resource Usage (Last Run) ]");
    LOG_USER("  - Peak RAM:       %.2f GB", (double)last_res.mem_rss_peak_bytes / (1024.0 * 1024.0 * 1024.0));
    if (lunavox::NVMLMonitor::instance().is_available()) {
        uint64_t vram_end = lunavox::NVMLMonitor::instance().get_used_vram();
        double delta_gb = (double)(vram_end > vram_start ? vram_end - vram_start : 0) / (1024.0 * 1024.0 * 1024.0);
        LOG_USER("  - GPU VRAM Delta: %.2f GB (%s)", delta_gb, lunavox::NVMLMonitor::instance().get_device_name().c_str());
    } else {
        LOG_USER("  - GPU VRAM Delta: N/A (NVML not initialized)");
    }
    LOG_USER("  - Peak Phys:      %.2f GB", (double)last_res.mem_phys_peak_bytes / (1024.0 * 1024.0 * 1024.0));
    
    LOG_USER("");
    LOG_USER("[ Inference Metrics ]");
    
    double avg_total_ms = 0;
    double avg_audio_sec = 0;
    double avg_rtf = 0;
    
    for (const auto & r : repeat_results) {
        double audio_sec = (double)r.audio.size() / r.sample_rate;
        double wall_sec = (double)r.t_total_ms / 1000.0;
        double rtf = audio_sec > 0 ? wall_sec / audio_sec : 0;
        
        avg_total_ms += r.t_total_ms;
        avg_audio_sec += audio_sec;
        avg_rtf += rtf;
    }
    
    avg_total_ms /= repeat;
    avg_audio_sec /= repeat;
    avg_rtf /= repeat;

    if (repeat == 1) {
        const auto & r = repeat_results[0];
        LOG_USER("  - Tokenization:          %6lld ms", (long long)r.t_tokenize_ms);
        LOG_USER("  - Speaker Encoding:      %6lld ms", (long long)r.t_encode_ms);
        LOG_USER("  - Code Generation:       %6lld ms", (long long)r.t_generate_ms);
        LOG_USER("  - Audio Decoding:        %6lld ms", (long long)r.t_decode_ms);
        LOG_USER("  -----------------------------------");
        LOG_USER("  - Total Latency:         %6lld ms", (long long)r.t_total_ms);
    } else {
        LOG_USER("  - Avg Total Latency:     %6.1f ms (N=%d)", avg_total_ms, repeat);
    }
    
    LOG_USER("  - Audio Duration:        %6.2f s", (float)avg_audio_sec);
    LOG_USER("  - Real-time Factor:      %6.2fx (%.3f RTF)", 1.0/avg_rtf, avg_rtf);
    
    LOG_USER("");
    LOG_USER("[ Result ]");
    if (repeat == 1) {
        LOG_USER("  - Saved to:       %s", output_file.c_str());
    } else {
        LOG_USER("  - Saved %d files to: %s_*", repeat, output_file.c_str());
    }
    LOG_USER("--------------------------------------------------");

    if (!stats_json_file.empty()) {
        FILE * jf = fopen(stats_json_file.c_str(), "wb");
        if (!jf) {
            LOG_WARN("Warning: failed to write stats JSON: %s", stats_json_file.c_str());
        } else {
            fprintf(jf, "{\n");
            fprintf(jf, "  \"t_load_ms\": %lld,\n", (long long) t_load_ms);
            fprintf(jf, "  \"t_warmup_ms\": %lld,\n", (long long) t_warmup_ms);
            fprintf(jf, "  \"runs\": [\n");
            for (size_t it = 0; it < repeat_results.size(); ++it) {
                const auto & r = repeat_results[it];
                double audio_sec = (double)r.audio.size() / r.sample_rate;
                double wall_sec = (double)r.t_total_ms / 1000.0;
                double rtf = audio_sec > 0 ? wall_sec / audio_sec : 0;
                
                const std::string spk_ep_json = json_escape(r.ort_provider_speaker_encoder);
                const std::string codec_ep_json = json_escape(r.ort_provider_codec_encoder);
                const std::string decoder_ep_json = json_escape(r.ort_provider_decoder);
                fprintf(
                    jf,
                    "  {\n"
                    "    \"run_id\": %d,\n"
                    "    \"success\": true,\n"
                    "    \"sample_rate\": %d,\n"
                    "    \"audio_samples\": %d,\n"
                    "    \"audio_sec\": %.6f,\n"
                    "    \"timing_ms\": {\n"
                    "      \"tokenize\": %lld,\n"
                    "      \"encode\": %lld,\n"
                    "      \"generate\": %lld,\n"
                    "      \"llama_prefill\": %lld,\n"
                    "      \"llama_decode_loop\": %lld,\n"
                    "      \"talker_post\": %lld,\n"
                    "      \"talker_decode\": %lld,\n"
                    "      \"talker_post_prep\": %lld,\n"
                    "      \"talker_post_copy\": %lld,\n"
                    "      \"predictor_sample\": %lld,\n"
                    "      \"decode\": %lld,\n"
                    "      \"ort_decoder_run\": %lld,\n"
                    "      \"decoder_tensor_prep\": %lld,\n"
                    "      \"decoder_ort_run\": %lld,\n"
                    "      \"decoder_tensor_extract\": %lld,\n"
                    "      \"decoder_state_trim\": %lld,\n"
                    "      \"pcm_gather\": %lld,\n"
                    "      \"first_audio\": %lld,\n"
                    "      \"total\": %lld\n"
                    "    },\n"
                    "    \"stream\": {\n"
                    "      \"first_chunk_frames\": %d,\n"
                    "      \"t_first_audio_ms\": %lld\n"
                    "    },\n"
                    "    \"rtf\": %.6f,\n"
                    "    \"mem\": {\n"
                    "      \"rss_start\": %llu,\n"
                    "      \"rss_end\": %llu,\n"
                    "      \"rss_peak\": %llu,\n"
                    "      \"phys_start\": %llu,\n"
                    "      \"phys_end\": %llu,\n"
                    "      \"phys_peak\": %llu\n"
                    "    },\n"
                    "    \"diag\": {\n"
                    "      \"pcm_peak\": %.6f,\n"
                    "      \"pcm_rms\": %.6f\n"
                    "    },\n"
                    "    \"ctx\": {\n"
                    "      \"ctx_required\": %d,\n"
                    "      \"ctx_allocated\": %d,\n"
                    "      \"ctx_cap\": %d,\n"
                    "      \"ctx_overflow\": %s\n"
                    "    },\n"
                    "    \"gen\": {\n"
                    "      \"code_frames\": %d,\n"
                    "      \"codebooks\": %d,\n"
                    "      \"code_min\": %d,\n"
                    "      \"code_max\": %d,\n"
                    "      \"eos_step\": %d,\n"
                    "      \"codes_hash\": %llu\n"
                    "    },\n"
                    "    \"ort_providers\": {\n"
                    "      \"speaker_encoder\": \"%s\",\n"
                    "      \"codec_encoder\": \"%s\",\n"
                    "      \"decoder\": \"%s\"\n"
                    "    }\n"
                    "  }%s\n",
                    (int)it + 1,
                    r.sample_rate,
                    (int) r.audio.size(),
                    audio_sec,
                    (long long) r.t_tokenize_ms,
                    (long long) r.t_encode_ms,
                    (long long) r.t_generate_ms,
                    (long long) r.t_llama_prefill_ms,
                    (long long) r.t_llama_decode_loop_ms,
                    (long long) r.t_talker_post_ms,
                    (long long) r.t_talker_decode_ms,
                    (long long) r.t_talker_post_prep_ms,
                    (long long) r.t_talker_post_copy_ms,
                    (long long) r.t_predictor_sample_ms,
                    (long long) r.t_decode_ms,
                    (long long) r.t_ort_decoder_run_ms,
                    (long long) r.t_decoder_tensor_prep_ms,
                    (long long) r.t_decoder_ort_run_ms,
                    (long long) r.t_decoder_tensor_extract_ms,
                    (long long) r.t_decoder_state_trim_ms,
                    (long long) r.t_pcm_gather_ms,
                    (long long) r.t_first_audio_ms,
                    (long long) r.t_total_ms,
                    r.first_chunk_frames_used,
                    (long long) r.t_first_audio_ms,
                    rtf,
                    (unsigned long long) r.mem_rss_start_bytes,
                    (unsigned long long) r.mem_rss_end_bytes,
                    (unsigned long long) r.mem_rss_peak_bytes,
                    (unsigned long long) r.mem_phys_start_bytes,
                    (unsigned long long) r.mem_phys_end_bytes,
                    (unsigned long long) r.mem_phys_peak_bytes,
                    r.pcm_peak,
                    r.pcm_rms,
                    r.ctx_required,
                    r.ctx_allocated,
                    r.ctx_cap,
                    r.ctx_overflow ? "true" : "false",
                    r.gen_code_frames,
                    r.gen_codebooks,
                    r.gen_code_min,
                    r.gen_code_max,
                    r.eos_step,
                    (unsigned long long) r.gen_codes_hash,
                    spk_ep_json.c_str(),
                    codec_ep_json.c_str(),
                    decoder_ep_json.c_str(),
                    (it == repeat_results.size() - 1) ? "" : ",");
            }
            fprintf(jf, "  ]\n");
            fprintf(jf, "}\n");
            fclose(jf);
        }
    }
    
    return 0;
}
