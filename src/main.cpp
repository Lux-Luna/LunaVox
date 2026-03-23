#include "qwen3_tts.h"
#include "logger.h"
#include "nvml_monitor.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

using namespace qwen3_tts;

static std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return s;
}

static bool parse_language_id(const std::string & language, int32_t & language_id_out) {
    const std::string lang = to_lower_ascii(language);
    if (lang == "en" || lang == "english")       language_id_out = 2050;
    else if (lang == "ru" || lang == "russian")  language_id_out = 2069;
    else if (lang == "zh" || lang == "chinese")  language_id_out = 2055;
    else if (lang == "ja" || lang == "japanese") language_id_out = 2058;
    else if (lang == "ko" || lang == "korean")   language_id_out = 2064;
    else if (lang == "de" || lang == "german")   language_id_out = 2053;
    else if (lang == "fr" || lang == "french")   language_id_out = 2061;
    else if (lang == "es" || lang == "spanish")  language_id_out = 2054;
    else if (lang == "it" || lang == "italian")  language_id_out = 2070;
    else if (lang == "pt" || lang == "portuguese") language_id_out = 2071;
    else if (lang == "none" || lang == "unspecified") language_id_out = -1;
    else return false;
    return true;
}

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

#ifdef _WIN32
static std::string wide_to_utf8(const std::wstring & ws) {
    if (ws.empty()) {
        return std::string();
    }
    int size = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(),
                                   nullptr, 0, nullptr, nullptr);
    if (size <= 0) {
        return std::string();
    }
    std::string utf8((size_t)size, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(),
                        &utf8[0], size, nullptr, nullptr);
    return utf8;
}

static std::vector<std::string> collect_cli_args_utf8(int argc, char ** argv) {
    std::vector<std::string> args;
    int wide_argc = 0;
    LPWSTR * wide_argv = CommandLineToArgvW(GetCommandLineW(), &wide_argc);
    if (wide_argv && wide_argc > 0) {
        args.reserve((size_t)wide_argc);
        for (int i = 0; i < wide_argc; ++i) {
            args.push_back(wide_to_utf8(wide_argv[i]));
        }
        LocalFree(wide_argv);
        return args;
    }

    args.reserve((size_t)argc);
    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i] ? argv[i] : "");
    }
    return args;
}
#endif

void print_usage(const char * program) {
    fprintf(stderr, "Usage: %s [options] -m <model_dir> -t <text>\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <dir>      Model directory (required)\n");
    fprintf(stderr, "  -t, --text <text>      Text to synthesize (required)\n");
    fprintf(stderr, "  -o, --output <file>    Output WAV file (default: output.wav)\n");
    fprintf(stderr, "  -r, --reference <file> Reference audio for voice cloning\n");
    fprintf(stderr, "  --ref-text <text>      Reference text for voice cloning\n");
    fprintf(stderr, "  --mode <mode>          Synthesis mode: base(default), clone, custom, design\n");
    fprintf(stderr, "  --instruct <text>      Instruct text for custom/design mode\n");
    fprintf(stderr, "  --speaker <name>       Speaker name for custom mode (Vivian,Ryan,Aiden,...)\n");
    fprintf(stderr, "  --temperature <val>    Sampling temperature (default: 0.9, 0=greedy)\n");
    fprintf(stderr, "  --top-k <n>            Top-k sampling (default: 50, 0=disabled)\n");
    fprintf(stderr, "  --top-p <val>          Top-p sampling (default: 1.0)\n");
    fprintf(stderr, "  --predictor-greedy     Use greedy decoding for predictor stage\n");
    fprintf(stderr, "  --predictor-temperature <val> Predictor stage temperature (default: 0.9)\n");
    fprintf(stderr, "  --predictor-top-k <n>  Predictor stage top-k (default: 50)\n");
    fprintf(stderr, "  --predictor-top-p <val> Predictor stage top-p (default: 1.0)\n");
    fprintf(stderr, "  --seed <n>            Talker sampler seed (default: random)\n");
    fprintf(stderr, "  --predictor-seed <n>  Predictor sampler seed (default: random)\n");
    fprintf(stderr, "  --max-tokens <n>       Maximum audio tokens (default: 4096)\n");
    fprintf(stderr, "  --repetition-penalty <val> Repetition penalty (default: 1.05)\n");
    fprintf(stderr, "  --ort-debug-log        Enable ORT warning logs (default: error-only)\n");
    fprintf(stderr, "  --stats-json <file>    Write timing/runtime stats JSON report\n");
    fprintf(stderr, "  -l, --language <lang>  Force language: en,ru,zh,ja,ko,de,fr,es,it,pt,none\n");
    fprintf(stderr, "  --no-auto-language     Disable language auto-detection (uses --language or en)\n");
    fprintf(stderr, "  -j, --threads <n>      Number of threads (default: 4)\n");
    fprintf(stderr, "  -v, --verbose          Show detailed debug logs\n");
    fprintf(stderr, "  -h, --help             Show this help\n");
    fprintf(stderr, "\n");
}

int main(int argc, char ** argv) {
    // Initialize logger first
    Logger::instance().init("logs/latest.log");

    uint64_t vram_start = 0;
    if (qwen3_tts::NVMLMonitor::instance().init()) {
        vram_start = qwen3_tts::NVMLMonitor::instance().get_used_vram();
    }

#ifdef _WIN32
    std::vector<std::string> args = collect_cli_args_utf8(argc, argv);
#else
    std::vector<std::string> args;
    args.reserve((size_t)argc);
    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i] ? argv[i] : "");
    }
#endif

    const char * program = args.empty() ? "qwen3-tts-cli" : args[0].c_str();
    std::string model_dir;
    std::string text;
    std::string output_file = "output.wav";
    std::string reference_audio;
    std::string stats_json_file;
    std::string mode = "base";
    std::string instruct_text;
    std::string speaker_name;
    bool verbose = false;
    
    qwen3_tts::tts_params params;
    params.auto_language = true;
    
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
        } else if (arg == "--top-k") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing top-k value");
                return 1;
            }
            params.top_k = std::stoi(args[i]);
        } else if (arg == "--top-p") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing top-p value");
                return 1;
            }
            params.top_p = std::stof(args[i]);
        } else if (arg == "--predictor-greedy") {
            params.predictor_do_sample = false;
        } else if (arg == "--predictor-temperature") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing predictor-temperature value");
                return 1;
            }
            params.predictor_temperature = std::stof(args[i]);
            params.predictor_do_sample = params.predictor_temperature > 0.0f;
        } else if (arg == "--predictor-top-k") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing predictor-top-k value");
                return 1;
            }
            params.predictor_top_k = std::stoi(args[i]);
        } else if (arg == "--predictor-top-p") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing predictor-top-p value");
                return 1;
            }
            params.predictor_top_p = std::stof(args[i]);
        } else if (arg == "--seed") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing seed value");
                return 1;
            }
            params.seed = std::stoi(args[i]);
        } else if (arg == "--predictor-seed") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing predictor-seed value");
                return 1;
            }
            params.predictor_seed = std::stoi(args[i]);
        } else if (arg == "--max-tokens") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing max-tokens value");
                return 1;
            }
            params.max_audio_tokens = std::stoi(args[i]);
        } else if (arg == "--repetition-penalty") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing repetition-penalty value");
                return 1;
            }
            params.repetition_penalty = std::stof(args[i]);
        } else if (arg == "--ort-debug-log") {
            params.ort_debug_log = true;
        } else if (arg == "--stats-json") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing stats-json file path");
                return 1;
            }
            stats_json_file = args[i];
        } else if (arg == "-l" || arg == "--language") {
            if (++i >= (int)args.size()) {
                LOG_ERROR("Error: missing language value");
                return 1;
            }
            std::string lang = args[i];
            int32_t parsed_language_id = 2050;
            if (!parse_language_id(lang, parsed_language_id)) {
                LOG_ERROR("Error: unknown language '%s'. Supported: en,ru,zh,ja,ko,de,fr,es,it,pt,none", lang.c_str());
                return 1;
            }
            params.language_id = parsed_language_id;
            params.auto_language = false;
        } else if (arg == "--no-auto-language") {
            params.auto_language = false;
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
    qwen3_tts::Qwen3TTS tts;

    qwen3_tts::set_ort_debug_log(params.ort_debug_log);
    
    LOG_USER("[LOAD] Loading Models...");
    if (!tts.load_models(model_dir, params.n_threads)) {
        LOG_ERROR("\nError during model load: %s", tts.get_error().c_str());
        return 1;
    }
    
    LOG_USER("[PARA] Text: \"%s\"", text.c_str());
    
    // Set progress callback
    tts.set_progress_callback([](int tokens, int max_tokens) {
        fprintf(stderr, "\rGenerating: %d/%d tokens   ", tokens, max_tokens);
    });
    
    // Infer mode from --reference if provided and mode is still "base"
    if (!reference_audio.empty() && mode == "base") {
        mode = "clone";
    }

    LOG_USER("[PARA] Mode: %s", mode.c_str());
    if (mode == "clone") LOG_USER("[PARA] Reference: %s", reference_audio.c_str());
    if (mode == "custom") LOG_USER("[PARA] Speaker: %s", speaker_name.c_str());
    if (!instruct_text.empty()) LOG_USER("[PARA] Instruct: %s", instruct_text.c_str());

    // Generate speech
    qwen3_tts::tts_result result;
    
    if (mode == "clone") {
        if (reference_audio.empty()) {
            LOG_ERROR("Error: clone mode requires --reference");
            return 1;
        }
        result = tts.synthesize_with_voice(text, reference_audio, params);
    } else if (mode == "custom") {
        if (speaker_name.empty()) {
            LOG_ERROR("Error: custom mode requires --speaker");
            return 1;
        }
        result = tts.synthesize_custom(text, speaker_name, instruct_text, params);
    } else if (mode == "design") {
        if (instruct_text.empty()) {
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
    
    LOG_USER("\rGenerating: [####################] 100%%          ");
    LOG_USER("");
    
    // Save output
    if (!qwen3_tts::save_audio_file(output_file, result.audio, result.sample_rate)) {
        LOG_ERROR("Error: failed to save output file: %s", output_file.c_str());
        return 1;
    }
    
    LOG_USER("");
    LOG_USER("[ Backend Configuration ]");
    LOG_USER("  - LLM Engine:     llama.cpp (%s)", qwen3_tts::Logger::instance().get_llama_backend_info().c_str());
    
    if (result.ort_provider_speaker_encoder == "not_loaded" && result.ort_provider_codec_encoder == "not_loaded") {
        LOG_USER("  - Audio Encoder:  ONNX Runtime (not loaded)");
    } else {
        LOG_USER("  - Audio Encoder:  ONNX Runtime (Codec: %s)", result.ort_provider_codec_encoder.c_str());
    }
    LOG_USER("  - Audio Decoder:  ONNX Runtime (%s)", result.ort_provider_decoder.c_str());
    
    LOG_USER("");
    LOG_USER("[ Resource Usage ]");
    LOG_USER("  - Peak RAM:       %.2f GB", (double)result.mem_rss_peak_bytes / (1024.0 * 1024.0 * 1024.0));
    
    if (qwen3_tts::NVMLMonitor::instance().is_available()) {
        uint64_t vram_end = qwen3_tts::NVMLMonitor::instance().get_used_vram();
        double delta_gb = (double)(vram_end > vram_start ? vram_end - vram_start : 0) / (1024.0 * 1024.0 * 1024.0);
        LOG_USER("  - GPU VRAM Delta: %.2f GB (%s)", delta_gb, qwen3_tts::NVMLMonitor::instance().get_device_name().c_str());
    } else {
        LOG_USER("  - GPU VRAM Delta: N/A (NVML not initialized)");
    }
    
    LOG_USER("  - Peak Phys:      %.2f GB", (double)result.mem_phys_peak_bytes / (1024.0 * 1024.0 * 1024.0));
    
    LOG_USER("");
    LOG_USER("[ Inference Metrics ]");
    const double audio_sec = (double)result.audio.size() / result.sample_rate;
    const double wall_sec = (double)result.t_total_ms / 1000.0;
    const double rtf = audio_sec > 0 ? wall_sec / audio_sec : 0;
    
    LOG_USER("  - Tokenization:          %6lld ms", (long long)result.t_tokenize_ms);
    LOG_USER("  - Speaker Encoding:      %6lld ms", (long long)result.t_encode_ms);
    LOG_USER("  - Code Generation:       %6lld ms", (long long)result.t_generate_ms);
    LOG_USER("  - Audio Decoding:        %6lld ms", (long long)result.t_decode_ms);
    LOG_USER("  -----------------------------------");
    LOG_USER("  - Total Latency:         %6lld ms", (long long)result.t_total_ms);
    LOG_USER("  - Audio Duration:        %6.2f s", (float)audio_sec);
    LOG_USER("  - Real-time Factor:      %6.2fx (%.3f RTF)", 1.0/rtf, rtf);
    
    LOG_USER("");
    LOG_USER("[ Result ]");
    LOG_USER("  - Saved to:       %s", output_file.c_str());
    LOG_USER("--------------------------------------------------");

    if (!stats_json_file.empty()) {
        FILE * jf = fopen(stats_json_file.c_str(), "wb");
        if (!jf) {
            LOG_WARN("Warning: failed to write stats JSON: %s", stats_json_file.c_str());
        } else {
            const std::string spk_ep_json = json_escape(result.ort_provider_speaker_encoder);
            const std::string codec_ep_json = json_escape(result.ort_provider_codec_encoder);
            const std::string decoder_ep_json = json_escape(result.ort_provider_decoder);
            fprintf(
                jf,
                "{\n"
                "  \"success\": true,\n"
                "  \"sample_rate\": %d,\n"
                "  \"audio_samples\": %d,\n"
                "  \"audio_sec\": %.6f,\n"
                "  \"timing_ms\": {\n"
                "    \"tokenize\": %lld,\n"
                "    \"encode\": %lld,\n"
                "    \"generate\": %lld,\n"
                "    \"decode\": %lld,\n"
                "    \"total\": %lld\n"
                "  },\n"
                "  \"rtf\": %.6f,\n"
                "  \"mem\": {\n"
                "    \"rss_start\": %llu,\n"
                "    \"rss_end\": %llu,\n"
                "    \"rss_peak\": %llu,\n"
                "    \"phys_start\": %llu,\n"
                "    \"phys_end\": %llu,\n"
                "    \"phys_peak\": %llu\n"
                "  },\n"
                "  \"diag\": {\n"
                "    \"spk_emb_dim\": %d,\n"
                "    \"eos_step\": %d,\n"
                "    \"effective_language_id\": %d,\n"
                "    \"used_auto_language\": %s,\n"
                "    \"gen_code_frames\": %d,\n"
                "    \"gen_code_min\": %d,\n"
                "    \"gen_code_max\": %d,\n"
                "    \"gen_codes_hash\": %llu,\n"
                "    \"trailing_count\": %d,\n"
                "    \"trailing_consumed\": %d,\n"
                "    \"pcm_peak\": %.9f,\n"
                "    \"pcm_rms\": %.9f\n"
                "  },\n"
                "  \"ort_providers\": {\n"
                "    \"speaker_encoder\": \"%s\",\n"
                "    \"codec_encoder\": \"%s\",\n"
                "    \"decoder\": \"%s\"\n"
                "  }\n"
                "}\n",
                result.sample_rate,
                (int) result.audio.size(),
                audio_sec,
                (long long) result.t_tokenize_ms,
                (long long) result.t_encode_ms,
                (long long) result.t_generate_ms,
                (long long) result.t_decode_ms,
                (long long) result.t_total_ms,
                rtf,
                (unsigned long long) result.mem_rss_start_bytes,
                (unsigned long long) result.mem_rss_end_bytes,
                (unsigned long long) result.mem_rss_peak_bytes,
                (unsigned long long) result.mem_phys_start_bytes,
                (unsigned long long) result.mem_phys_end_bytes,
                (unsigned long long) result.mem_phys_peak_bytes,
                (int) result.spk_emb_dim,
                (int) result.eos_step,
                (int) result.effective_language_id,
                result.used_auto_language ? "true" : "false",
                (int) result.gen_code_frames,
                (int) result.gen_code_min,
                (int) result.gen_code_max,
                (unsigned long long) result.gen_codes_hash,
                (int) result.trailing_count,
                (int) result.trailing_consumed,
                result.pcm_peak,
                result.pcm_rms,
                spk_ep_json.c_str(),
                codec_ep_json.c_str(),
                decoder_ep_json.c_str());
            fclose(jf);
        }
    }
    
    return 0;
}
