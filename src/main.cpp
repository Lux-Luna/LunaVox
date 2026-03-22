#include "qwen3_tts.h"

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
    fprintf(stderr, "  -l, --language <lang>  Force language: en,ru,zh,ja,ko,de,fr,es,it,pt\n");
    fprintf(stderr, "  --no-auto-language     Disable language auto-detection (uses --language or en)\n");
    fprintf(stderr, "  -j, --threads <n>      Number of threads (default: 4)\n");
    fprintf(stderr, "  -h, --help             Show this help\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s -m ./models/base_small -t \"Hello, world!\" -o hello.wav\n", program);
    fprintf(stderr, "  %s -m ./models/base_small -t \"Hello!\" -r reference.wav -o cloned.wav\n", program);
    fprintf(stderr, "  %s -m ./models/custom --mode custom --speaker Vivian --instruct \"Speak gently\" -t \"Hello\" -o custom.wav\n", program);
    fprintf(stderr, "  %s -m ./models/design --mode design --instruct \"A warm female voice\" -t \"Hello\" -o design.wav\n", program);
}

int main(int argc, char ** argv) {
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
                fprintf(stderr, "Error: missing model directory\n");
                return 1;
            }
            model_dir = args[i];
        } else if (arg == "-t" || arg == "--text") {
            if (++i >= (int)args.size()) {
                fprintf(stderr, "Error: missing text\n");
                return 1;
            }
            text = args[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= (int)args.size()) {
                fprintf(stderr, "Error: missing output file\n");
                return 1;
            }
            output_file = args[i];
        } else if (arg == "-r" || arg == "--reference") {
            if (++i >= (int)args.size()) {
                fprintf(stderr, "Error: missing reference audio\n");
                return 1;
            }
            reference_audio = args[i];
        } else if (arg == "--temperature") {
            if (++i >= (int)args.size()) {
                fprintf(stderr, "Error: missing temperature value\n");
                return 1;
            }
            params.temperature = std::stof(args[i]);
        } else if (arg == "--top-k") {
            if (++i >= (int)args.size()) {
                fprintf(stderr, "Error: missing top-k value\n");
                return 1;
            }
            params.top_k = std::stoi(args[i]);
        } else if (arg == "--top-p") {
            if (++i >= (int)args.size()) {
                fprintf(stderr, "Error: missing top-p value\n");
                return 1;
            }
            params.top_p = std::stof(args[i]);
        } else if (arg == "--predictor-greedy") {
            params.predictor_do_sample = false;
        } else if (arg == "--predictor-temperature") {
            if (++i >= (int)args.size()) {
                fprintf(stderr, "Error: missing predictor-temperature value\n");
                return 1;
            }
            params.predictor_temperature = std::stof(args[i]);
            params.predictor_do_sample = params.predictor_temperature > 0.0f;
        } else if (arg == "--predictor-top-k") {
            if (++i >= (int)args.size()) {
                fprintf(stderr, "Error: missing predictor-top-k value\n");
                return 1;
            }
            params.predictor_top_k = std::stoi(args[i]);
        } else if (arg == "--predictor-top-p") {
            if (++i >= (int)args.size()) {
                fprintf(stderr, "Error: missing predictor-top-p value\n");
                return 1;
            }
            params.predictor_top_p = std::stof(args[i]);
        } else if (arg == "--seed") {
            if (++i >= (int)args.size()) {
                fprintf(stderr, "Error: missing seed value\n");
                return 1;
            }
            params.seed = std::stoi(args[i]);
        } else if (arg == "--predictor-seed") {
            if (++i >= (int)args.size()) {
                fprintf(stderr, "Error: missing predictor-seed value\n");
                return 1;
            }
            params.predictor_seed = std::stoi(args[i]);
        } else if (arg == "--max-tokens") {
            if (++i >= (int)args.size()) {
                fprintf(stderr, "Error: missing max-tokens value\n");
                return 1;
            }
            params.max_audio_tokens = std::stoi(args[i]);
        } else if (arg == "--repetition-penalty") {
            if (++i >= (int)args.size()) {
                fprintf(stderr, "Error: missing repetition-penalty value\n");
                return 1;
            }
            params.repetition_penalty = std::stof(args[i]);
        } else if (arg == "--ort-debug-log") {
            params.ort_debug_log = true;
        } else if (arg == "--stats-json") {
            if (++i >= (int)args.size()) {
                fprintf(stderr, "Error: missing stats-json file path\n");
                return 1;
            }
            stats_json_file = args[i];
        } else if (arg == "-l" || arg == "--language") {
            if (++i >= (int)args.size()) {
                fprintf(stderr, "Error: missing language value\n");
                return 1;
            }
            std::string lang = args[i];
            int32_t parsed_language_id = 2050;
            if (!parse_language_id(lang, parsed_language_id)) {
                fprintf(stderr, "Error: unknown language '%s'. Supported: en,ru,zh,ja,ko,de,fr,es,it,pt\n", lang.c_str());
                return 1;
            }
            params.language_id = parsed_language_id;
            params.auto_language = false;
        } else if (arg == "--no-auto-language") {
            params.auto_language = false;
        } else if (arg == "-j" || arg == "--threads") {
            if (++i >= (int)args.size()) {
                fprintf(stderr, "Error: missing threads value\n");
                return 1;
            }
            params.n_threads = std::stoi(args[i]);
        } else if (arg == "--mode") {
            if (++i >= (int)args.size()) {
                fprintf(stderr, "Error: missing mode value\n");
                return 1;
            }
            mode = to_lower_ascii(args[i]);
            if (mode != "base" && mode != "clone" && mode != "custom" && mode != "design") {
                fprintf(stderr, "Error: unknown mode '%s'. Use: base, clone, custom, design\n", mode.c_str());
                return 1;
            }
        } else if (arg == "--instruct") {
            if (++i >= (int)args.size()) {
                fprintf(stderr, "Error: missing instruct text\n");
                return 1;
            }
            instruct_text = args[i];
        } else if (arg == "--speaker") {
            if (++i >= (int)args.size()) {
                fprintf(stderr, "Error: missing speaker name\n");
                return 1;
            }
            speaker_name = args[i];
        } else {
            fprintf(stderr, "Error: unknown argument: %s\n", arg.c_str());
            print_usage(program);
            return 1;
        }
    }
    
    // Validate required arguments
    if (model_dir.empty()) {
        fprintf(stderr, "Error: model directory is required\n");
        print_usage(program);
        return 1;
    }
    
    if (text.empty()) {
        fprintf(stderr, "Error: text is required\n");
        print_usage(program);
        return 1;
    }
    
    // Initialize TTS
    qwen3_tts::Qwen3TTS tts;

    qwen3_tts::set_ort_debug_log(params.ort_debug_log);
    
    fprintf(stderr, "Loading models from: %s\n", model_dir.c_str());
    if (!tts.load_models(model_dir, params.n_threads)) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return 1;
    }
    
    // Set progress callback
    tts.set_progress_callback([](int tokens, int max_tokens) {
        fprintf(stderr, "\rGenerating: %d/%d tokens", tokens, max_tokens);
    });
    
    // Infer mode from --reference if provided and mode is still "base"
    if (!reference_audio.empty() && mode == "base") {
        mode = "clone";
    }

    // Generate speech
    qwen3_tts::tts_result result;
    
    if (mode == "clone") {
        if (reference_audio.empty()) {
            fprintf(stderr, "Error: clone mode requires --reference\n");
            return 1;
        }
        fprintf(stderr, "Synthesizing with voice cloning: \"%s\"\n", text.c_str());
        fprintf(stderr, "Reference audio: %s\n", reference_audio.c_str());
        result = tts.synthesize_with_voice(text, reference_audio, params);
    } else if (mode == "custom") {
        if (speaker_name.empty()) {
            fprintf(stderr, "Error: custom mode requires --speaker\n");
            return 1;
        }
        fprintf(stderr, "Synthesizing with custom voice: \"%s\" (speaker: %s)\n", text.c_str(), speaker_name.c_str());
        if (!instruct_text.empty()) {
            fprintf(stderr, "Instruct: %s\n", instruct_text.c_str());
        }
        result = tts.synthesize_custom(text, speaker_name, instruct_text, params);
    } else if (mode == "design") {
        if (instruct_text.empty()) {
            fprintf(stderr, "Error: design mode requires --instruct\n");
            return 1;
        }
        fprintf(stderr, "Synthesizing with voice design: \"%s\"\n", text.c_str());
        fprintf(stderr, "Design instruct: %s\n", instruct_text.c_str());
        result = tts.synthesize_design(text, instruct_text, params);
    } else {
        // base mode
        fprintf(stderr, "Synthesizing: \"%s\"\n", text.c_str());
        result = tts.synthesize(text, params);
    }
    
    if (!result.success) {
        fprintf(stderr, "\nError: %s\n", result.error_msg.c_str());
        return 1;
    }
    
    fprintf(stderr, "\n");
    
    // Save output
    if (!qwen3_tts::save_audio_file(output_file, result.audio, result.sample_rate)) {
        fprintf(stderr, "Error: failed to save output file: %s\n", output_file.c_str());
        return 1;
    }
    
    fprintf(stderr, "Output saved to: %s\n", output_file.c_str());
    fprintf(stderr, "Audio duration: %.2f seconds\n", 
            (float)result.audio.size() / result.sample_rate);

    if (!stats_json_file.empty()) {
        FILE * jf = fopen(stats_json_file.c_str(), "wb");
        if (!jf) {
            fprintf(stderr, "Warning: failed to write stats JSON: %s\n", stats_json_file.c_str());
        } else {
            const double audio_sec =
                result.sample_rate > 0 ? (double) result.audio.size() / (double) result.sample_rate : 0.0;
            const double wall_sec = (double) result.t_total_ms / 1000.0;
            const double rtf = audio_sec > 0.0 ? wall_sec / audio_sec : 0.0;
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
                "  \"diagnostics\": {\n"
                "    \"spk_emb_dim\": %d,\n"
                "    \"spk_emb_l2\": %.9f,\n"
                "    \"spk_emb_nan_count\": %d,\n"
                "    \"spk_emb_inf_count\": %d,\n"
                "    \"ref_code_frames\": %d,\n"
                "    \"ref_codebooks\": %d,\n"
                "    \"ref_code_min\": %d,\n"
                "    \"ref_code_max\": %d,\n"
                "    \"gen_code_frames\": %d,\n"
                "    \"gen_codebooks\": %d,\n"
                "    \"gen_code_min\": %d,\n"
                "    \"gen_code_max\": %d,\n"
                "    \"gen_codes_hash_hex\": \"%016llx\",\n"
                "    \"eos_step\": %d,\n"
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
                (double) result.spk_emb_l2,
                (int) result.spk_emb_nan_count,
                (int) result.spk_emb_inf_count,
                (int) result.ref_code_frames,
                (int) result.ref_codebooks,
                (int) result.ref_code_min,
                (int) result.ref_code_max,
                (int) result.gen_code_frames,
                (int) result.gen_codebooks,
                (int) result.gen_code_min,
                (int) result.gen_code_max,
                (unsigned long long) result.gen_codes_hash,
                (int) result.eos_step,
                (int) result.trailing_count,
                (int) result.trailing_consumed,
                (double) result.pcm_peak,
                (double) result.pcm_rms,
                spk_ep_json.c_str(),
                codec_ep_json.c_str(),
                decoder_ep_json.c_str());
            fclose(jf);
        }
    }
    
    // Print timing
    if (params.print_timing) {
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Load:      %6lld ms\n", (long long)result.t_load_ms);
        fprintf(stderr, "  Tokenize:  %6lld ms\n", (long long)result.t_tokenize_ms);
        fprintf(stderr, "  Encode:    %6lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Generate:  %6lld ms\n", (long long)result.t_generate_ms);
        fprintf(stderr, "  Decode:    %6lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:     %6lld ms\n", (long long)result.t_total_ms);
    }
    
    return 0;
}
