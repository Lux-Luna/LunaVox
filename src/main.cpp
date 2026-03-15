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
                        utf8.data(), size, nullptr, nullptr);
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
    fprintf(stderr, "  --temperature <val>    Sampling temperature (default: 0.9, 0=greedy)\n");
    fprintf(stderr, "  --top-k <n>            Top-k sampling (default: 50, 0=disabled)\n");
    fprintf(stderr, "  --top-p <val>          Top-p sampling (default: 1.0)\n");
    fprintf(stderr, "  --max-tokens <n>       Maximum audio tokens (default: 4096)\n");
    fprintf(stderr, "  --repetition-penalty <val> Repetition penalty (default: 1.05)\n");
    fprintf(stderr, "  -l, --language <lang>  Force language: en,ru,zh,ja,ko,de,fr,es,it,pt\n");
    fprintf(stderr, "  --no-auto-language     Disable language auto-detection (uses --language or en)\n");
    fprintf(stderr, "  -j, --threads <n>      Number of threads (default: 4)\n");
    fprintf(stderr, "  -h, --help             Show this help\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello, world!\" -o hello.wav\n", program);
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" -r reference.wav -o cloned.wav\n", program);
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
    
    fprintf(stderr, "Loading models from: %s\n", model_dir.c_str());
    if (!tts.load_models(model_dir)) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return 1;
    }
    
    // Set progress callback
    tts.set_progress_callback([](int tokens, int max_tokens) {
        fprintf(stderr, "\rGenerating: %d/%d tokens", tokens, max_tokens);
    });
    
    // Generate speech
    qwen3_tts::tts_result result;
    
    if (reference_audio.empty()) {
        fprintf(stderr, "Synthesizing: \"%s\"\n", text.c_str());
        result = tts.synthesize(text, params);
    } else {
        fprintf(stderr, "Synthesizing with voice cloning: \"%s\"\n", text.c_str());
        fprintf(stderr, "Reference audio: %s\n", reference_audio.c_str());
        result = tts.synthesize_with_voice(text, reference_audio, params);
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
