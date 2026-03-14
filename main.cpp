#include "qwen3_tts.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <iostream> // Added for std::cin and std::getline

void print_usage(const char * program) {
    fprintf(stderr, "Usage: %s [options] -m <model_dir>\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <dir>      Model directory (required)\n");
    fprintf(stderr, "  -t, --text <text>      Text to synthesize (required unless interactive or saving speaker)\n");
    fprintf(stderr, "  -i, --interactive      Run in interactive loop mode (load once, generate many)\n");
    fprintf(stderr, "  -o, --output <file>    Output WAV file (default: output.wav)\n");
    fprintf(stderr, "  -r, --reference <file> Reference audio for voice cloning\n");
    fprintf(stderr, "  -s, --speaker <file>   Load precomputed speaker embedding (.spk)\n");
    fprintf(stderr, "  --save-speaker <file>  Extract embedding from -r and save to file\n");
    fprintf(stderr, "  --temperature <val>    Sampling temperature (default: 0.9, 0=greedy)\n");
    fprintf(stderr, "  --top-k <n>            Top-k sampling (default: 50, 0=disabled)\n");
    fprintf(stderr, "  --top-p <val>          Top-p sampling (default: 1.0)\n");
    fprintf(stderr, "  --max-tokens <n>       Maximum audio tokens (default: 4096)\n");
    fprintf(stderr, "  --repetition-penalty <val> Repetition penalty (default: 1.05)\n");
    fprintf(stderr, "  -l, --language <lang>  Language: en,ru,zh,ja,ko,de,fr,es (default: en)\n");
    fprintf(stderr, "  -j, --threads <n>      Number of threads (default: 4)\n");
    fprintf(stderr, "  -h, --help             Show this help\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello, world!\" -o hello.wav\n", program);
    fprintf(stderr, "  %s -m ./models -i -r reference.wav -o output.wav\n", program);
    fprintf(stderr, "  %s -m ./models -r ref.wav --save-speaker voice.spk\n", program);
    fprintf(stderr, "  %s -m ./models -s voice.spk -t \"Hello, world!\" -o output.wav\n", program);
}

int main(int argc, char ** argv) {
    std::string model_dir;
    std::string text;
    std::string output_file = "output.wav";
    std::string reference_audio;
    std::string speaker_file;
    std::string save_speaker_file;
    bool interactive = false;
    
    qwen3_tts::tts_params params;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing model directory\n");
                return 1;
            }
            model_dir = argv[i];
        } else if (arg == "-t" || arg == "--text") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing text\n");
                return 1;
            }
            text = argv[i];
        } else if (arg == "-i" || arg == "--interactive" || arg == "--server") {
            interactive = true;
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing output file\n");
                return 1;
            }
            output_file = argv[i];
        } else if (arg == "-r" || arg == "--reference") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing reference audio\n");
                return 1;
            }
            reference_audio = argv[i];
        } else if (arg == "-s" || arg == "--speaker") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing speaker file\n");
                return 1;
            }
            speaker_file = argv[i];
        } else if (arg == "--save-speaker") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing save speaker file\n");
                return 1;
            }
            save_speaker_file = argv[i];
        } else if (arg == "--temperature") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing temperature value\n");
                return 1;
            }
            params.temperature = std::stof(argv[i]);
        } else if (arg == "--top-k") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing top-k value\n");
                return 1;
            }
            params.top_k = std::stoi(argv[i]);
        } else if (arg == "--top-p") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing top-p value\n");
                return 1;
            }
            params.top_p = std::stof(argv[i]);
        } else if (arg == "--max-tokens") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing max-tokens value\n");
                return 1;
            }
            params.max_audio_tokens = std::stoi(argv[i]);
        } else if (arg == "--repetition-penalty") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing repetition-penalty value\n");
                return 1;
            }
            params.repetition_penalty = std::stof(argv[i]);
        } else if (arg == "-l" || arg == "--language") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing language value\n");
                return 1;
            }
            std::string lang = argv[i];
            if (lang == "en" || lang == "english")       params.language_id = 2050;
            else if (lang == "ru" || lang == "russian")  params.language_id = 2069;
            else if (lang == "zh" || lang == "chinese")  params.language_id = 2055;
            else if (lang == "ja" || lang == "japanese")  params.language_id = 2058;
            else if (lang == "ko" || lang == "korean")   params.language_id = 2064;
            else if (lang == "de" || lang == "german")   params.language_id = 2053;
            else if (lang == "fr" || lang == "french")   params.language_id = 2061;
            else if (lang == "es" || lang == "spanish")  params.language_id = 2054;
            else if (lang == "it" || lang == "italian")  params.language_id = 2070;
            else if (lang == "pt" || lang == "portuguese") params.language_id = 2071;
            else {
                fprintf(stderr, "Error: unknown language '%s'. Supported: en,ru,zh,ja,ko,de,fr,es,it,pt\n", lang.c_str());
                return 1;
            }
        } else if (arg == "-j" || arg == "--threads") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing threads value\n");
                return 1;
            }
            params.n_threads = std::stoi(argv[i]);
        } else {
            fprintf(stderr, "Error: unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validate required arguments
    if (model_dir.empty()) {
        fprintf(stderr, "Error: model directory is required\n");
        print_usage(argv[0]);
        return 1;
    }
    
    // Ensure reference audio is provided if we are saving a speaker
    if (!save_speaker_file.empty() && reference_audio.empty()) {
        fprintf(stderr, "Error: --save-speaker requires a reference audio file (-r)\n");
        print_usage(argv[0]);
        return 1;
    }
    
    // Relaxed text validation: text is not required if we are just saving a speaker embedding
    if (text.empty() && !interactive && save_speaker_file.empty()) {
        fprintf(stderr, "Error: text is required unless running in interactive mode (-i) or saving a speaker embedding (--save-speaker)\n");
        print_usage(argv[0]);
        return 1;
    }
    
    // Initialize TTS
    qwen3_tts::Qwen3TTS tts;
    
    fprintf(stderr, "Loading models from: %s\n", model_dir.c_str());
    if (!tts.load_models(model_dir)) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return 1;
    }
 
    // Handle saving speaker embedding if requested
    if (!save_speaker_file.empty() && !reference_audio.empty()) {
        fprintf(stderr, "Extracting speaker embedding from %s...\n", reference_audio.c_str());
        std::vector<float> emb;
        if (tts.extract_speaker_embedding(reference_audio, emb)) {
            if (qwen3_tts::save_speaker_embedding(save_speaker_file, emb)) {
                fprintf(stderr, "Successfully saved speaker embedding to %s\n", save_speaker_file.c_str());
                // If no text was provided, the user just wanted to extract the voice. Exit cleanly.
                if (text.empty() && !interactive) return 0;
            } else {
                fprintf(stderr, "Error: Failed to write speaker file %s\n", save_speaker_file.c_str());
                return 1;
            }
        } else {
            fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
            return 1;
        }
    }

    // Handle loading precomputed speaker embedding
    std::vector<float> precomputed_emb;
    bool has_precomputed = false;
    if (!speaker_file.empty()) {
        if (!qwen3_tts::load_speaker_embedding(speaker_file, precomputed_emb)) {
            fprintf(stderr, "Error: Failed to load speaker embedding from %s\n", speaker_file.c_str());
            return 1;
        }
        has_precomputed = true;
        fprintf(stderr, "Loaded precomputed speaker embedding from %s\n", speaker_file.c_str());
    }

    // Set progress callback
    tts.set_progress_callback([](int tokens, int max_tokens) {
        fprintf(stderr, "\rGenerating: %d/%d tokens", tokens, max_tokens);
    });

    // Helper lambda to run synthesis and save
    auto run_synthesis = [&](const std::string& current_text) {
        qwen3_tts::tts_result result;
        
        if (has_precomputed) {
            fprintf(stderr, "Synthesizing with precomputed voice: \"%s\"\n", current_text.c_str());
            result = tts.synthesize_with_embedding(current_text, precomputed_emb, params);
        } else if (!reference_audio.empty()) {
            fprintf(stderr, "Synthesizing with voice cloning: \"%s\"\n", current_text.c_str());
            result = tts.synthesize_with_voice(current_text, reference_audio, params);
        } else {
            fprintf(stderr, "Synthesizing: \"%s\"\n", current_text.c_str());
            result = tts.synthesize(current_text, params);
        }

        if (!result.success) {
            fprintf(stderr, "\nError: %s\n", result.error_msg.c_str());
            return false;
        }
        
        fprintf(stderr, "\n");
        
        // Save output
        if (!qwen3_tts::save_audio_file(output_file, result.audio, result.sample_rate)) {
            fprintf(stderr, "Error: failed to save output file: %s\n", output_file.c_str());
            return false;
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
        return true;
    };

    if (interactive) {
        fprintf(stderr, "\n==================================================\n");
        fprintf(stderr, "Interactive Mode Started.\n");
        fprintf(stderr, "Output will be overwritten to: %s\n", output_file.c_str());
        if (has_precomputed) {
            fprintf(stderr, "Using precomputed reference voice: %s\n", speaker_file.c_str());
        } else if (!reference_audio.empty()) {
            fprintf(stderr, "Using static reference voice: %s\n", reference_audio.c_str());
        }
        fprintf(stderr, "Type your text and press Enter to synthesize.\n");
        fprintf(stderr, "Type 'exit' or 'quit' to stop the server.\n");
        fprintf(stderr, "==================================================\n\n");

        std::string line;
        while (true) {
            fprintf(stderr, "> ");
            if (!std::getline(std::cin, line)) {
                break; // EOF reached
            }
            
            // Trim whitespace (optional, but good practice)
            size_t first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) continue; // Empty line
            line = line.substr(first, line.find_last_not_of(" \t\r\n") - first + 1);

            if (line == "exit" || line == "quit") {
                break;
            }

            run_synthesis(line);
            fprintf(stderr, "\n");
        }
    } else {
        // Standard single-run mode
        run_synthesis(text);
    }
    
    return 0;
}