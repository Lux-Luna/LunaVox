#include "qwen3_tts.h"
#include "gguf_loader.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <cmath>
#include <fstream>
#include <cstdint>
#include <cstdlib>

#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace qwen3_tts {

static bool utf8_next_codepoint(const std::string & text, size_t & i, uint32_t & cp) {
    if (i >= text.size()) {
        return false;
    }

    const unsigned char c0 = (unsigned char) text[i];
    if (c0 < 0x80) {
        cp = c0;
        ++i;
        return true;
    }

    if ((c0 & 0xE0) == 0xC0 && i + 1 < text.size()) {
        cp = ((uint32_t)(c0 & 0x1F) << 6) | (uint32_t)((unsigned char)text[i + 1] & 0x3F);
        i += 2;
        return true;
    }
    if ((c0 & 0xF0) == 0xE0 && i + 2 < text.size()) {
        cp = ((uint32_t)(c0 & 0x0F) << 12) |
             ((uint32_t)((unsigned char)text[i + 1] & 0x3F) << 6) |
             (uint32_t)((unsigned char)text[i + 2] & 0x3F);
        i += 3;
        return true;
    }
    if ((c0 & 0xF8) == 0xF0 && i + 3 < text.size()) {
        cp = ((uint32_t)(c0 & 0x07) << 18) |
             ((uint32_t)((unsigned char)text[i + 1] & 0x3F) << 12) |
             ((uint32_t)((unsigned char)text[i + 2] & 0x3F) << 6) |
             (uint32_t)((unsigned char)text[i + 3] & 0x3F);
        i += 4;
        return true;
    }

    // Invalid UTF-8 byte, consume one byte to keep scanning robust.
    cp = c0;
    ++i;
    return true;
}

static bool is_cjk_ideograph(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0x2F800 && cp <= 0x2FA1F);
}

static int32_t detect_language_id_from_text(const std::string & text) {
    int64_t n_han = 0;
    int64_t n_kana = 0;
    int64_t n_hangul = 0;
    int64_t n_cyrillic = 0;
    int64_t n_latin = 0;

    size_t i = 0;
    uint32_t cp = 0;
    while (utf8_next_codepoint(text, i, cp)) {
        if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z') ||
            (cp >= 0x00C0 && cp <= 0x024F)) {
            ++n_latin;
            continue;
        }
        if ((cp >= 0x3040 && cp <= 0x309F) || // Hiragana
            (cp >= 0x30A0 && cp <= 0x30FF) || // Katakana
            (cp >= 0x31F0 && cp <= 0x31FF)) { // Katakana Phonetic Extensions
            ++n_kana;
            continue;
        }
        if ((cp >= 0xAC00 && cp <= 0xD7AF) || // Hangul syllables
            (cp >= 0x1100 && cp <= 0x11FF) || // Hangul Jamo
            (cp >= 0x3130 && cp <= 0x318F)) { // Hangul Compatibility Jamo
            ++n_hangul;
            continue;
        }
        if ((cp >= 0x0400 && cp <= 0x04FF) || // Cyrillic
            (cp >= 0x0500 && cp <= 0x052F)) {
            ++n_cyrillic;
            continue;
        }
        if (is_cjk_ideograph(cp)) {
            ++n_han;
            continue;
        }
    }

    if (n_kana > 0) return 2058;      // ja
    if (n_hangul > 0) return 2064;    // ko
    if (n_cyrillic > 0) return 2069;  // ru
    if (n_han > 0) return 2055;       // zh
    if (n_latin > 0) return 2050;     // en
    return 2050;
}

static const char * language_name_from_id(int32_t language_id) {
    switch (language_id) {
        case 2050: return "en";
        case 2069: return "ru";
        case 2055: return "zh";
        case 2058: return "ja";
        case 2064: return "ko";
        case 2053: return "de";
        case 2061: return "fr";
        case 2054: return "es";
        case 2070: return "it";
        case 2071: return "pt";
        default:   return "unknown";
    }
}

static int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct process_memory_snapshot {
    uint64_t rss_bytes = 0;
    uint64_t phys_footprint_bytes = 0;
};

static bool get_process_memory_snapshot(process_memory_snapshot & out) {
#ifdef __APPLE__
    mach_task_basic_info_data_t basic_info = {};
    mach_msg_type_number_t basic_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&basic_info), &basic_count) != KERN_SUCCESS) {
        return false;
    }
    out.rss_bytes = (uint64_t) basic_info.resident_size;

    task_vm_info_data_t vm_info = {};
    mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&vm_info), &vm_count) == KERN_SUCCESS) {
        out.phys_footprint_bytes = (uint64_t) vm_info.phys_footprint;
    } else {
        out.phys_footprint_bytes = out.rss_bytes;
    }
    return true;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS memCounters;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &memCounters, sizeof(memCounters))) {
        out.rss_bytes = (uint64_t)memCounters.WorkingSetSize;
        out.phys_footprint_bytes = out.rss_bytes;
        return true;
    }
    return false;
#else
    struct rusage usage = {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return false;
    }
    out.rss_bytes = (uint64_t) usage.ru_maxrss * 1024ULL;
    out.phys_footprint_bytes = out.rss_bytes;
    return true;
#endif
}

static std::string format_bytes(uint64_t bytes) {
    static const char * units[] = { "B", "KB", "MB", "GB", "TB" };
    double val = (double) bytes;
    int unit = 0;
    while (val >= 1024.0 && unit < 4) {
        val /= 1024.0;
        ++unit;
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f %s", val, units[unit]);
    return std::string(buf);
}

static void log_memory_usage(const char * label) {
    process_memory_snapshot mem;
    if (!get_process_memory_snapshot(mem)) {
        fprintf(stderr, "  [mem] %-24s unavailable\n", label);
        return;
    }
    fprintf(stderr, "  [mem] %-24s rss=%s  phys=%s\n",
            label, format_bytes(mem.rss_bytes).c_str(),
            format_bytes(mem.phys_footprint_bytes).c_str());
}

static void resample_linear(const float * input, int input_len, int input_rate,
                            std::vector<float> & output, int output_rate) {
    double ratio = (double)input_rate / output_rate;
    int output_len = (int)((double)input_len / ratio);
    output.resize(output_len);
    
    for (int i = 0; i < output_len; ++i) {
        double src_idx = i * ratio;
        int idx0 = (int)src_idx;
        int idx1 = idx0 + 1;
        double frac = src_idx - idx0;
        
        if (idx1 >= input_len) {
            output[i] = input[input_len - 1];
        } else {
            output[i] = (float)((1.0 - frac) * input[idx0] + frac * input[idx1]);
        }
    }
}

Qwen3TTS::Qwen3TTS() = default;

Qwen3TTS::~Qwen3TTS() = default;

namespace {

static bool file_exists_readable(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

} // namespace

bool Qwen3TTS::load_models_new_layout(const std::string & model_dir, int32_t n_threads) {
    talker_model_path_ = model_dir + "/qwen3_tts_talker.q5_k.gguf";
    predictor_model_path_ = model_dir + "/qwen3_tts_predictor.q8_0.gguf";
    speaker_model_path_ = model_dir + "/qwen3_tts_speaker_encoder.gguf";
    codec_encoder_model_path_ = model_dir + "/qwen3_tts_codec_encoder.gguf";
    decoder_model_path_ = model_dir + "/qwen3_tts_codec_decoder.gguf";
    embeddings_dir_path_ = model_dir + "/embeddings";

    const std::string text_emb = embeddings_dir_path_ + "/text_embedding_projected.npy";
    const std::string codec_emb0 = embeddings_dir_path_ + "/codec_embedding_0.npy";
    const std::string tokenizer_json = model_dir + "/tokenizer.json";

    const std::string required[] = {
        talker_model_path_,
        predictor_model_path_,
        speaker_model_path_,
        codec_encoder_model_path_,
        decoder_model_path_,
        text_emb,
        codec_emb0,
        tokenizer_json,
    };
    for (const auto & p : required) {
        if (!file_exists_readable(p)) {
            error_msg_ = "New layout missing required file: " + p;
            return false;
        }
    }

    if (!assets_.load(model_dir)) {
        error_msg_ = "Failed to load embeddings assets: " + assets_.get_error();
        return false;
    }
    assets_loaded_ = true;

    int64_t t_tokenizer_start = get_time_ms();
    {
        GGUFLoader loader;
        if (!loader.open(speaker_model_path_)) {
            error_msg_ = "Failed to open speaker GGUF for tokenizer load: " + loader.get_error();
            return false;
        }
        if (!tokenizer_.load_from_gguf(loader.get_ctx())) {
            error_msg_ = "Failed to load text tokenizer from speaker GGUF: " + tokenizer_.get_error();
            return false;
        }
        fprintf(stderr, "  Text tokenizer loaded (new layout): vocab_size=%d (%s ms)\n",
                tokenizer_.get_config().vocab_size,
                std::to_string((long long)(get_time_ms() - t_tokenizer_start)).c_str());
    }

    const char * lib_dir_env = std::getenv("QWEN3_TTS_LIB_DIR");
    std::string lib_dir = lib_dir_env && lib_dir_env[0] ? std::string(lib_dir_env) : std::string("lib");
    if (!talker_predictor_.load(lib_dir, talker_model_path_, predictor_model_path_, assets_, n_threads)) {
        error_msg_ = "Failed to initialize llama talker/predictor runtime: " + talker_predictor_.get_error();
        return false;
    }
    talker_predictor_loaded_ = true;

    // Speaker encoder still uses current GGML path, but now with dedicated GGUF.
    encoder_loaded_ = false;

    if (!low_mem_mode_) {
        if (!audio_decoder_.load_model(decoder_model_path_)) {
            error_msg_ = "Failed to load codec decoder GGUF: " + audio_decoder_.get_error();
            return false;
        }
        decoder_loaded_ = true;
    } else {
        decoder_loaded_ = false;
        fprintf(stderr, "  Codec decoder: deferred (lazy load)\n");
    }

    models_loaded_ = true;
    return true;
}

bool Qwen3TTS::load_models(const std::string & model_dir) {
    int64_t t_start = get_time_ms();
    log_memory_usage("load/start");

    // Reset all runtime components first.
    models_loaded_ = false;
    encoder_loaded_ = false;
    talker_predictor_loaded_ = false;
    assets_loaded_ = false;
    decoder_loaded_ = false;
    error_msg_.clear();
    speaker_model_path_.clear();
    codec_encoder_model_path_.clear();
    talker_model_path_.clear();
    predictor_model_path_.clear();
    embeddings_dir_path_.clear();
    decoder_model_path_.clear();

    talker_predictor_.unload();
    assets_.clear();
    audio_decoder_.unload_model();

    const char * low_mem_env = std::getenv("QWEN3_TTS_LOW_MEM");
    low_mem_mode_ = low_mem_env && low_mem_env[0] != '\0' && low_mem_env[0] != '0';
    if (low_mem_mode_) {
        fprintf(stderr, "  Low-memory mode enabled (lazy decoder + component unloads)\n");
    }

    const int32_t n_threads = 4;

    if (!load_models_new_layout(model_dir, n_threads)) {
        return false;
    }

    int64_t t_end = get_time_ms();
    fprintf(stderr, "Loaded models in NEW layout (%s ms)\n", std::to_string((long long)(t_end - t_start)).c_str());
    log_memory_usage("load/end");
    return true;
}

tts_result Qwen3TTS::synthesize(const std::string & text,
                                 const tts_params & params) {
    tts_result result;
    
    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }
    
    // For synthesis without voice cloning, use a zero speaker embedding.
    const int32_t emb_dim = talker_predictor_.hidden_dim();
    std::vector<float> zero_embedding((size_t) emb_dim, 0.0f);
    
    return synthesize_internal(text, zero_embedding.data(), params, result);
}

tts_result Qwen3TTS::synthesize_with_voice(const std::string & text,
                                            const std::string & reference_audio,
                                            const tts_params & params) {
    tts_result result;
    
    std::vector<float> ref_samples;
    int ref_sample_rate;
    if (!load_audio_file(reference_audio, ref_samples, ref_sample_rate)) {
        result.error_msg = "Failed to load reference audio: " + reference_audio;
        return result;
    }
    
    const int target_rate = 24000;
    if (ref_sample_rate != target_rate) {
        fprintf(stderr, "Resampling audio from %d Hz to %d Hz...\n", ref_sample_rate, target_rate);
        std::vector<float> resampled;
        resample_linear(ref_samples.data(), (int)ref_samples.size(), ref_sample_rate, resampled, target_rate);
        ref_samples = std::move(resampled);
    }
    
    return synthesize_with_voice(text, ref_samples.data(), (int32_t)ref_samples.size(), params);
}

tts_result Qwen3TTS::synthesize_with_voice(const std::string & text,
                                            const float * ref_samples, int32_t n_ref_samples,
                                            const tts_params & params) {
    tts_result result;
    
    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    if (!encoder_loaded_) {
        const std::string & encoder_model_path = speaker_model_path_;
        if (encoder_model_path.empty()) {
            result.error_msg = "Internal error: missing speaker model path for lazy encoder load";
            return result;
        }
        int64_t t_encoder_load_start = get_time_ms();
        if (!audio_encoder_.load_model(encoder_model_path)) {
            result.error_msg = "Failed to load speaker encoder: " + audio_encoder_.get_error();
            return result;
        }
        encoder_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Speaker encoder lazy-loaded in %s ms\n",
                    std::to_string((long long)(get_time_ms() - t_encoder_load_start)).c_str());
            log_memory_usage("voice/after-encoder-load");
        }
    }
    
    int64_t t_encode_start = get_time_ms();
    std::vector<float> speaker_embedding;
    
    if (!audio_encoder_.encode(ref_samples, n_ref_samples, speaker_embedding)) {
        result.error_msg = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
        return result;
    }
    result.t_encode_ms = get_time_ms() - t_encode_start;
    
    if (params.print_progress) {
        fprintf(stderr, "Speaker embedding extracted: %s floats\n", std::to_string(speaker_embedding.size()).c_str());
    }
    
    return synthesize_internal(text, speaker_embedding.data(), params, result);
}

bool Qwen3TTS::extract_speaker_embedding(const float * ref_samples, int32_t n_ref_samples,
                                          std::vector<float> & embedding,
                                          const tts_params & params) {
    if (!models_loaded_) {
        error_msg_ = "Models not loaded";
        return false;
    }

    if (!encoder_loaded_) {
        const std::string & encoder_model_path = speaker_model_path_;
        if (encoder_model_path.empty()) {
            error_msg_ = "Internal error: missing speaker model path for lazy encoder load";
            return false;
        }
        int64_t t_encoder_load_start = get_time_ms();
        if (!audio_encoder_.load_model(encoder_model_path)) {
            error_msg_ = "Failed to load speaker encoder: " + audio_encoder_.get_error();
            return false;
        }
        encoder_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Speaker encoder lazy-loaded in %s ms\n",
                    std::to_string((long long)(get_time_ms() - t_encoder_load_start)).c_str());
        }
    }

    if (!audio_encoder_.encode(ref_samples, n_ref_samples, embedding)) {
        error_msg_ = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
        return false;
    }

    return true;
}

tts_result Qwen3TTS::synthesize_with_embedding(const std::string & text,
                                                const float * embedding, int32_t embedding_size,
                                                const tts_params & params) {
    tts_result result;

    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    if (embedding == nullptr || embedding_size <= 0) {
        result.error_msg = "Invalid speaker embedding";
        return result;
    }

    return synthesize_internal(text, embedding, params, result);
}

tts_result Qwen3TTS::synthesize_internal(const std::string & text,
                                          const float * speaker_embedding,
                                          const tts_params & params,
                                          tts_result & result) {
    int64_t t_total_start = get_time_ms();
    auto sample_memory = [&](const char * stage) {
        process_memory_snapshot mem;
        if (!get_process_memory_snapshot(mem)) {
            return;
        }
        if (result.mem_rss_start_bytes == 0) {
            result.mem_rss_start_bytes = mem.rss_bytes;
            result.mem_phys_start_bytes = mem.phys_footprint_bytes;
        }
        result.mem_rss_end_bytes = mem.rss_bytes;
        result.mem_phys_end_bytes = mem.phys_footprint_bytes;
        if (mem.rss_bytes > result.mem_rss_peak_bytes) {
            result.mem_rss_peak_bytes = mem.rss_bytes;
        }
        if (mem.phys_footprint_bytes > result.mem_phys_peak_bytes) {
            result.mem_phys_peak_bytes = mem.phys_footprint_bytes;
        }
        if (params.print_timing) {
            fprintf(stderr, "  [mem] %-24s rss=%s  phys=%s\n",
                    stage,
                    format_bytes(mem.rss_bytes).c_str(),
                    format_bytes(mem.phys_footprint_bytes).c_str());
        }
    };
    sample_memory("synth/start");
    
    // Step 2: Tokenize input text
    int64_t t_tokenize_start = get_time_ms();
    std::vector<int32_t> text_tokens = tokenizer_.encode(text);
    std::vector<int32_t> role_prefix_tokens = tokenizer_.encode("<|im_start|>assistant\n");
    result.t_tokenize_ms = get_time_ms() - t_tokenize_start;
    sample_memory("synth/after-tokenize");
    
    if (text_tokens.empty()) {
        result.error_msg = "Failed to tokenize text";
        return result;
    }
    if (role_prefix_tokens.empty()) {
        result.error_msg = "Failed to tokenize role prefix";
        return result;
    }
    
    if (params.print_progress) {
        fprintf(stderr, "Text tokenized: %s tokens\n", std::to_string(text_tokens.size()).c_str());
        fprintf(stderr, "  Tokens: ");
        for (size_t i = 0; i < std::min(text_tokens.size(), (size_t)10); ++i) {
            fprintf(stderr, "%d ", text_tokens[i]);
        }
        if (text_tokens.size() > 10) fprintf(stderr, "...");
        fprintf(stderr, "\n");
    }

    int32_t effective_language_id = params.language_id;
    if (params.auto_language) {
        effective_language_id = detect_language_id_from_text(text);
    }
    result.effective_language_id = effective_language_id;
    result.used_auto_language = params.auto_language;

    if (params.print_timing || params.print_progress) {
        fprintf(stderr, "Language selection: %s -> %s (%d)\n",
                params.auto_language ? "auto" : "manual",
                language_name_from_id(effective_language_id),
                effective_language_id);
    }
    
    // Step 3: Generate speech codes (llama talker + predictor)
    int64_t t_generate_start = get_time_ms();
    std::vector<int32_t> speech_codes;
    int n_codebooks = 16;
    if (!talker_predictor_loaded_) {
        result.error_msg = "Talker/predictor runtime is not loaded";
        return result;
    }
    if (!talker_predictor_.generate(
            text_tokens,
            role_prefix_tokens,
            speaker_embedding,
            params.max_audio_tokens,
            effective_language_id,
            params.repetition_penalty,
            params.temperature,
            params.top_p,
            params.top_k,
            speech_codes)) {
        result.error_msg = "Failed to generate speech codes: " + talker_predictor_.get_error();
        return result;
    }
    result.t_generate_ms = get_time_ms() - t_generate_start;
    sample_memory("synth/after-generate");
    
    int n_frames = (int)speech_codes.size() / n_codebooks;
    
    if (params.print_progress) {
        fprintf(stderr, "Speech codes generated: %d frames x %d codebooks\n", n_frames, n_codebooks);
    }
    
    if (n_frames == 0) {
        result.error_msg = "No speech codes generated";
        return result;
    }

    // Step 4: Decode speech codes to waveform using vocoder
    int64_t t_decode_start = get_time_ms();
    if (!decoder_loaded_) {
        int64_t t_decoder_load_start = get_time_ms();
        if (decoder_model_path_.empty()) {
            result.error_msg = "Internal error: missing vocoder model path";
            return result;
        }
        if (!audio_decoder_.load_model(decoder_model_path_)) {
            result.error_msg = "Failed to load vocoder: " + audio_decoder_.get_error();
            return result;
        }
        decoder_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Vocoder lazy-loaded in %s ms\n",
                    std::to_string((long long)(get_time_ms() - t_decoder_load_start)).c_str());
            sample_memory("synth/after-vocoder-load");
        }
    }
    
    if (!audio_decoder_.decode(speech_codes.data(), n_frames, result.audio)) {
        result.error_msg = "Failed to decode speech codes: " + audio_decoder_.get_error();
        return result;
    }
    result.t_decode_ms = get_time_ms() - t_decode_start;
    sample_memory("synth/after-decode");

    if (low_mem_mode_) {
        audio_decoder_.unload_model();
        decoder_loaded_ = false;
        sample_memory("synth/after-vocoder-unload");
    }
    
    result.sample_rate = audio_decoder_.get_config().sample_rate;
    result.success = true;
    result.t_total_ms = get_time_ms() - t_total_start;
    sample_memory("synth/end");
    
    if (params.print_timing) {
        const double audio_sec = result.sample_rate > 0
            ? (double) result.audio.size() / (double) result.sample_rate : 0.0;
        const double wall_sec = (double) result.t_total_ms / 1000.0;
        const double realtime_factor = audio_sec > 0.0 ? wall_sec / audio_sec : 0.0;
        const double x_realtime = wall_sec > 0.0 ? audio_sec / wall_sec : 0.0;
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Tokenization:    %s ms\n", std::to_string((long long)result.t_tokenize_ms).c_str());
        fprintf(stderr, "  Speaker encode:  %s ms\n", std::to_string((long long)result.t_encode_ms).c_str());
        fprintf(stderr, "  Code generation: %s ms\n", std::to_string((long long)result.t_generate_ms).c_str());
        fprintf(stderr, "  Vocoder decode:  %s ms\n", std::to_string((long long)result.t_decode_ms).c_str());
        fprintf(stderr, "  Total:           %s ms\n", std::to_string((long long)result.t_total_ms).c_str());
        fprintf(stderr, "  Audio duration:  %.2f s\n", audio_sec);
        fprintf(stderr, "  Throughput:      %.2fx realtime (RTF=%.3f)\n", x_realtime, realtime_factor);
        fprintf(stderr, "\nMemory:\n");
        fprintf(stderr, "  RSS start/end:   %s -> %s\n",
                format_bytes(result.mem_rss_start_bytes).c_str(),
                format_bytes(result.mem_rss_end_bytes).c_str());
        fprintf(stderr, "  RSS peak:        %s\n",
                format_bytes(result.mem_rss_peak_bytes).c_str());
        fprintf(stderr, "  Phys start/end:  %s -> %s\n",
                format_bytes(result.mem_phys_start_bytes).c_str(),
                format_bytes(result.mem_phys_end_bytes).c_str());
        fprintf(stderr, "  Phys peak:       %s\n",
                format_bytes(result.mem_phys_peak_bytes).c_str());
    }
    
    return result;
}

void Qwen3TTS::set_progress_callback(tts_progress_callback_t callback) {
    progress_callback_ = callback;
}

// WAV file loading (16-bit PCM or 32-bit float)
bool load_audio_file(const std::string & path, std::vector<float> & samples, 
                     int & sample_rate) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open WAV file: %s\n", path.c_str());
        return false;
    }
    
    // Read RIFF header
    char riff[4];
    if (fread(riff, 1, 4, f) != 4 || strncmp(riff, "RIFF", 4) != 0) {
        fprintf(stderr, "ERROR: Not a RIFF file\n");
        fclose(f);
        return false;
    }
    
    uint32_t file_size;
    if (fread(&file_size, 4, 1, f) != 1) {
        fclose(f);
        return false;
    }
    
    char wave[4];
    if (fread(wave, 1, 4, f) != 4 || strncmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "ERROR: Not a WAVE file\n");
        fclose(f);
        return false;
    }
    
    // Find fmt and data chunks
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sr = 0;
    uint16_t bits_per_sample = 0;
    
    while (!feof(f)) {
        char chunk_id[4];
        uint32_t chunk_size;
        
        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;
        
        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            if (fread(&audio_format, 2, 1, f) != 1) break;
            if (fread(&num_channels, 2, 1, f) != 1) break;
            if (fread(&sr, 4, 1, f) != 1) break;
            fseek(f, 6, SEEK_CUR);  // Skip byte rate and block align
            if (fread(&bits_per_sample, 2, 1, f) != 1) break;
            
            // Skip any extra format bytes
            if (chunk_size > 16) {
                fseek(f, chunk_size - 16, SEEK_CUR);
            }
        }
        else if (strncmp(chunk_id, "data", 4) == 0) {
            sample_rate = sr;
            
            if (audio_format == 1) {  // PCM
                if (bits_per_sample == 16) {
                    int n_samples = chunk_size / (2 * num_channels);
                    samples.resize(n_samples);
                    
                    std::vector<int16_t> raw(n_samples * num_channels);
                    if (fread(raw.data(), 2, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                        fclose(f);
                        return false;
                    }
                    
                    // Convert to mono float
                    for (int i = 0; i < n_samples; ++i) {
                        float sum = 0.0f;
                        for (int c = 0; c < num_channels; ++c) {
                            sum += raw[i * num_channels + c] / 32768.0f;
                        }
                        samples[i] = sum / num_channels;
                    }
                }
                else if (bits_per_sample == 32) {
                    int n_samples = chunk_size / (4 * num_channels);
                    samples.resize(n_samples);
                    
                    std::vector<int32_t> raw(n_samples * num_channels);
                    if (fread(raw.data(), 4, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                        fclose(f);
                        return false;
                    }
                    
                    // Convert to mono float
                    for (int i = 0; i < n_samples; ++i) {
                        float sum = 0.0f;
                        for (int c = 0; c < num_channels; ++c) {
                            sum += raw[i * num_channels + c] / 2147483648.0f;
                        }
                        samples[i] = sum / num_channels;
                    }
                }
                else {
                    fprintf(stderr, "ERROR: Unsupported bits per sample: %d\n", bits_per_sample);
                    fclose(f);
                    return false;
                }
            }
            else if (audio_format == 3) {  // IEEE float
                int n_samples = chunk_size / (4 * num_channels);
                samples.resize(n_samples);
                
                std::vector<float> raw(n_samples * num_channels);
                if (fread(raw.data(), 4, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                    fclose(f);
                    return false;
                }
                
                // Convert to mono
                for (int i = 0; i < n_samples; ++i) {
                    float sum = 0.0f;
                    for (int c = 0; c < num_channels; ++c) {
                        sum += raw[i * num_channels + c];
                    }
                    samples[i] = sum / num_channels;
                }
            }
            else {
                fprintf(stderr, "ERROR: Unsupported audio format: %d\n", audio_format);
                fclose(f);
                return false;
            }
            
            fclose(f);
            return true;
        }
        else {
            // Skip unknown chunk
            fseek(f, chunk_size, SEEK_CUR);
        }
    }
    
    fprintf(stderr, "ERROR: No data chunk found\n");
    fclose(f);
    return false;
}

// WAV file saving (16-bit PCM at specified sample rate)
bool save_audio_file(const std::string & path, const std::vector<float> & samples,
                     int sample_rate) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot create WAV file: %s\n", path.c_str());
        return false;
    }
    
    // WAV header parameters
    uint16_t num_channels = 1;
    uint16_t bits_per_sample = 16;
    uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    uint16_t block_align = num_channels * bits_per_sample / 8;
    uint32_t data_size = samples.size() * block_align;
    uint32_t file_size = 36 + data_size;
    
    // Write RIFF header
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    
    // Write fmt chunk
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size, 4, 1, f);
    uint16_t audio_format = 1;  // PCM
    fwrite(&audio_format, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    uint32_t sr = sample_rate;
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);
    
    // Write data chunk
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    
    // Convert float samples to 16-bit PCM and write
    for (size_t i = 0; i < samples.size(); ++i) {
        // Clamp to [-1, 1] and convert to int16
        float sample = samples[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        int16_t pcm_sample = (int16_t)(sample * 32767.0f);
        fwrite(&pcm_sample, 2, 1, f);
    }
    
    fclose(f);
    return true;
}

} // namespace qwen3_tts
