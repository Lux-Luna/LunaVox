#include "qwen3_tts.h"
#include "gguf_loader.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <limits>
#include <thread>
#include <mutex>
#include <deque>
#include <atomic>
#include <condition_variable>

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

static std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return s;
}

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

static int32_t parse_positive_int_env(const char * name) {
    const char * env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return 0;
    }

    char * end = nullptr;
    errno = 0;
    long v = std::strtol(env, &end, 10);
    if (errno != 0 || !end || *end != '\0' || v <= 0 ||
        v > (long)(std::numeric_limits<int32_t>::max)()) {
        return 0;
    }
    return (int32_t) v;
}

static bool parse_bool_env(const char * name) {
    const char * env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return false;
    }
    std::string v = to_lower_ascii(env);
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

static int32_t detect_auto_n_threads() {
    // Optional override for benchmarking/deployment:
    // QWEN3_TTS_THREADS=<n>
    const int32_t env_threads = parse_positive_int_env("QWEN3_TTS_THREADS");
    if (env_threads > 0) {
        return env_threads;
    }

    // Prefer higher throughput by default, capped to a sane upper bound.
    const unsigned int hc = std::thread::hardware_concurrency();
    if (hc == 0) {
        return 8;
    }

    int32_t threads = (int32_t) hc;
    if (threads < 1) {
        threads = 1;
    }
    if (threads > 16) {
        threads = 16;
    }
    return threads;
}

static int32_t resolve_effective_n_threads(const tts_params & params) {
    return params.n_threads > 0 ? params.n_threads : detect_auto_n_threads();
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

static bool resolve_tokenizer_model_path(const std::string & model_dir,
                                         std::string & tokenizer_model_path,
                                         std::string & error_msg) {
    const char * candidates[] = {
        "qwen3-tts-tokenizer-f16.gguf",
        "qwen3-tts-tokenizer-q8_0.gguf",
    };

    for (const char * candidate : candidates) {
        std::string path = model_dir + "/" + candidate;
        FILE * f = fopen(path.c_str(), "r");
        if (!f) {
            continue;
        }
        fclose(f);
        tokenizer_model_path = path;
        return true;
    }

    error_msg =
        "Required tokenizer model file not found: tried " +
        model_dir + "/qwen3-tts-tokenizer-f16.gguf and " +
        model_dir + "/qwen3-tts-tokenizer-q8_0.gguf";
    return false;
}

static const char * display_backend_pref(const std::string & pref) {
    return pref.empty() ? "auto" : pref.c_str();
}

Qwen3TTS::Qwen3TTS() = default;

Qwen3TTS::~Qwen3TTS() = default;

bool Qwen3TTS::load_models(const std::string & model_dir) {
    int64_t t_start = get_time_ms();
    log_memory_usage("load/start");

    transformer_.unload_model();
    audio_decoder_.unload_model();
    transformer_loaded_ = false;
    decoder_loaded_ = false;
    
    // Construct model paths (fixed mixed-quant main model; tokenizer prefers F16 with Q8_0 fallback).
    std::string tts_model_path = model_dir + "/qwen3-tts-0.6B-base.gguf";
    std::string tokenizer_model_path;
    {
        FILE * tts_file = fopen(tts_model_path.c_str(), "r");
        if (!tts_file) {
            error_msg_ = "Required model file not found: " + tts_model_path;
            return false;
        }
        fclose(tts_file);
    }
    if (!resolve_tokenizer_model_path(model_dir, tokenizer_model_path, error_msg_)) {
        return false;
    }
    tts_model_path_ = tts_model_path;
    decoder_model_path_ = tokenizer_model_path;
    encoder_loaded_ = false;
    transformer_loaded_ = false;
    decoder_loaded_ = false;

    const char * low_mem_env = std::getenv("QWEN3_TTS_LOW_MEM");
    low_mem_mode_ = low_mem_env && low_mem_env[0] != '\0' && low_mem_env[0] != '0';
    if (low_mem_mode_) {
        fprintf(stderr, "  Low-memory mode enabled (lazy decoder + component unloads)\n");
    }

    const std::string pref_spk = resolve_backend_preference_for_component("AudioTokenizerEncoder");
    const std::string pref_tfm = resolve_backend_preference_for_component("TTSTransformer");
    const char * pref_code_env = std::getenv("QWEN3_TTS_BACKEND_CODE_PREDICTOR");
    const std::string pref_code = (pref_code_env && pref_code_env[0] != '\0')
        ? std::string(pref_code_env)
        : pref_tfm;
    const std::string pref_dec = resolve_backend_preference_for_component("AudioTokenizerDecoder");
    fprintf(stderr, "  Component backend policy:\n");
    fprintf(stderr, "    Speaker Encoder: %s\n", display_backend_pref(pref_spk));
    fprintf(stderr, "    Talker:          %s\n", display_backend_pref(pref_tfm));
    fprintf(stderr, "    Code Predictor:  %s\n", display_backend_pref(pref_code));
    fprintf(stderr, "    Codec Encoder:   disabled on current inference path\n");
    fprintf(stderr, "    Codec Decoder:   %s\n", display_backend_pref(pref_dec));
    
    // Load TTS model (contains text tokenizer + transformer for generation)
    fprintf(stderr, "Loading TTS model from %s...\n", tts_model_path.c_str());
    
    // Load text tokenizer from TTS model
    int64_t t_tokenizer_start = get_time_ms();
    {
        GGUFLoader loader;
        if (!loader.open(tts_model_path)) {
            error_msg_ = "Failed to open TTS model: " + loader.get_error();
            return false;
        }
        
        if (!tokenizer_.load_from_gguf(loader.get_ctx())) {
            error_msg_ = "Failed to load text tokenizer: " + tokenizer_.get_error();
            return false;
        }
        fprintf(stderr, "  Text tokenizer loaded: vocab_size=%d (%lld ms)\n",
                tokenizer_.get_config().vocab_size,
                (long long)(get_time_ms() - t_tokenizer_start));
    }
    log_memory_usage("load/after-tokenizer");
    
    // Speaker encoder is loaded lazily on first voice cloning request.
    fprintf(stderr, "  Speaker encoder: deferred (lazy load)\n");
    
    // Load TTS transformer from TTS model
    int64_t t_transformer_start = get_time_ms();
    if (!transformer_.load_model(tts_model_path)) {
        error_msg_ = "Failed to load TTS transformer: " + transformer_.get_error();
        return false;
    }
    transformer_loaded_ = true;
    fprintf(stderr, "  TTS transformer loaded: hidden_size=%d, n_layers=%d (%lld ms)\n",
            transformer_.get_config().hidden_size, transformer_.get_config().n_layers,
            (long long)(get_time_ms() - t_transformer_start));
    fprintf(stderr, "    Runtime Talker backend: %s\n", transformer_.backend_name());
    fprintf(stderr, "    Runtime Code Predictor backend: %s\n",
            transformer_.code_predictor_backend_name());
    log_memory_usage("load/after-transformer");
    
    if (!low_mem_mode_) {
        // Load vocoder (audio decoder) from tokenizer model
        fprintf(stderr, "Loading vocoder from %s...\n", tokenizer_model_path.c_str());
        int64_t t_decoder_start = get_time_ms();
        if (!audio_decoder_.load_model(tokenizer_model_path)) {
            error_msg_ = "Failed to load vocoder: " + audio_decoder_.get_error();
            return false;
        }
        decoder_loaded_ = true;
        fprintf(stderr, "  Vocoder loaded: sample_rate=%d, n_codebooks=%d (%lld ms)\n",
                audio_decoder_.get_config().sample_rate, audio_decoder_.get_config().n_codebooks,
                (long long)(get_time_ms() - t_decoder_start));
        fprintf(stderr, "    Runtime backend: %s\n", audio_decoder_.backend_name());
        log_memory_usage("load/after-vocoder");
    } else {
        fprintf(stderr, "  Vocoder: deferred (lazy load)\n");
    }
    
    models_loaded_ = true;
    
    int64_t t_end = get_time_ms();
    fprintf(stderr, "All models loaded in %lld ms\n", (long long)(t_end - t_start));
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
    
    // For basic synthesis without voice cloning, we use a zero speaker embedding
    // This will use the model's default voice characteristics
    std::vector<float> zero_embedding(transformer_.get_config().hidden_size, 0.0f);
    
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
        if (tts_model_path_.empty()) {
            result.error_msg = "Internal error: missing TTS model path for lazy encoder load";
            return result;
        }
        int64_t t_encoder_load_start = get_time_ms();
        if (!audio_encoder_.load_model(tts_model_path_)) {
            result.error_msg = "Failed to load speaker encoder: " + audio_encoder_.get_error();
            return result;
        }
        encoder_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Speaker encoder lazy-loaded in %lld ms\n",
                    (long long)(get_time_ms() - t_encoder_load_start));
            fprintf(stderr, "    Runtime backend: %s\n", audio_encoder_.backend_name());
            log_memory_usage("voice/after-encoder-load");
        }
    }
    
    int64_t t_encode_start = get_time_ms();
    std::vector<float> speaker_embedding;
    const int32_t effective_n_threads = resolve_effective_n_threads(params);
    audio_encoder_.set_n_threads(effective_n_threads);
    
    if (!audio_encoder_.encode(ref_samples, n_ref_samples, speaker_embedding)) {
        result.error_msg = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
        return result;
    }
    result.t_encode_ms = get_time_ms() - t_encode_start;
    
    if (params.print_progress) {
        fprintf(stderr, "Speaker embedding extracted: %zu floats\n", speaker_embedding.size());
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
        if (tts_model_path_.empty()) {
            error_msg_ = "Internal error: missing TTS model path for lazy encoder load";
            return false;
        }
        int64_t t_encoder_load_start = get_time_ms();
        if (!audio_encoder_.load_model(tts_model_path_)) {
            error_msg_ = "Failed to load speaker encoder: " + audio_encoder_.get_error();
            return false;
        }
        encoder_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Speaker encoder lazy-loaded in %lld ms\n",
                    (long long)(get_time_ms() - t_encoder_load_start));
            fprintf(stderr, "    Runtime backend: %s\n", audio_encoder_.backend_name());
        }
    }

    const int32_t effective_n_threads = resolve_effective_n_threads(params);
    audio_encoder_.set_n_threads(effective_n_threads);

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
    const int32_t effective_n_threads = resolve_effective_n_threads(params);
    if (params.print_timing || params.print_progress) {
        fprintf(stderr, "Thread config: %d (%s)\n",
                effective_n_threads,
                params.n_threads > 0 ? "manual" : "auto");
    }
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
    std::vector<int32_t> text_tokens = tokenizer_.encode_for_tts(text);
    result.t_tokenize_ms = get_time_ms() - t_tokenize_start;
    sample_memory("synth/after-tokenize");
    
    if (text_tokens.empty()) {
        result.error_msg = "Failed to tokenize text";
        return result;
    }
    
    if (params.print_progress) {
        fprintf(stderr, "Text tokenized: %zu tokens\n", text_tokens.size());
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
    
    bool streaming_decode = params.streaming_decode || parse_bool_env("QWEN3_TTS_STREAMING_DECODE");
    if (streaming_decode && low_mem_mode_) {
        if (params.print_timing || params.print_progress) {
            fprintf(stderr, "  Streaming decode disabled in low-memory mode\n");
        }
        streaming_decode = false;
    }
    int32_t decode_chunk_frames = params.decode_chunk_frames > 0
        ? params.decode_chunk_frames
        : parse_positive_int_env("QWEN3_TTS_DECODE_CHUNK_FRAMES");
    if (decode_chunk_frames <= 0) {
        decode_chunk_frames = 32;
    }
    int32_t max_queued_chunks = params.streaming_max_queued_chunks > 0
        ? params.streaming_max_queued_chunks
        : parse_positive_int_env("QWEN3_TTS_STREAMING_MAX_QUEUED_CHUNKS");
    if (max_queued_chunks <= 0) {
        max_queued_chunks = 4;
    }
    int32_t decode_batch_chunks = params.streaming_decode_batch_chunks > 0
        ? params.streaming_decode_batch_chunks
        : parse_positive_int_env("QWEN3_TTS_STREAMING_DECODE_BATCH_CHUNKS");
    if (decode_batch_chunks <= 0) {
        decode_batch_chunks = 1;
    }
    if (streaming_decode && (params.print_timing || params.print_progress)) {
        fprintf(stderr, "Streaming decode: enabled (chunk=%d frames, max-queue=%d, batch=%d chunks)\n",
                decode_chunk_frames, max_queued_chunks, decode_batch_chunks);
    }
    result.streaming_decode_used = streaming_decode;

    auto ensure_decoder_loaded = [&]() -> bool {
        if (!decoder_loaded_) {
            int64_t t_decoder_load_start = get_time_ms();
            if (decoder_model_path_.empty()) {
                result.error_msg = "Internal error: missing vocoder model path";
                return false;
            }
            if (!audio_decoder_.load_model(decoder_model_path_)) {
                result.error_msg = "Failed to load vocoder: " + audio_decoder_.get_error();
                return false;
            }
            decoder_loaded_ = true;
            if (params.print_timing) {
                fprintf(stderr, "  Vocoder lazy-loaded in %lld ms\n",
                        (long long)(get_time_ms() - t_decoder_load_start));
                fprintf(stderr, "    Runtime backend: %s\n", audio_decoder_.backend_name());
                sample_memory("synth/after-vocoder-load");
            }
        }
        audio_decoder_.set_n_threads(effective_n_threads);
        return true;
    };

    if (streaming_decode && !ensure_decoder_loaded()) {
        return result;
    }

    // Step 3: Generate speech codes using TTS transformer
    int64_t t_generate_start = get_time_ms();
    if (!transformer_loaded_) {
        int64_t t_reload_start = get_time_ms();
        if (!transformer_.load_model(tts_model_path_)) {
            result.error_msg = "Failed to reload TTS transformer: " + transformer_.get_error();
            return result;
        }
        transformer_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Transformer reloaded in %lld ms\n",
                    (long long)(get_time_ms() - t_reload_start));
            sample_memory("synth/after-transformer-reload");
        }
    }
    transformer_.set_n_threads(effective_n_threads);
    transformer_.clear_kv_cache();
    
    std::vector<int32_t> speech_codes;
    std::vector<float> streamed_audio;
    std::atomic<int64_t> streamed_decode_ms(0);
    std::atomic<int32_t> streamed_decode_chunks(0);
    std::atomic<int32_t> streamed_decode_batches(0);
    std::string stream_error;
    std::mutex stream_error_mutex;
    std::atomic<bool> stream_worker_failed(false);
    const int n_codebooks = transformer_.get_config().n_codebooks;
    int64_t stream_generate_end_ms = 0;
    int64_t stream_decode_first_start_ms = -1;
    int64_t stream_decode_last_end_ms = -1;
    std::vector<std::pair<int64_t, int64_t>> stream_decode_windows;

    if (streaming_decode) {
        auto set_stream_error = [&](const std::string & msg) {
            std::lock_guard<std::mutex> lock(stream_error_mutex);
            if (stream_error.empty()) {
                stream_error = msg;
            }
        };

        struct stream_queue_state {
            std::mutex mutex;
            std::condition_variable cv;
            std::deque<std::vector<int32_t>> chunks;
            bool producer_done = false;
        };
        stream_queue_state queue_state;

        auto decode_chunk = [&](const std::vector<int32_t> & chunk_codes) -> bool {
            if (chunk_codes.empty()) {
                return true;
            }
            const int chunk_frames = (int)chunk_codes.size() / n_codebooks;
            std::vector<float> chunk_audio;
            const int64_t t_decode_chunk_start = get_time_ms();
            if (!audio_decoder_.decode(chunk_codes.data(), chunk_frames, chunk_audio)) {
                set_stream_error("Failed to decode speech codes: " + audio_decoder_.get_error());
                return false;
            }
            const int64_t t_decode_chunk_end = get_time_ms();
            streamed_decode_ms.fetch_add(t_decode_chunk_end - t_decode_chunk_start, std::memory_order_relaxed);
            if (stream_decode_first_start_ms < 0) {
                stream_decode_first_start_ms = t_decode_chunk_start;
            }
            stream_decode_last_end_ms = t_decode_chunk_end;
            stream_decode_windows.emplace_back(t_decode_chunk_start, t_decode_chunk_end);
            streamed_decode_batches.fetch_add(1, std::memory_order_relaxed);
            streamed_audio.insert(streamed_audio.end(), chunk_audio.begin(), chunk_audio.end());
            return true;
        };

        std::thread decode_worker([&]() {
            std::vector<std::vector<int32_t>> batch_chunks;
            batch_chunks.reserve((size_t) decode_batch_chunks);
            std::vector<int32_t> merged_chunk_codes;
            while (true) {
                batch_chunks.clear();
                {
                    std::unique_lock<std::mutex> lock(queue_state.mutex);
                    queue_state.cv.wait(lock, [&]() {
                        return stream_worker_failed.load(std::memory_order_relaxed) ||
                               !queue_state.chunks.empty() ||
                               queue_state.producer_done;
                    });
                    if (stream_worker_failed.load(std::memory_order_relaxed)) {
                        return;
                    }
                    if (queue_state.chunks.empty()) {
                        if (queue_state.producer_done) {
                            return;
                        }
                        continue;
                    }
                    batch_chunks.emplace_back(std::move(queue_state.chunks.front()));
                    queue_state.chunks.pop_front();
                    while ((int) batch_chunks.size() < decode_batch_chunks && !queue_state.chunks.empty()) {
                        batch_chunks.emplace_back(std::move(queue_state.chunks.front()));
                        queue_state.chunks.pop_front();
                    }
                    queue_state.cv.notify_all();
                }

                size_t total_codes = 0;
                for (const auto & chunk : batch_chunks) {
                    total_codes += chunk.size();
                }
                merged_chunk_codes.clear();
                merged_chunk_codes.reserve(total_codes);
                for (const auto & chunk : batch_chunks) {
                    merged_chunk_codes.insert(merged_chunk_codes.end(), chunk.begin(), chunk.end());
                }
                streamed_decode_chunks.fetch_add((int32_t) batch_chunks.size(), std::memory_order_relaxed);

                if (!decode_chunk(merged_chunk_codes)) {
                    stream_worker_failed.store(true, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(queue_state.mutex);
                    queue_state.producer_done = true;
                    queue_state.cv.notify_all();
                    return;
                }
            }
        });

        std::vector<int32_t> pending_chunk_codes;
        pending_chunk_codes.reserve((size_t)decode_chunk_frames * (size_t)n_codebooks);

        auto enqueue_pending_chunk = [&]() -> bool {
            if (pending_chunk_codes.empty()) {
                return true;
            }
            std::unique_lock<std::mutex> lock(queue_state.mutex);
            queue_state.cv.wait(lock, [&]() {
                return stream_worker_failed.load(std::memory_order_relaxed) ||
                       (int)queue_state.chunks.size() < max_queued_chunks;
            });
            if (stream_worker_failed.load(std::memory_order_relaxed)) {
                return false;
            }
            queue_state.chunks.emplace_back(std::move(pending_chunk_codes));
            pending_chunk_codes.clear();
            pending_chunk_codes.reserve((size_t)decode_chunk_frames * (size_t)n_codebooks);
            queue_state.cv.notify_all();
            return true;
        };

        auto on_frame = [&](const int32_t * frame_codes, int32_t cb_count, int32_t) -> bool {
            if (stream_worker_failed.load(std::memory_order_relaxed)) {
                return false;
            }
            if (cb_count != n_codebooks) {
                set_stream_error("Internal error: unexpected codebook count");
                stream_worker_failed.store(true, std::memory_order_relaxed);
                return false;
            }
            pending_chunk_codes.insert(pending_chunk_codes.end(),
                                       frame_codes, frame_codes + cb_count);
            const int pending_frames = (int)pending_chunk_codes.size() / n_codebooks;
            if (pending_frames >= decode_chunk_frames) {
                if (!enqueue_pending_chunk()) {
                    return false;
                }
            }
            return true;
        };

        const bool generate_ok = transformer_.generate(
            text_tokens.data(), (int32_t)text_tokens.size(),
            speaker_embedding, params.max_audio_tokens, speech_codes,
            effective_language_id, params.repetition_penalty,
            params.temperature, params.top_k, on_frame);
        stream_generate_end_ms = get_time_ms();
        result.t_generate_ms = stream_generate_end_ms - t_generate_start;

        bool enqueue_ok = true;
        if (generate_ok) {
            enqueue_ok = enqueue_pending_chunk();
        }
        {
            std::lock_guard<std::mutex> lock(queue_state.mutex);
            queue_state.producer_done = true;
        }
        queue_state.cv.notify_all();
        decode_worker.join();

        if (!generate_ok) {
            result.error_msg = "Failed to generate speech codes: " + transformer_.get_error();
            std::lock_guard<std::mutex> lock(stream_error_mutex);
            if (!stream_error.empty()) {
                result.error_msg = stream_error;
            }
            return result;
        }
        if (!enqueue_ok || stream_worker_failed.load(std::memory_order_relaxed)) {
            std::lock_guard<std::mutex> lock(stream_error_mutex);
            if (stream_error.empty()) {
                stream_error = "Streaming decoder worker aborted";
            }
            result.error_msg = stream_error;
            return result;
        }
    } else {
        if (!transformer_.generate(text_tokens.data(), (int32_t)text_tokens.size(),
                                   speaker_embedding, params.max_audio_tokens, speech_codes,
                                   effective_language_id, params.repetition_penalty,
                                   params.temperature, params.top_k)) {
            result.error_msg = "Failed to generate speech codes: " + transformer_.get_error();
            return result;
        }
        stream_generate_end_ms = get_time_ms();
        result.t_generate_ms = stream_generate_end_ms - t_generate_start;
    }
    sample_memory("synth/after-generate");
    
    int n_frames = (int)speech_codes.size() / n_codebooks;
    
    if (params.print_progress) {
        fprintf(stderr, "Speech codes generated: %d frames x %d codebooks\n", n_frames, n_codebooks);
    }
    
    if (n_frames == 0) {
        result.error_msg = "No speech codes generated";
        return result;
    }

    if (low_mem_mode_) {
        transformer_.unload_model();
        transformer_loaded_ = false;
        sample_memory("synth/after-transformer-unload");
    }
    
    // Step 4: Decode speech codes to waveform using vocoder
    if (streaming_decode) {
        result.audio = std::move(streamed_audio);
        result.t_decode_ms = streamed_decode_ms.load(std::memory_order_relaxed);
    } else {
        int64_t t_decode_start = get_time_ms();
        if (!ensure_decoder_loaded()) {
            return result;
        }
        if (!audio_decoder_.decode(speech_codes.data(), n_frames, result.audio)) {
            result.error_msg = "Failed to decode speech codes: " + audio_decoder_.get_error();
            return result;
        }
        result.t_decode_ms = get_time_ms() - t_decode_start;
    }
    sample_memory("synth/after-decode");

    if (low_mem_mode_) {
        audio_decoder_.unload_model();
        decoder_loaded_ = false;
        sample_memory("synth/after-vocoder-unload");
    }
    
    result.sample_rate = audio_decoder_.get_config().sample_rate;
    result.success = true;
    result.t_total_ms = get_time_ms() - t_total_start;

    if (streaming_decode) {
        result.streaming_decode_chunks = streamed_decode_chunks.load(std::memory_order_relaxed);
        result.streaming_decode_batches = streamed_decode_batches.load(std::memory_order_relaxed);
        result.streaming_decode_wall_ms =
            (stream_decode_first_start_ms >= 0 && stream_decode_last_end_ms >= stream_decode_first_start_ms)
                ? (stream_decode_last_end_ms - stream_decode_first_start_ms)
                : 0;

        int64_t overlap_ms = 0;
        for (const auto & window : stream_decode_windows) {
            const int64_t lo = std::max(window.first, t_generate_start);
            const int64_t hi = std::min(window.second, stream_generate_end_ms);
            if (hi > lo) {
                overlap_ms += (hi - lo);
            }
        }
        result.streaming_overlap_ms = overlap_ms;

        const int64_t overlap_den = std::min<int64_t>(result.t_generate_ms, result.t_decode_ms);
        result.streaming_overlap_ratio = overlap_den > 0
            ? (float) overlap_ms / (float) overlap_den
            : 0.0f;

        const int64_t serial_est_ms = result.t_generate_ms + result.t_decode_ms;
        result.streaming_pipeline_saved_ms = serial_est_ms > result.t_total_ms
            ? (serial_est_ms - result.t_total_ms)
            : 0;
    } else {
        result.streaming_decode_chunks = 0;
        result.streaming_decode_batches = 0;
        result.streaming_decode_wall_ms = 0;
        result.streaming_overlap_ms = 0;
        result.streaming_overlap_ratio = 0.0f;
        result.streaming_pipeline_saved_ms = 0;
    }

    sample_memory("synth/end");
    
    if (params.print_timing) {
        const double audio_sec = result.sample_rate > 0
            ? (double) result.audio.size() / (double) result.sample_rate : 0.0;
        const double wall_sec = (double) result.t_total_ms / 1000.0;
        const double realtime_factor = audio_sec > 0.0 ? wall_sec / audio_sec : 0.0;
        const double x_realtime = wall_sec > 0.0 ? audio_sec / wall_sec : 0.0;
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Tokenization:    %lld ms\n", (long long)result.t_tokenize_ms);
        fprintf(stderr, "  Speaker encode:  %lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Code generation: %lld ms\n", (long long)result.t_generate_ms);
        fprintf(stderr, "  Vocoder decode:  %lld ms\n", (long long)result.t_decode_ms);
        if (streaming_decode) {
            fprintf(stderr, "    (streaming mode: generation/decode overlap enabled)\n");
            fprintf(stderr, "    chunks=%d, batches=%d, decode-wall=%lld ms, overlap=%lld ms (ratio=%.2f), pipeline-saved=%lld ms\n",
                    result.streaming_decode_chunks,
                    result.streaming_decode_batches,
                    (long long) result.streaming_decode_wall_ms,
                    (long long) result.streaming_overlap_ms,
                    result.streaming_overlap_ratio,
                    (long long) result.streaming_pipeline_saved_ms);
        }
        fprintf(stderr, "  Total:           %lld ms\n", (long long)result.t_total_ms);
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
    const uint64_t data_size64 = static_cast<uint64_t>(samples.size()) * block_align;
    const uint64_t file_size64 = 36ull + data_size64;
    if (data_size64 > (std::numeric_limits<uint32_t>::max)() ||
        file_size64 > (std::numeric_limits<uint32_t>::max)()) {
        fprintf(stderr, "ERROR: WAV output too large for RIFF/WAVE 32-bit header fields\n");
        fclose(f);
        return false;
    }
    uint32_t data_size = static_cast<uint32_t>(data_size64);
    uint32_t file_size = static_cast<uint32_t>(file_size64);
    
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
