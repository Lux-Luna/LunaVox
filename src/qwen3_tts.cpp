#include "qwen3_tts.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

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

namespace {

static bool utf8_next_codepoint(const std::string & text, size_t & i, uint32_t & cp) {
    if (i >= text.size()) return false;
    const unsigned char c0 = (unsigned char) text[i];
    if (c0 < 0x80) {
        cp = c0;
        ++i;
        return true;
    }
    if ((c0 & 0xE0) == 0xC0 && i + 1 < text.size()) {
        cp = ((uint32_t) (c0 & 0x1F) << 6) | (uint32_t) ((unsigned char) text[i + 1] & 0x3F);
        i += 2;
        return true;
    }
    if ((c0 & 0xF0) == 0xE0 && i + 2 < text.size()) {
        cp = ((uint32_t) (c0 & 0x0F) << 12) |
             ((uint32_t) ((unsigned char) text[i + 1] & 0x3F) << 6) |
             (uint32_t) ((unsigned char) text[i + 2] & 0x3F);
        i += 3;
        return true;
    }
    if ((c0 & 0xF8) == 0xF0 && i + 3 < text.size()) {
        cp = ((uint32_t) (c0 & 0x07) << 18) |
             ((uint32_t) ((unsigned char) text[i + 1] & 0x3F) << 12) |
             ((uint32_t) ((unsigned char) text[i + 2] & 0x3F) << 6) |
             (uint32_t) ((unsigned char) text[i + 3] & 0x3F);
        i += 4;
        return true;
    }
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
    int64_t n_han = 0, n_kana = 0, n_hangul = 0, n_cyrillic = 0, n_latin = 0;
    size_t i = 0;
    uint32_t cp = 0;
    while (utf8_next_codepoint(text, i, cp)) {
        if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z') || (cp >= 0x00C0 && cp <= 0x024F)) {
            ++n_latin;
            continue;
        }
        if ((cp >= 0x3040 && cp <= 0x309F) || (cp >= 0x30A0 && cp <= 0x30FF) || (cp >= 0x31F0 && cp <= 0x31FF)) {
            ++n_kana;
            continue;
        }
        if ((cp >= 0xAC00 && cp <= 0xD7AF) || (cp >= 0x1100 && cp <= 0x11FF) || (cp >= 0x3130 && cp <= 0x318F)) {
            ++n_hangul;
            continue;
        }
        if ((cp >= 0x0400 && cp <= 0x04FF) || (cp >= 0x0500 && cp <= 0x052F)) {
            ++n_cyrillic;
            continue;
        }
        if (is_cjk_ideograph(cp)) {
            ++n_han;
        }
    }
    if (n_kana > 0) return 2058;
    if (n_hangul > 0) return 2064;
    if (n_cyrillic > 0) return 2069;
    if (n_han > 0) return 2055;
    if (n_latin > 0) return 2050;
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
        default: return "unknown";
    }
}

static int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct process_memory_snapshot {
    uint64_t rss_bytes = 0;
    uint64_t phys_footprint_bytes = 0;
};

static bool get_process_memory_snapshot(process_memory_snapshot & out) {
#ifdef __APPLE__
    mach_task_basic_info_data_t basic_info = {};
    mach_msg_type_number_t basic_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&basic_info), &basic_count) !=
        KERN_SUCCESS) {
        return false;
    }
    out.rss_bytes = (uint64_t) basic_info.resident_size;
    task_vm_info_data_t vm_info = {};
    mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vm_info), &vm_count) == KERN_SUCCESS) {
        out.phys_footprint_bytes = (uint64_t) vm_info.phys_footprint;
    } else {
        out.phys_footprint_bytes = out.rss_bytes;
    }
    return true;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS memCounters;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &memCounters, sizeof(memCounters))) {
        out.rss_bytes = (uint64_t) memCounters.WorkingSetSize;
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
    static const char * units[] = {"B", "KB", "MB", "GB", "TB"};
    double val = (double) bytes;
    int unit = 0;
    while (val >= 1024.0 && unit < 4) {
        val /= 1024.0;
        ++unit;
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f %s", val, units[unit]);
    return std::string(buf);
}

static bool file_exists_readable(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

static bool env_flag_true(const char * name, bool default_value = false) {
    const char * v = std::getenv(name);
    if (!v || !v[0]) {
        return default_value;
    }
    if (v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T') {
        return true;
    }
    if (v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F') {
        return false;
    }
    return default_value;
}

static uint64_t fnv1a_u64(const int32_t * data, size_t n) {
    static constexpr uint64_t kOffset = 1469598103934665603ULL;
    static constexpr uint64_t kPrime = 1099511628211ULL;
    uint64_t h = kOffset;
    for (size_t i = 0; i < n; ++i) {
        uint32_t v = (uint32_t) data[i];
        for (int b = 0; b < 4; ++b) {
            h ^= (uint64_t) ((v >> (8 * b)) & 0xFFu);
            h *= kPrime;
        }
    }
    return h;
}

static void count_nonfinite(const float * data, int32_t n, int32_t & n_nan, int32_t & n_inf) {
    n_nan = 0;
    n_inf = 0;
    if (!data || n <= 0) {
        return;
    }
    for (int32_t i = 0; i < n; ++i) {
        const float v = data[i];
        if (std::isnan(v)) {
            ++n_nan;
        } else if (!std::isfinite(v)) {
            ++n_inf;
        }
    }
}

static float l2_norm(const float * data, int32_t n) {
    if (!data || n <= 0) {
        return 0.0f;
    }
    long double acc = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        const long double v = (long double) data[i];
        acc += v * v;
    }
    return (float) std::sqrt((double) acc);
}

static void minmax_i32(const int32_t * data, int32_t n, int32_t & out_min, int32_t & out_max) {
    out_min = std::numeric_limits<int32_t>::max();
    out_max = std::numeric_limits<int32_t>::min();
    if (!data || n <= 0) {
        return;
    }
    for (int32_t i = 0; i < n; ++i) {
        out_min = std::min(out_min, data[i]);
        out_max = std::max(out_max, data[i]);
    }
}

static void pcm_peak_rms(const std::vector<float> & audio, float & peak, float & rms) {
    peak = 0.0f;
    rms = 0.0f;
    if (audio.empty()) {
        return;
    }
    long double sum_sq = 0.0;
    for (float v : audio) {
        const float av = std::fabs(v);
        peak = std::max(peak, av);
        sum_sq += (long double) v * (long double) v;
    }
    rms = (float) std::sqrt((double) (sum_sq / (long double) audio.size()));
}

// --- JSON Parsing and Base64 Utils for Vivian.json style references ---

static std::vector<uint8_t> base64_decode(const std::string &in) {
    std::vector<uint8_t> out;
    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++) T["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[i]] = i;

    int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (T[c] == -1) break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(uint8_t((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

static bool simple_extract_json_string(const std::string &json, const std::string &key, std::string &out) {
    std::string qkey = "\"" + key + "\"";
    size_t pos = json.find(qkey);
    if (pos == std::string::npos) return false;
    pos = json.find(":", pos + qkey.length());
    if (pos == std::string::npos) return false;
    pos = json.find("\"", pos);
    if (pos == std::string::npos) return false;
    size_t end = json.find("\"", pos + 1);
    while (end != std::string::npos && json[end - 1] == '\\') {
        end = json.find("\"", end + 1);
    }
    if (end == std::string::npos) return false;
    out = json.substr(pos + 1, end - pos - 1);
    return true;
}

static bool simple_extract_json_codes(const std::string &json, std::vector<int32_t> &codes, int32_t &n_frames) {
    codes.clear();
    n_frames = 0;
    std::string key = "\"codes\"";
    size_t pos = json.find(key);
    if (pos == std::string::npos) return false;
    pos = json.find(":", pos + key.length());
    if (pos == std::string::npos) return false;
    pos = json.find("[", pos);
    if (pos == std::string::npos) return false;
    
    size_t i = pos + 1;
    bool in_inner = false;
    std::string current_num;
    while (i < json.size()) {
        char c = json[i];
        if (c == '[') {
            if (in_inner) return false;
            in_inner = true;
            n_frames++;
        } else if (c == ']') {
            if (in_inner) {
                if (!current_num.empty()) {
                    codes.push_back(std::stoi(current_num));
                    current_num.clear();
                }
                in_inner = false;
            } else {
                break;
            }
        } else if (std::isdigit((unsigned char)c) || c == '-') {
            current_num += c;
        } else if (c == ',' || std::isspace((unsigned char)c)) {
            if (!current_num.empty()) {
                codes.push_back(std::stoi(current_num));
                current_num.clear();
            }
        }
        i++;
    }
    return true;
}

} // namespace

Qwen3TTS::Qwen3TTS() = default;
Qwen3TTS::~Qwen3TTS() = default;

bool Qwen3TTS::load_models_new_layout(const std::string & model_dir, int32_t n_threads) {
    talker_model_path_ = model_dir + "/qwen3_tts_talker.q5_k.gguf";
    predictor_model_path_ = model_dir + "/qwen3_tts_predictor.q8_0.gguf";
    speaker_onnx_path_ = model_dir + "/qwen3_tts_speaker_encoder.fp16.onnx";
    codec_encoder_onnx_path_ = model_dir + "/qwen3_tts_codec_encoder.fp16.onnx";
    decoder_onnx_path_ = model_dir + "/qwen3_tts_decoder.fp16.onnx";
    embeddings_dir_path_ = model_dir + "/embeddings";
    tokenizer_json_path_ = model_dir + "/tokenizer.json";

    const std::string text_emb = embeddings_dir_path_ + "/text_embedding_projected.npy";
    const std::string codec_emb0 = embeddings_dir_path_ + "/codec_embedding_0.npy";

    const std::string required[] = {
        talker_model_path_,
        predictor_model_path_,
        speaker_onnx_path_,
        codec_encoder_onnx_path_,
        decoder_onnx_path_,
        text_emb,
        codec_emb0,
        tokenizer_json_path_,
    };
    for (const auto & p : required) {
        if (!file_exists_readable(p)) {
            error_msg_ = "Model layout missing required file: " + p;
            return false;
        }
    }

    if (!assets_.load(model_dir)) {
        error_msg_ = "Failed to load embedding assets: " + assets_.get_error();
        return false;
    }
    assets_loaded_ = true;

    int64_t t_tok = get_time_ms();
    if (!tokenizer_.load_from_json(tokenizer_json_path_)) {
        error_msg_ = "Failed to load tokenizer.json: " + tokenizer_.get_error();
        return false;
    }
    fprintf(stderr,
            "  Text tokenizer loaded from tokenizer.json: vocab_size=%d (%lld ms)\n",
            tokenizer_.get_config().vocab_size,
            (long long) (get_time_ms() - t_tok));

    const char * lib_dir_env = std::getenv("QWEN3_TTS_LIB_DIR");
    std::string lib_dir = lib_dir_env && lib_dir_env[0] ? std::string(lib_dir_env) : std::string("lib");
    if (!talker_predictor_.load(lib_dir, talker_model_path_, predictor_model_path_, assets_, n_threads)) {
        error_msg_ = "Failed to initialize llama talker/predictor runtime: " + talker_predictor_.get_error();
        return false;
    }
    talker_predictor_loaded_ = true;

    if (!low_mem_mode_) {
        if (!decoder_.load_model(decoder_onnx_path_, n_threads)) {
            error_msg_ = "Failed to load decoder ONNX: " + decoder_.get_error();
            return false;
        }
        decoder_loaded_ = true;
    } else {
        decoder_loaded_ = false;
        fprintf(stderr, "  Decoder ONNX: deferred (lazy load)\n");
    }

    models_loaded_ = true;
    return true;
}

bool Qwen3TTS::load_models(const std::string & model_dir, int32_t n_threads) {
    int64_t t_start = get_time_ms();

    models_loaded_ = false;
    speaker_encoder_loaded_ = false;
    codec_encoder_loaded_ = false;
    talker_predictor_loaded_ = false;
    assets_loaded_ = false;
    decoder_loaded_ = false;
    error_msg_.clear();

    speaker_onnx_path_.clear();
    codec_encoder_onnx_path_.clear();
    talker_model_path_.clear();
    predictor_model_path_.clear();
    embeddings_dir_path_.clear();
    decoder_onnx_path_.clear();
    tokenizer_json_path_.clear();

    talker_predictor_.unload();
    assets_.clear();
    speaker_encoder_.unload_model();
    codec_encoder_.unload_model();
    decoder_.unload_model();

    const char * low_mem_env = std::getenv("QWEN3_TTS_LOW_MEM");
    low_mem_mode_ = low_mem_env && low_mem_env[0] != '\0' && low_mem_env[0] != '0';
    if (low_mem_mode_) {
        fprintf(stderr, "  Low-memory mode enabled (lazy decoder + deferred encoders)\n");
    }

    const int32_t effective_threads = std::max(1, n_threads);
    if (!load_models_new_layout(model_dir, effective_threads)) {
        return false;
    }

    fprintf(stderr, "Loaded models in NEW layout (%lld ms)\n", (long long) (get_time_ms() - t_start));
    return true;
}

tts_result Qwen3TTS::synthesize(const std::string & text, const tts_params & params) {
    tts_result result;
    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }
    return synthesize_internal(text, nullptr, nullptr, 0, params, result);
}

tts_result Qwen3TTS::synthesize_with_voice(
    const std::string & text,
    const std::string & reference_audio,
    const tts_params & params) {
    tts_result result;
    
    // Check if reference_audio is a .json file
    if (reference_audio.length() >= 5 && 
        reference_audio.substr(reference_audio.length() - 5) == ".json") {
             
        if (!models_loaded_) {
            result.error_msg = "Models not loaded";
            return result;
        }
        
        std::ifstream fin(reference_audio, std::ios::binary);
        if (!fin) {
            result.error_msg = "Failed to load reference JSON: " + reference_audio;
            return result;
        }
        std::string json_content((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
        
        std::string spk_emb_b64;
        if (!simple_extract_json_string(json_content, "spk_emb", spk_emb_b64)) {
            result.error_msg = "Failed to find spk_emb in reference JSON";
            return result;
        }
        
        std::vector<uint8_t> emb_bytes = base64_decode(spk_emb_b64);
        if (emb_bytes.empty() || emb_bytes.size() % sizeof(float) != 0) {
            result.error_msg = "Invalid spk_emb base64 data";
            return result;
        }
        std::vector<float> speaker_embedding(emb_bytes.size() / sizeof(float));
        std::memcpy(speaker_embedding.data(), emb_bytes.data(), emb_bytes.size());

        std::vector<int32_t> ref_codes;
        int32_t n_ref_frames = 0;
        if (!simple_extract_json_codes(json_content, ref_codes, n_ref_frames)) {
            result.error_msg = "Failed to parse codes from reference JSON";
            return result;
        }
        
        result.spk_emb_dim = (int32_t) speaker_embedding.size();
        result.spk_emb_l2 = l2_norm(speaker_embedding.data(), (int32_t) speaker_embedding.size());
        count_nonfinite(
            speaker_embedding.data(),
            (int32_t) speaker_embedding.size(),
            result.spk_emb_nan_count,
            result.spk_emb_inf_count);
            
        if (result.spk_emb_nan_count > 0 || result.spk_emb_inf_count > 0) {
            result.error_msg = "[clone/JSON] Invalid speaker embedding: non-finite values detected";
            return result;
        }
        
        if (talker_predictor_.hidden_dim() > 0 && (int32_t) speaker_embedding.size() != talker_predictor_.hidden_dim()) {
            fprintf(stderr, "[clone/JSON] Speaker embedding dim mismatch: got=%d expected=%d. Truncating/Padding for cross-verification...\n",
                          (int) speaker_embedding.size(), (int) talker_predictor_.hidden_dim());
            speaker_embedding.resize(talker_predictor_.hidden_dim(), 0.0f);
        }
        
        result.ref_code_frames = n_ref_frames;
        minmax_i32(ref_codes.data(), (int32_t) ref_codes.size(), result.ref_code_min, result.ref_code_max);
        if (n_ref_frames > 0 && (ref_codes.empty() || (int32_t) ref_codes.size() != n_ref_frames * 16)) {
            result.error_msg = "[clone/JSON] Invalid ref_codes shape; expected T x 16";
            return result;
        }
        
        if (params.print_progress) {
            fprintf(stderr,
                    "Reference features extracted from JSON: spk_dim=%d, ref_codes=%d frames x 16\n",
                    (int) speaker_embedding.size(),
                    n_ref_frames);
        }

        const bool use_clone_icl = env_flag_true("QWEN3_TTS_CLONE_USE_ICL", false);
        const int32_t * ref_codes_ptr = (use_clone_icl && !ref_codes.empty()) ? ref_codes.data() : nullptr;
        const int32_t ref_frames_for_gen = (ref_codes_ptr != nullptr) ? n_ref_frames : 0;
        if (params.print_progress && !use_clone_icl) {
            fprintf(stderr, "Clone mode: using x-vector-only prompt from JSON (set QWEN3_TTS_CLONE_USE_ICL=1 to enable ICL)\n");
        }

        int64_t t_encode = get_time_ms();
        result.t_encode_ms = get_time_ms() - t_encode; // 0 ms basically, keeping stats uniform
        
        return synthesize_internal(
            text,
            speaker_embedding.data(),
            ref_codes_ptr,
            ref_frames_for_gen,
            params,
            result);
    }
    
    std::vector<float> ref_samples;
    int ref_sr = 0;
    if (!load_audio_file(reference_audio, ref_samples, ref_sr)) {
        result.error_msg = "Failed to load reference audio: " + reference_audio;
        return result;
    }
    if (ref_sr != 24000) {
        std::vector<float> resampled;
        if (!resample_windowed_sinc(ref_samples.data(), (int32_t) ref_samples.size(), ref_sr, resampled, 24000)) {
            result.error_msg = "Failed to resample reference audio";
            return result;
        }
        ref_samples = std::move(resampled);
    }
    return synthesize_with_voice(text, ref_samples.data(), (int32_t) ref_samples.size(), params);
}

tts_result Qwen3TTS::synthesize_with_voice(
    const std::string & text,
    const float * ref_samples,
    int32_t n_ref_samples,
    const tts_params & params) {
    tts_result result;
    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }
    if (!ref_samples || n_ref_samples <= 0) {
        result.error_msg = "Invalid reference audio samples";
        return result;
    }

    int64_t t_encode = get_time_ms();
    if (!speaker_encoder_loaded_) {
        if (!speaker_encoder_.load_model(speaker_onnx_path_, params.n_threads)) {
            result.error_msg = "Failed to load speaker encoder ONNX: " + speaker_encoder_.get_error();
            return result;
        }
        speaker_encoder_loaded_ = true;
    }
    if (!codec_encoder_loaded_) {
        if (!codec_encoder_.load_model(codec_encoder_onnx_path_, params.n_threads)) {
            result.error_msg = "Failed to load codec encoder ONNX: " + codec_encoder_.get_error();
            return result;
        }
        codec_encoder_loaded_ = true;
    }

    // Default behavior aligns with Qwen3-TTS-GGUF: keep full reference audio.
    // Optional cap can be enabled via env QWEN3_TTS_CLONE_MAX_REF_SAMPLES (>0).
    std::vector<float> clone_ref_buffer;
    const float * clone_ref_ptr = ref_samples;
    int32_t clone_ref_samples = n_ref_samples;
    const char * clone_cap_env = std::getenv("QWEN3_TTS_CLONE_MAX_REF_SAMPLES");
    if (clone_cap_env && clone_cap_env[0] != '\0') {
        char * end_ptr = nullptr;
        const long parsed = std::strtol(clone_cap_env, &end_ptr, 10);
        if (end_ptr != clone_cap_env && parsed > 0 && parsed < INT32_MAX) {
            const int32_t cap_samples = (int32_t) parsed;
            if (clone_ref_samples > cap_samples) {
                clone_ref_buffer.assign(ref_samples, ref_samples + (size_t) cap_samples);
                clone_ref_ptr = clone_ref_buffer.data();
                clone_ref_samples = cap_samples;
                if (params.print_progress || params.print_timing) {
                    fprintf(stderr,
                            "Clone reference capped by QWEN3_TTS_CLONE_MAX_REF_SAMPLES=%d (original=%d)\n",
                            cap_samples,
                            n_ref_samples);
                }
            }
        }
    }

    std::vector<float> speaker_embedding;
    if (!speaker_encoder_.encode(clone_ref_ptr, clone_ref_samples, speaker_embedding)) {
        result.error_msg = "[clone/speaker] Failed to extract speaker embedding: " + speaker_encoder_.get_error();
        return result;
    }
    result.spk_emb_dim = (int32_t) speaker_embedding.size();
    result.spk_emb_l2 = l2_norm(speaker_embedding.data(), (int32_t) speaker_embedding.size());
    count_nonfinite(
        speaker_embedding.data(),
        (int32_t) speaker_embedding.size(),
        result.spk_emb_nan_count,
        result.spk_emb_inf_count);
    if (result.spk_emb_nan_count > 0 || result.spk_emb_inf_count > 0) {
        result.error_msg = "[clone/speaker] Invalid speaker embedding: non-finite values detected";
        return result;
    }
    if (talker_predictor_.hidden_dim() > 0 && (int32_t) speaker_embedding.size() != talker_predictor_.hidden_dim()) {
        char buf[256];
        std::snprintf(
            buf,
            sizeof(buf),
            "[clone/speaker] Speaker embedding dim mismatch: got=%d expected=%d",
            (int) speaker_embedding.size(),
            (int) talker_predictor_.hidden_dim());
        result.error_msg = buf;
        return result;
    }

    std::vector<int32_t> ref_codes;
    int32_t n_ref_frames = 0;
    if (!codec_encoder_.encode(clone_ref_ptr, clone_ref_samples, ref_codes, n_ref_frames)) {
        result.error_msg = "[clone/codec_encoder] Failed to encode reference audio codes: " + codec_encoder_.get_error();
        return result;
    }
    result.ref_code_frames = n_ref_frames;
    minmax_i32(ref_codes.data(), (int32_t) ref_codes.size(), result.ref_code_min, result.ref_code_max);
    if (n_ref_frames <= 0 || ref_codes.empty() || (int32_t) ref_codes.size() != n_ref_frames * 16) {
        result.error_msg = "[clone/codec_encoder] Invalid ref_codes shape; expected T x 16";
        return result;
    }
    for (int32_t c : ref_codes) {
        if (c < 0 || c >= 2048) {
            result.error_msg = "[clone/codec_encoder] Invalid ref code id out of [0, 2047]";
            return result;
        }
    }
    result.t_encode_ms = get_time_ms() - t_encode;

    if (params.print_progress) {
        fprintf(stderr,
                "Reference features extracted: spk_dim=%d, ref_codes=%d frames x 16\n",
                (int) speaker_embedding.size(),
                n_ref_frames);
    }

    // Keep clone stable when no reference transcript is available:
    // by default, use x-vector-only cloning (speaker embedding only).
    // Enable ICL ref-code fusion explicitly via env:
    //   QWEN3_TTS_CLONE_USE_ICL=1
    const bool use_clone_icl = env_flag_true("QWEN3_TTS_CLONE_USE_ICL", false);
    const int32_t * ref_codes_ptr = (use_clone_icl && !ref_codes.empty()) ? ref_codes.data() : nullptr;
    const int32_t ref_frames_for_gen = (ref_codes_ptr != nullptr) ? n_ref_frames : 0;
    if (params.print_progress && !use_clone_icl) {
        fprintf(stderr, "Clone mode: using x-vector-only prompt (set QWEN3_TTS_CLONE_USE_ICL=1 to enable ICL)\n");
    }

    return synthesize_internal(
        text,
        speaker_embedding.data(),
        ref_codes_ptr,
        ref_frames_for_gen,
        params,
        result);
}

bool Qwen3TTS::extract_speaker_embedding(
    const float * ref_samples,
    int32_t n_ref_samples,
    std::vector<float> & embedding,
    const tts_params & params) {
    if (!models_loaded_) {
        error_msg_ = "Models not loaded";
        return false;
    }
    if (!ref_samples || n_ref_samples <= 0) {
        error_msg_ = "Invalid reference audio samples";
        return false;
    }
    if (!speaker_encoder_loaded_) {
        if (!speaker_encoder_.load_model(speaker_onnx_path_, params.n_threads)) {
            error_msg_ = "Failed to load speaker encoder ONNX: " + speaker_encoder_.get_error();
            return false;
        }
        speaker_encoder_loaded_ = true;
    }
    if (!speaker_encoder_.encode(ref_samples, n_ref_samples, embedding)) {
        error_msg_ = "Failed to extract speaker embedding: " + speaker_encoder_.get_error();
        return false;
    }
    return true;
}

tts_result Qwen3TTS::synthesize_with_embedding(
    const std::string & text,
    const float * embedding,
    int32_t embedding_size,
    const tts_params & params) {
    tts_result result;
    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }
    if (!embedding || embedding_size <= 0) {
        result.error_msg = "Invalid speaker embedding";
        return result;
    }
    if (talker_predictor_.hidden_dim() > 0 && embedding_size != talker_predictor_.hidden_dim()) {
        char buf[256];
        std::snprintf(
            buf,
            sizeof(buf),
            "Speaker embedding dim mismatch: got=%d expected=%d",
            embedding_size,
            talker_predictor_.hidden_dim());
        result.error_msg = buf;
        return result;
    }
    return synthesize_internal(text, embedding, nullptr, 0, params, result);
}

tts_result Qwen3TTS::synthesize_internal(
    const std::string & text,
    const float * speaker_embedding,
    const int32_t * ref_codes,
    int32_t n_ref_frames,
    const tts_params & params,
    tts_result & result) {
    int64_t t_total_start = get_time_ms();

    auto sample_memory = [&](const char * stage) {
        process_memory_snapshot mem;
        if (!get_process_memory_snapshot(mem)) return;
        if (result.mem_rss_start_bytes == 0) {
            result.mem_rss_start_bytes = mem.rss_bytes;
            result.mem_phys_start_bytes = mem.phys_footprint_bytes;
        }
        result.mem_rss_end_bytes = mem.rss_bytes;
        result.mem_phys_end_bytes = mem.phys_footprint_bytes;
        result.mem_rss_peak_bytes = std::max(result.mem_rss_peak_bytes, mem.rss_bytes);
        result.mem_phys_peak_bytes = std::max(result.mem_phys_peak_bytes, mem.phys_footprint_bytes);
        if (params.print_timing) {
            fprintf(stderr,
                    "  [mem] %-24s rss=%s  phys=%s\n",
                    stage,
                    format_bytes(mem.rss_bytes).c_str(),
                    format_bytes(mem.phys_footprint_bytes).c_str());
        }
    };

    sample_memory("synth/start");

    int64_t t_tok = get_time_ms();
    std::vector<int32_t> text_tokens = tokenizer_.encode(text);
    std::vector<int32_t> role_prefix_tokens = tokenizer_.encode("<|im_start|>assistant\n");
    if (role_prefix_tokens.size() != 3 || role_prefix_tokens[0] != 151644 || role_prefix_tokens[1] != 77091 ||
        role_prefix_tokens[2] != 198) {
        role_prefix_tokens = {151644, 77091, 198};
    }
    result.t_tokenize_ms = get_time_ms() - t_tok;
    sample_memory("synth/after-tokenize");

    if (text_tokens.empty()) {
        result.error_msg = "Failed to tokenize text";
        return result;
    }
    if (role_prefix_tokens.empty()) {
        result.error_msg = "Failed to tokenize role prefix";
        return result;
    }

    int32_t effective_language_id = params.auto_language ? detect_language_id_from_text(text) : params.language_id;
    result.effective_language_id = effective_language_id;
    result.used_auto_language = params.auto_language;
    if (params.print_timing || params.print_progress) {
        fprintf(stderr,
                "Language selection: %s -> %s (%d)\n",
                params.auto_language ? "auto" : "manual",
                language_name_from_id(effective_language_id),
                effective_language_id);
    }

    int64_t t_generate = get_time_ms();
    std::vector<int32_t> speech_codes;
    if (!talker_predictor_loaded_) {
        result.error_msg = "Talker/predictor runtime is not loaded";
        return result;
    }
    if (!talker_predictor_.generate(
            text_tokens,
            role_prefix_tokens,
            speaker_embedding,
            ref_codes,
            n_ref_frames,
            params.max_audio_tokens,
            effective_language_id,
            params.repetition_penalty,
            params.temperature,
            params.top_p,
            params.top_k,
            params.predictor_do_sample,
            params.predictor_temperature,
            params.predictor_top_p,
            params.predictor_top_k,
            params.seed,
            params.predictor_seed,
            speech_codes)) {
        result.error_msg = "Failed to generate speech codes: " + talker_predictor_.get_error();
        return result;
    }
    result.eos_step = talker_predictor_.last_eos_step();
    result.trailing_count = talker_predictor_.last_trailing_count();
    result.trailing_consumed = talker_predictor_.last_trailing_consumed();
    result.gen_code_frames = (int32_t) speech_codes.size() / 16;
    minmax_i32(speech_codes.data(), (int32_t) speech_codes.size(), result.gen_code_min, result.gen_code_max);
    result.gen_codes_hash = fnv1a_u64(speech_codes.data(), speech_codes.size());
    result.t_generate_ms = get_time_ms() - t_generate;
    sample_memory("synth/after-generate");

    const int n_codebooks = 16;
    int n_frames = (int) speech_codes.size() / n_codebooks;
    if (n_frames <= 0) {
        result.error_msg = "No speech codes generated";
        return result;
    }
    if ((int32_t) speech_codes.size() != n_frames * n_codebooks) {
        result.error_msg = "[generate] Invalid generated codes shape; expected T x 16";
        return result;
    }
    for (int32_t c : speech_codes) {
        if (c < 0 || c >= 2048) {
            result.error_msg = "[generate] Invalid generated code id out of [0, 2047]";
            return result;
        }
    }
    if (params.print_progress) {
        fprintf(stderr, "Speech codes generated: %d frames x %d codebooks\n", n_frames, n_codebooks);
    }

    int64_t t_decode = get_time_ms();
    if (!decoder_loaded_) {
        if (!decoder_.load_model(decoder_onnx_path_, params.n_threads)) {
            result.error_msg = "Failed to load decoder ONNX: " + decoder_.get_error();
            return result;
        }
        decoder_loaded_ = true;
    }
    if (!decoder_.decode(speech_codes.data(), n_frames, result.audio)) {
        result.error_msg = "Failed to decode speech codes: " + decoder_.get_error();
        return result;
    }
    result.t_decode_ms = get_time_ms() - t_decode;
    pcm_peak_rms(result.audio, result.pcm_peak, result.pcm_rms);
    sample_memory("synth/after-decode");

    if (low_mem_mode_) {
        decoder_.unload_model();
        decoder_loaded_ = false;
        sample_memory("synth/after-decoder-unload");
    }

    result.sample_rate = decoder_.sample_rate();
    result.success = true;
    result.t_total_ms = get_time_ms() - t_total_start;
    sample_memory("synth/end");

    if (params.print_timing) {
        double audio_sec = result.sample_rate > 0 ? (double) result.audio.size() / (double) result.sample_rate : 0.0;
        double wall_sec = (double) result.t_total_ms / 1000.0;
        double rtf = audio_sec > 0.0 ? wall_sec / audio_sec : 0.0;
        double xrt = wall_sec > 0.0 ? audio_sec / wall_sec : 0.0;

        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Tokenization:    %lld ms\n", (long long) result.t_tokenize_ms);
        fprintf(stderr, "  Speaker encode:  %lld ms\n", (long long) result.t_encode_ms);
        fprintf(stderr, "  Code generation: %lld ms\n", (long long) result.t_generate_ms);
        fprintf(stderr, "  Decoder ONNX:    %lld ms\n", (long long) result.t_decode_ms);
        fprintf(stderr, "  Total:           %lld ms\n", (long long) result.t_total_ms);
        fprintf(stderr, "  Audio duration:  %.2f s\n", audio_sec);
        fprintf(stderr, "  Throughput:      %.2fx realtime (RTF=%.3f)\n", xrt, rtf);
    }
    return result;
}

void Qwen3TTS::set_progress_callback(tts_progress_callback_t callback) {
    progress_callback_ = callback;
}

bool load_audio_file(const std::string & path, std::vector<float> & samples, int & sample_rate) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open WAV file: %s\n", path.c_str());
        return false;
    }

    char riff[4];
    if (fread(riff, 1, 4, f) != 4 || strncmp(riff, "RIFF", 4) != 0) {
        fclose(f);
        fprintf(stderr, "ERROR: Not a RIFF file\n");
        return false;
    }
    uint32_t file_size = 0;
    if (fread(&file_size, 4, 1, f) != 1) {
        fclose(f);
        return false;
    }
    (void) file_size;
    char wave[4];
    if (fread(wave, 1, 4, f) != 4 || strncmp(wave, "WAVE", 4) != 0) {
        fclose(f);
        fprintf(stderr, "ERROR: Not a WAVE file\n");
        return false;
    }

    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sr = 0;
    while (!feof(f)) {
        char chunk_id[4];
        uint32_t chunk_size = 0;
        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;

        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            if (fread(&audio_format, 2, 1, f) != 1) break;
            if (fread(&num_channels, 2, 1, f) != 1) break;
            if (fread(&sr, 4, 1, f) != 1) break;
            fseek(f, 6, SEEK_CUR);
            if (fread(&bits_per_sample, 2, 1, f) != 1) break;
            if (chunk_size > 16) {
                fseek(f, chunk_size - 16, SEEK_CUR);
            }
        } else if (strncmp(chunk_id, "data", 4) == 0) {
            sample_rate = (int) sr;
            if (audio_format == 1 && bits_per_sample == 16) {
                int n = (int) (chunk_size / (2 * num_channels));
                std::vector<int16_t> raw((size_t) n * (size_t) num_channels);
                if (fread(raw.data(), 2, raw.size(), f) != raw.size()) {
                    fclose(f);
                    return false;
                }
                samples.assign((size_t) n, 0.0f);
                for (int i = 0; i < n; ++i) {
                    float sum = 0.0f;
                    for (int c = 0; c < num_channels; ++c) {
                        sum += raw[(size_t) i * (size_t) num_channels + (size_t) c] / 32768.0f;
                    }
                    samples[(size_t) i] = sum / (float) num_channels;
                }
            } else if (audio_format == 3 && bits_per_sample == 32) {
                int n = (int) (chunk_size / (4 * num_channels));
                std::vector<float> raw((size_t) n * (size_t) num_channels);
                if (fread(raw.data(), 4, raw.size(), f) != raw.size()) {
                    fclose(f);
                    return false;
                }
                samples.assign((size_t) n, 0.0f);
                for (int i = 0; i < n; ++i) {
                    float sum = 0.0f;
                    for (int c = 0; c < num_channels; ++c) {
                        sum += raw[(size_t) i * (size_t) num_channels + (size_t) c];
                    }
                    samples[(size_t) i] = sum / (float) num_channels;
                }
            } else {
                fclose(f);
                fprintf(stderr, "ERROR: Unsupported WAV format: audio_format=%u, bits=%u\n", audio_format, bits_per_sample);
                return false;
            }
            fclose(f);
            return true;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }
    fclose(f);
    fprintf(stderr, "ERROR: No data chunk found\n");
    return false;
}

bool save_audio_file(const std::string & path, const std::vector<float> & samples, int sample_rate) {
    try {
        std::filesystem::path p(path);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }
    } catch (...) {
        // Continue and let fopen report a concrete error if directory creation fails.
    }
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot create WAV file: %s\n", path.c_str());
        return false;
    }
    uint16_t num_channels = 1;
    uint16_t bits_per_sample = 16;
    uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    uint16_t block_align = num_channels * bits_per_sample / 8;
    uint32_t data_size = (uint32_t) samples.size() * block_align;
    uint32_t file_size = 36 + data_size;

    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size, 4, 1, f);
    uint16_t audio_format = 1;
    fwrite(&audio_format, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    uint32_t sr = (uint32_t) sample_rate;
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);

    for (float s : samples) {
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        int16_t pcm = (int16_t) (s * 32767.0f);
        fwrite(&pcm, 2, 1, f);
    }
    fclose(f);
    return true;
}

} // namespace qwen3_tts
