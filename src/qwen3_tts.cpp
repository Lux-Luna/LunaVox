#include "qwen3_tts.h"
#include "logger.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <filesystem>
namespace fs = std::filesystem;

#include "json_utils.h"

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

static std::string join_csv(const std::vector<std::string> & items) {
    std::string out;
    for (size_t i = 0; i < items.size(); ++i) {
        if (i > 0) out += ", ";
        out += items[i];
    }
    return out;
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

static std::string to_lower_ascii(std::string s) {
    for (char & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

static bool starts_with(const std::string & s, const std::string & prefix) {
    return s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
}

static bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() &&
           std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

static int artifact_score(const std::string & name_lc) {
    // Prefer stable runtime artifacts when multiple files coexist.
    if (name_lc.find(".q8_0.") != std::string::npos) return 500;
    if (name_lc.find(".q5_k.") != std::string::npos) return 420;
    if (name_lc.find(".q4_k.") != std::string::npos) return 380;
    if (name_lc.find(".f16.") != std::string::npos) return 320;
    if (name_lc.find(".fp16.") != std::string::npos) return 320;
    if (name_lc.find(".fp32.") != std::string::npos) return 300;
    return 100;
}

static bool resolve_artifact_path(
    const std::string & model_dir,
    const std::vector<std::string> & preferred_names,
    const std::string & prefix_lc,
    const std::string & ext_lc,
    std::string & out_path) {
    for (const std::string & name : preferred_names) {
        const std::string path = model_dir + "/" + name;
        if (file_exists_readable(path)) {
            out_path = path;
            return true;
        }
    }

    std::error_code ec;
    fs::directory_iterator it(fs::path(model_dir), fs::directory_options::skip_permission_denied, ec);
    fs::directory_iterator end;
    if (ec) {
        return false;
    }

    int best_score = std::numeric_limits<int>::min();
    std::string best_name;
    std::string best_path;
    for (; it != end; it.increment(ec)) {
        if (ec) {
            break;
        }
        const auto & entry = *it;
        if (!entry.is_regular_file(ec) || ec) {
            ec.clear();
            continue;
        }
        const std::string name = entry.path().filename().string();
        const std::string name_lc = to_lower_ascii(name);
        if (!starts_with(name_lc, prefix_lc) || !ends_with(name_lc, ext_lc)) {
            continue;
        }

        int score = artifact_score(name_lc);
        for (size_t i = 0; i < preferred_names.size(); ++i) {
            if (name_lc == to_lower_ascii(preferred_names[i])) {
                score = 10000 - (int) i;
                break;
            }
        }

        if (score > best_score || (score == best_score && name_lc < best_name)) {
            best_score = score;
            best_name = name_lc;
            best_path = entry.path().string();
        }
    }

    if (!best_path.empty()) {
        out_path = best_path;
        return true;
    }
    return false;
}

static bool dir_exists(const fs::path & p) {
    std::error_code ec;
    return fs::exists(p, ec) && fs::is_directory(p, ec);
}

static bool try_pick_existing_dir(const fs::path & p, std::string & out) {
    if (dir_exists(p)) {
        out = p.lexically_normal().string();
        return true;
    }
    return false;
}

static bool resolve_model_dir(const std::string & requested, std::string & resolved, std::string & err) {
    fs::path req_path(requested);

    // 1) as-is (works when cwd is project root)
    if (try_pick_existing_dir(req_path, resolved)) {
        return true;
    }

    if (!req_path.is_relative()) {
        err = "Model directory does not exist: " + requested;
        return false;
    }

    std::error_code ec;
    const fs::path cwd = fs::current_path(ec);
    if (!ec) {
        // 2) cwd / requested
        if (try_pick_existing_dir(cwd / req_path, resolved)) {
            return true;
        }
        // 3) cwd / lunavox / requested (common when launching from workspace parent)
        if (try_pick_existing_dir(cwd / "lunavox" / req_path, resolved)) {
            return true;
        }
        // 4) parent(cwd) / requested
        if (try_pick_existing_dir(cwd.parent_path() / req_path, resolved)) {
            return true;
        }
    }

    // 5) light-weight recursive fallback by leaf directory name
    const std::string target_leaf = req_path.filename().string();
    if (target_leaf.empty()) {
        err = "Invalid model directory: " + requested;
        return false;
    }

    std::vector<std::string> hits;
    auto scan_root = [&](const fs::path & root) {
        std::error_code sec;
        if (!dir_exists(root)) {
            return;
        }
        fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, sec);
        fs::recursive_directory_iterator end;
        for (; it != end; it.increment(sec)) {
            if (sec) {
                sec.clear();
                continue;
            }
            if (it.depth() > 4) {
                it.disable_recursion_pending();
                continue;
            }
            const auto & entry = *it;
            std::error_code eec;
            if (!entry.is_directory(eec) || eec) {
                continue;
            }
            if (entry.path().filename().string() == target_leaf && dir_exists(entry.path())) {
                hits.push_back(entry.path().lexically_normal().string());
            }
        }
    };

    if (!ec) {
        scan_root(cwd);
        scan_root(cwd / "lunavox");
    }

    std::sort(hits.begin(), hits.end());
    hits.erase(std::unique(hits.begin(), hits.end()), hits.end());
    if (hits.size() == 1) {
        resolved = hits[0];
        return true;
    }
    if (hits.size() > 1) {
        err = "Model directory '" + requested + "' is ambiguous; candidates: " + hits[0];
        for (size_t i = 1; i < hits.size() && i < 3; ++i) {
            err += ", " + hits[i];
        }
        return false;
    }

    err = "Model directory does not exist: " + requested;
    return false;
}

static std::string sanitize_model_dir_arg(std::string s) {
    auto is_trim_char = [](char c) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\"' || c == '\'';
    };

    while (!s.empty() && is_trim_char(s.front())) {
        s.erase(s.begin());
    }
    while (!s.empty() && is_trim_char(s.back())) {
        s.pop_back();
    }

    // PowerShell multiline continuation mistakes can leave a trailing backtick
    // in the literal argument (e.g. "models/custom`"). Normalize it away.
    while (!s.empty() && s.back() == '`') {
        s.pop_back();
    }
    while (!s.empty() && is_trim_char(s.back())) {
        s.pop_back();
    }

    return s;
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

} // namespace

Qwen3TTS::Qwen3TTS() = default;
Qwen3TTS::~Qwen3TTS() = default;

bool Qwen3TTS::supports_mode(const std::string & mode) const {
    const std::string m = to_lower_ascii(mode);
    if (profile_.model_type == "base") {
        return m == "base" || m == "clone";
    }
    if (profile_.model_type == "custom") {
        return m == "custom";
    }
    if (profile_.model_type == "design") {
        return m == "design";
    }
    return false;
}

bool Qwen3TTS::resolve_language_id(const std::string & name_or_alias, int32_t & language_id_out) const {
    const int32_t id = profile_.resolve_language_id(name_or_alias);
    if (id == -2) {
        return false;
    }
    language_id_out = id;
    return true;
}

std::vector<std::string> Qwen3TTS::supported_speakers() const {
    return profile_.speaker_names_sorted();
}

std::vector<std::string> Qwen3TTS::supported_languages() const {
    std::vector<std::string> names;
    names.reserve(profile_.language_map.size());
    for (const auto & kv : profile_.language_map) {
        names.push_back(kv.first);
    }
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    return names;
}

bool Qwen3TTS::load_model_profile(const std::string & path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        error_msg_ = "Failed to open model profile: " + path;
        return false;
    }
    const std::string json((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());

    RuntimeModelProfile p;
    std::string model_type;
    std::string model_size;
    if (!json_extract_string(json, "model_type", model_type) ||
        !json_extract_string(json, "model_size", model_size)) {
        error_msg_ = "model_profile.json missing model_type/model_size";
        return false;
    }
    p.model_type = profile_to_lower_ascii(model_type);
    p.model_size = profile_to_lower_ascii(model_size);

    if (!json_extract_int(json, "version", p.version) ||
        !json_extract_bool(json, "instruct_support", p.instruct_support) ||
        !json_extract_int(json, "talker_n_ctx", p.talker_n_ctx) ||
        !json_extract_int(json, "talker_n_ctx_train", p.talker_n_ctx_train) ||
        !json_extract_int(json, "predictor_n_ctx", p.predictor_n_ctx) ||
        !json_extract_int(json, "codec_id_start", p.codec_id_start) ||
        !json_extract_int(json, "codec_id_end", p.codec_id_end) ||
        !json_extract_int(json, "codec_num_codebooks", p.codec_num_codebooks) ||
        !json_extract_int(json, "predictor_vocab_size", p.predictor_vocab_size) ||
        !json_extract_int(json, "codec_pad_id", p.codec_pad_id) ||
        !json_extract_int(json, "codec_bos_id", p.codec_bos_id) ||
        !json_extract_int(json, "codec_eos_id", p.codec_eos_id) ||
        !json_extract_int(json, "codec_think_id", p.codec_think_id) ||
        !json_extract_int(json, "codec_nothink_id", p.codec_nothink_id) ||
        !json_extract_int(json, "codec_think_bos_id", p.codec_think_bos_id) ||
        !json_extract_int(json, "codec_think_eos_id", p.codec_think_eos_id) ||
        !json_extract_int(json, "tts_pad_id", p.tts_pad_id) ||
        !json_extract_int(json, "tts_bos_id", p.tts_bos_id) ||
        !json_extract_int(json, "tts_eos_id", p.tts_eos_id) ||
        !json_extract_int(json, "min_new_tokens", p.min_new_tokens) ||
        !json_extract_int(json, "suppress_from", p.suppress_from) ||
        !json_extract_int(json, "suppress_to", p.suppress_to) ||
        !json_extract_int(json, "default_max_new_tokens", p.default_max_new_tokens) ||
        !json_extract_float(json, "default_temperature", p.default_temperature) ||
        !json_extract_float(json, "default_top_p", p.default_top_p) ||
        !json_extract_int(json, "default_top_k", p.default_top_k) ||
        !json_extract_float(json, "default_repetition_penalty", p.default_repetition_penalty) ||
        !json_extract_bool(json, "default_predictor_do_sample", p.default_predictor_do_sample) ||
        !json_extract_float(json, "default_predictor_temperature", p.default_predictor_temperature) ||
        !json_extract_float(json, "default_predictor_top_p", p.default_predictor_top_p) ||
        !json_extract_int(json, "default_predictor_top_k", p.default_predictor_top_k) ||
        !json_extract_int(json, "default_seed", p.default_seed) ||
        !json_extract_int(json, "default_predictor_seed", p.default_predictor_seed)) {
        error_msg_ = "model_profile.json missing required numeric/boolean fields";
        return false;
    }

    std::vector<std::string> speaker_names;
    std::vector<std::string> speaker_dialects;
    std::vector<std::string> language_names;
    std::vector<int32_t> speaker_ids;
    std::vector<int32_t> language_ids;

    if (!json_extract_string_array(json, "speaker_names", speaker_names) ||
        !json_extract_flat_int_array(json, "speaker_ids", speaker_ids) ||
        !json_extract_string_array(json, "speaker_dialects", speaker_dialects) ||
        !json_extract_string_array(json, "language_names", language_names) ||
        !json_extract_flat_int_array(json, "language_ids", language_ids)) {
        error_msg_ = "model_profile.json missing required speaker/language maps";
        return false;
    }

    if (speaker_names.size() != speaker_ids.size() || speaker_names.size() != speaker_dialects.size()) {
        error_msg_ = "model_profile.json speaker map lengths do not match";
        return false;
    }
    if (language_names.size() != language_ids.size()) {
        error_msg_ = "model_profile.json language map lengths do not match";
        return false;
    }

    for (size_t i = 0; i < speaker_names.size(); ++i) {
        const std::string name = profile_to_lower_ascii(speaker_names[i]);
        p.speaker_map[name] = speaker_ids[i];
        p.speaker_dialect_map[name] = profile_to_lower_ascii(speaker_dialects[i]);
    }
    for (size_t i = 0; i < language_names.size(); ++i) {
        p.language_map[profile_to_lower_ascii(language_names[i])] = language_ids[i];
    }

    std::string reason;
    if (!p.is_valid(&reason)) {
        error_msg_ = "Invalid model_profile.json: " + reason;
        return false;
    }

    profile_ = std::move(p);
    return true;
}

bool Qwen3TTS::load_models_new_layout(const std::string & model_dir, int32_t n_threads) {
    talker_model_path_ = model_dir + "/qwen3_tts_talker.q5_k.gguf";
    if (!file_exists_readable(talker_model_path_)) {
        error_msg_ = "Model layout missing Talker GGUF in: " + model_dir;
        return false;
    }
    if (!resolve_artifact_path(
            model_dir,
            {"qwen3_tts_predictor.q8_0.gguf", "qwen3_tts_predictor.q5_k.gguf", "qwen3_tts_predictor.f16.gguf"},
            "qwen3_tts_predictor",
            ".gguf",
            predictor_model_path_)) {
        error_msg_ = "Model layout missing Predictor GGUF in: " + model_dir;
        return false;
    }
    if (!resolve_artifact_path(
            model_dir,
            {"qwen3_tts_decoder.fp16.onnx", "qwen3_tts_decoder.fp32.onnx"},
            "qwen3_tts_decoder",
            ".onnx",
            decoder_onnx_path_)) {
        error_msg_ = "Model layout missing decoder ONNX in: " + model_dir;
        return false;
    }

    // Optional encoders
    if (!resolve_artifact_path(
            model_dir,
            {"qwen3_tts_speaker_encoder.fp16.onnx", "qwen3_tts_speaker_encoder.fp32.onnx"},
            "qwen3_tts_speaker_encoder",
            ".onnx",
            speaker_onnx_path_)) {
        speaker_onnx_path_ = model_dir + "/qwen3_tts_speaker_encoder.fp16.onnx";
    }
    if (!resolve_artifact_path(
            model_dir,
            {"qwen3_tts_codec_encoder.fp16.onnx", "qwen3_tts_codec_encoder.fp32.onnx"},
            "qwen3_tts_codec_encoder",
            ".onnx",
            codec_encoder_onnx_path_)) {
        codec_encoder_onnx_path_ = model_dir + "/qwen3_tts_codec_encoder.fp16.onnx";
    }

    embeddings_dir_path_ = model_dir + "/embeddings";
    tokenizer_json_path_ = model_dir + "/tokenizer.json";
    model_profile_json_path_ = model_dir + "/model_profile.json";

    const std::string text_emb = embeddings_dir_path_ + "/text_embedding_projected.npy";
    const std::string codec_emb0 = embeddings_dir_path_ + "/codec_embedding_0.npy";

    const std::string required[] = {
        talker_model_path_,
        predictor_model_path_,
        decoder_onnx_path_,
        text_emb,
        codec_emb0,
        tokenizer_json_path_,
        model_profile_json_path_,
    };
    for (const auto & p : required) {
        if (!file_exists_readable(p)) {
            error_msg_ = "Model layout missing required file: " + p;
            return false;
        }
    }

    if (!load_model_profile(model_profile_json_path_)) {
        return false;
    }

    // Optional encoders (only needed for on-the-fly .wav voice cloning)
    bool has_speaker_enc = file_exists_readable(speaker_onnx_path_);
    bool has_codec_enc = file_exists_readable(codec_encoder_onnx_path_);
    if (!has_speaker_enc || !has_codec_enc) {
        LOG_INFO("  Encoders: missing or partial (%s/%s). .wav cloning will be disabled.", 
                  has_speaker_enc ? "speaker_ok" : "speaker_missing",
                  has_codec_enc ? "codec_ok" : "codec_missing");
    }

    if (!assets_.load(model_dir, profile_.codec_num_codebooks)) {
        error_msg_ = "Failed to load embedding assets: " + assets_.get_error();
        return false;
    }
    assets_loaded_ = true;

    int64_t t_tok = get_time_ms();
    if (!tokenizer_.load_from_json(tokenizer_json_path_)) {
        error_msg_ = "Failed to load tokenizer.json: " + tokenizer_.get_error();
        return false;
    }
    LOG_DEBUG("  Text tokenizer loaded from tokenizer.json: vocab_size=%d (%lld ms)",
             tokenizer_.get_config().vocab_size,
             (long long) (get_time_ms() - t_tok));

    const char * lib_dir_env = std::getenv("QWEN3_TTS_LIB_DIR");
    std::string lib_dir = lib_dir_env && lib_dir_env[0] ? std::string(lib_dir_env) : std::string("lib/llama");
    if (!talker_predictor_.load(lib_dir, talker_model_path_, predictor_model_path_, assets_, profile_, n_threads)) {
        error_msg_ = "Failed to initialize llama talker/predictor runtime: " + talker_predictor_.get_error();
        return false;
    }
    talker_predictor_loaded_ = true;

    // Check for Llama.cpp acceleration fallback
    std::string metadata_path = find_metadata_json();
    if (!metadata_path.empty()) {
        std::ifstream f(metadata_path);
        if (f.is_open()) {
            std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            size_t llama_start = content.find("\"llama\"");
            if (llama_start != std::string::npos) {
                std::string llama_section = content.substr(llama_start);
                std::string intent;
                if (json_extract_string(llama_section, "backend", intent)) {
                    std::string actual = Logger::instance().get_llama_backend_info();
                    std::string actual_lower = actual;
                    std::transform(actual_lower.begin(), actual_lower.end(), actual_lower.begin(), ::tolower);
                    
                    bool requested_accel = (intent == "cuda" || intent == "vulkan" || intent == "rocm" || intent == "metal" || intent == "sycl");
                    bool is_cpu = (actual_lower.find("cpu") != std::string::npos || actual_lower == "cpu" || actual_lower.empty());
                    
                    if (requested_accel && is_cpu) {
                        LOG_WARN("Llama.cpp dependency (%s) is incomplete, falling back to CPU.", intent.c_str());
                    }
                }
            }
        }
    }

    if (!low_mem_mode_) {
        if (!decoder_.load_model(decoder_onnx_path_, n_threads)) {
            error_msg_ = "Failed to load decoder ONNX: " + decoder_.get_error();
            return false;
        }
        decoder_loaded_ = true;
        LOG_DEBUG("  Decoder providers: %s", decoder_.provider_summary().c_str());
    } else {
        decoder_loaded_ = false;
        LOG_DEBUG("  Decoder ONNX: deferred (lazy load)");
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
    hot_rows_preloaded_ = false;
    error_msg_.clear();

    speaker_onnx_path_.clear();
    codec_encoder_onnx_path_.clear();
    talker_model_path_.clear();
    predictor_model_path_.clear();
    embeddings_dir_path_.clear();
    decoder_onnx_path_.clear();
    tokenizer_json_path_.clear();
    model_profile_json_path_.clear();
    profile_ = RuntimeModelProfile{};

    talker_predictor_.unload();
    assets_.clear();
    speaker_encoder_.unload_model();
    codec_encoder_.unload_model();
    decoder_.unload_model();

    const char * low_mem_env = std::getenv("QWEN3_TTS_LOW_MEM");
    low_mem_mode_ = low_mem_env && low_mem_env[0] != '\0' && low_mem_env[0] != '0';
    if (low_mem_mode_) {
        LOG_DEBUG("  Low-memory mode enabled (lazy decoder + deferred encoders)");
    }

    const std::string model_dir_sanitized = sanitize_model_dir_arg(model_dir);
    if (model_dir_sanitized.empty()) {
        error_msg_ = "Model directory is empty after argument normalization";
        return false;
    }
    if (model_dir_sanitized != model_dir) {
        LOG_WARN("Normalized model_dir argument: '%s' -> '%s'", model_dir.c_str(), model_dir_sanitized.c_str());
    }

    std::string resolved_model_dir;
    std::string resolve_err;
    if (!resolve_model_dir(model_dir_sanitized, resolved_model_dir, resolve_err)) {
        error_msg_ = resolve_err;
        return false;
    }
    if (resolved_model_dir != model_dir_sanitized) {
        LOG_INFO("Resolved model_dir: %s -> %s", model_dir_sanitized.c_str(), resolved_model_dir.c_str());
    }

    const int32_t effective_threads = std::max(1, n_threads);
    if (!load_models_new_layout(resolved_model_dir, effective_threads)) {
        return false;
    }

    LOG_DEBUG("Loaded models in NEW layout (%lld ms)", (long long) (get_time_ms() - t_start));
    return true;
}

bool Qwen3TTS::preload_hot_embedding_rows() {
    std::lock_guard<std::mutex> guard(hot_rows_preload_mu_);
    if (hot_rows_preloaded_) {
        return true;
    }
    if (!assets_loaded_ || !assets_.is_loaded()) {
        error_msg_ = "Embedding assets are not loaded";
        return false;
    }

    const int32_t dim = assets_.hidden_dim();
    if (dim <= 0) {
        error_msg_ = "Invalid embedding hidden dim during hot preload";
        return false;
    }

    volatile float touch_sink = 0.0f;
    auto touch_row = [&](const float * row, const char * name) -> bool {
        if (!row) {
            error_msg_ = std::string("Missing embedding row during hot preload: ") + name;
            return false;
        }
        touch_sink += row[0];
        touch_sink += row[(size_t) dim / 2];
        touch_sink += row[(size_t) dim - 1];
        return true;
    };

    std::vector<int32_t> hot_text_rows = {
        profile_.tts_pad_id,
        profile_.tts_bos_id,
        profile_.tts_eos_id,
    };
    auto append_hot_text_tokens = [&](const std::string & marker) {
        const std::vector<int32_t> marker_tokens = tokenizer_.encode(marker);
        hot_text_rows.insert(hot_text_rows.end(), marker_tokens.begin(), marker_tokens.end());
    };
    append_hot_text_tokens("<|im_start|>assistant\n");
    append_hot_text_tokens("<|im_start|>user\n");
    append_hot_text_tokens("<|im_end|>\n");
    std::sort(hot_text_rows.begin(), hot_text_rows.end());
    hot_text_rows.erase(std::unique(hot_text_rows.begin(), hot_text_rows.end()), hot_text_rows.end());

    std::vector<int32_t> hot_codec_q0_rows = {
        profile_.codec_pad_id,
        profile_.codec_bos_id,
        profile_.codec_eos_id,
        profile_.codec_think_id,
        profile_.codec_nothink_id,
        profile_.codec_think_bos_id,
        profile_.codec_think_eos_id,
    };
    for (const auto & kv : profile_.language_map) {
        hot_codec_q0_rows.push_back(kv.second);
    }
    for (const auto & kv : profile_.speaker_map) {
        hot_codec_q0_rows.push_back(kv.second);
    }
    std::sort(hot_codec_q0_rows.begin(), hot_codec_q0_rows.end());
    hot_codec_q0_rows.erase(std::unique(hot_codec_q0_rows.begin(), hot_codec_q0_rows.end()), hot_codec_q0_rows.end());

    for (int32_t tid : hot_text_rows) {
        if (!touch_row(assets_.text_row(tid), "hot text row")) {
            return false;
        }
    }

    for (int32_t cid : hot_codec_q0_rows) {
        if (!touch_row(assets_.codec_row(0, cid), "hot codec q0 row")) {
            return false;
        }
    }

    // Touch one row from each codec table to reduce first-access page faults.
    for (int32_t q = 0; q < profile_.codec_num_codebooks; ++q) {
        if (!touch_row(assets_.codec_row(q, 0), "codec table first row")) {
            return false;
        }
    }

    (void) touch_sink;
    hot_rows_preloaded_ = true;
    LOG_DEBUG("  Embedding hot-row preload completed");
    return true;
}

tts_result Qwen3TTS::synthesize(const std::string & text, const tts_params & params) {
    tts_result result;
    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }
    if (!supports_mode("base")) {
        result.error_msg = "Current model does not support base mode synthesis";
        return result;
    }
    return synthesize_internal(text, nullptr, nullptr, 0, params, result);
}

tts_result Qwen3TTS::synthesize_with_voice(
    const std::string & text,
    const std::string & reference_audio,
    const tts_params & params) {
    tts_result result;
    tts_params params_copy = params;
    if (!supports_mode("clone")) {
        result.error_msg = "Current model does not support clone mode synthesis";
        return result;
    }
    
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
        if (!json_extract_string(json_content, "spk_emb", spk_emb_b64)) {
            result.error_msg = "Failed to find spk_emb in reference JSON";
            return result;
        }
        
        std::string ref_text_json;
        if (json_extract_string(json_content, "ref-text", ref_text_json) ||
            json_extract_string(json_content, "text", ref_text_json)) {
            params_copy.ref_text = ref_text_json;
        }
        
        std::vector<uint8_t> emb_bytes = qwen3_tts::base64_decode(spk_emb_b64);
        if (emb_bytes.empty() || emb_bytes.size() % sizeof(float) != 0) {
            result.error_msg = "Invalid spk_emb base64 data";
            return result;
        }
        std::vector<float> speaker_embedding(emb_bytes.size() / sizeof(float));
        std::memcpy(speaker_embedding.data(), emb_bytes.data(), emb_bytes.size());

        std::vector<int32_t> ref_codes;
        int32_t n_ref_frames = 0;
        // Re-implementing n_ref_frames calculation since json_extract_flat_int_array is flat
        if (!json_extract_flat_int_array(json_content, "codes", ref_codes)) {
            result.error_msg = "Failed to parse codes from reference JSON";
            return result;
        }
        if (ref_codes.empty() || ((int32_t) ref_codes.size() % profile_.codec_num_codebooks) != 0) {
            result.error_msg = "[clone/JSON] Invalid ref_codes shape; expected T x codec_num_codebooks";
            return result;
        }
        n_ref_frames = (int32_t) ref_codes.size() / profile_.codec_num_codebooks;
        
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
            LOG_WARN("[clone/JSON] Speaker embedding dim mismatch: got=%d expected=%d. Truncating/Padding for cross-verification...",
                          (int) speaker_embedding.size(), (int) talker_predictor_.hidden_dim());
            speaker_embedding.resize(talker_predictor_.hidden_dim(), 0.0f);
        }
        
        result.ref_code_frames = n_ref_frames;
        result.ref_codebooks = profile_.codec_num_codebooks;
        minmax_i32(ref_codes.data(), (int32_t) ref_codes.size(), result.ref_code_min, result.ref_code_max);
        if (n_ref_frames <= 0 ||
            (int32_t) ref_codes.size() != n_ref_frames * profile_.codec_num_codebooks) {
            result.error_msg = "[clone/JSON] Invalid ref_codes shape; expected T x codec_num_codebooks";
            return result;
        }
        for (int32_t c : ref_codes) {
            if (c < profile_.codec_id_start || c >= profile_.codec_id_end) {
                result.error_msg = "[clone/JSON] Invalid ref code id out of codec range from model_profile.json";
                return result;
            }
        }
        
        if (params.print_progress) {
            LOG_INFO("Reference features extracted from JSON: spk_dim=%d, ref_codes=%d frames x %d",
                    (int) speaker_embedding.size(),
                    n_ref_frames,
                    profile_.codec_num_codebooks);
        }

        const bool use_clone_icl = !params_copy.ref_text.empty() && !ref_codes.empty() && n_ref_frames > 0;
        const int32_t * ref_codes_ptr = use_clone_icl ? ref_codes.data() : nullptr;
        const int32_t ref_frames_for_gen = (ref_codes_ptr != nullptr) ? n_ref_frames : 0;
        if (params.print_progress && !use_clone_icl) {
            LOG_INFO("Clone mode: using x-vector-only prompt (missing ref_text for ICL)");
        }

        int64_t t_encode = get_time_ms();
        result.t_encode_ms = get_time_ms() - t_encode; // 0 ms basically, keeping stats uniform

        result.ort_provider_speaker_encoder = speaker_encoder_loaded_ ? speaker_encoder_.provider_summary() : "not_loaded";
        result.ort_provider_codec_encoder = codec_encoder_loaded_ ? codec_encoder_.provider_summary() : "not_loaded";
        result.ort_provider_decoder = decoder_loaded_ ? decoder_.provider_summary() : "not_loaded";
        
        return synthesize_internal(
            text,
            speaker_embedding.data(),
            ref_codes_ptr,
            ref_frames_for_gen,
            params_copy,
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
    if (!supports_mode("clone")) {
        result.error_msg = "Current model does not support clone mode synthesis";
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
        LOG_INFO("  Speaker encoder providers: %s", speaker_encoder_.provider_summary().c_str());
    }
    if (!codec_encoder_loaded_) {
        if (!codec_encoder_.load_model(codec_encoder_onnx_path_, params.n_threads)) {
            result.error_msg = "Failed to load codec encoder ONNX: " + codec_encoder_.get_error();
            return result;
        }
        codec_encoder_loaded_ = true;
        LOG_INFO("  Codec encoder providers: %s", codec_encoder_.provider_summary().c_str());
    }

    std::vector<float> speaker_embedding;
    if (!speaker_encoder_.encode(ref_samples, n_ref_samples, speaker_embedding)) {
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
    if (!codec_encoder_.encode(ref_samples, n_ref_samples, ref_codes, n_ref_frames)) {
        result.error_msg = "[clone/codec_encoder] Failed to encode reference audio codes: " + codec_encoder_.get_error();
        return result;
    }
    result.ref_code_frames = n_ref_frames;
    result.ref_codebooks = profile_.codec_num_codebooks;
    minmax_i32(ref_codes.data(), (int32_t) ref_codes.size(), result.ref_code_min, result.ref_code_max);
    if (n_ref_frames <= 0 || ref_codes.empty() ||
        (int32_t) ref_codes.size() != n_ref_frames * profile_.codec_num_codebooks) {
        result.error_msg = "[clone/codec_encoder] Invalid ref_codes shape; expected T x codec_num_codebooks";
        return result;
    }
    for (int32_t c : ref_codes) {
        if (c < profile_.codec_id_start || c >= profile_.codec_id_end) {
            result.error_msg = "[clone/codec_encoder] Invalid ref code id out of codec range from model_profile.json";
            return result;
        }
    }
    result.t_encode_ms = get_time_ms() - t_encode;

    if (params.print_progress) {
        LOG_INFO("Reference features extracted: spk_dim=%d, ref_codes=%d frames x %d",
                (int) speaker_embedding.size(),
                n_ref_frames,
                profile_.codec_num_codebooks);
    }

    const bool use_clone_icl = !params.ref_text.empty() && !ref_codes.empty() && n_ref_frames > 0;
    const int32_t * ref_codes_ptr = use_clone_icl ? ref_codes.data() : nullptr;
    const int32_t ref_frames_for_gen = (ref_codes_ptr != nullptr) ? n_ref_frames : 0;
    if (params.print_progress && !use_clone_icl) {
        LOG_INFO("Clone mode: using x-vector-only prompt (missing ref_text for ICL)");
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

tts_result Qwen3TTS::synthesize_custom(
    const std::string & text,
    const std::string & speaker,
    const std::string & instruct,
    const tts_params & params_in) {
    tts_result result;
    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }
    if (!supports_mode("custom")) {
        result.error_msg = "Current model does not support custom mode synthesis";
        return result;
    }
    if (!profile_.instruct_support && !instruct.empty()) {
        result.error_msg = "This model does not support instruct in custom mode";
        return result;
    }

    const int32_t spk_id = profile_.resolve_speaker_id(speaker);
    if (spk_id < 0) {
        const auto speakers = supported_speakers();
        result.error_msg = "Unknown speaker name: " + speaker + ". Available: " + join_csv(speakers);
        return result;
    }

    // Get speaker embedding from codec embedding row
    const float * emb = assets_.codec_row(0, spk_id);
    if (!emb) {
        result.error_msg = "Failed to get codec embedding for speaker ID ";
        result.error_msg += std::to_string(spk_id);
        return result;
    }

    // Build params with instruct
    tts_params p = params_in;
    p.instruct = instruct;

    if (params_in.print_progress) {
        LOG_INFO("Custom voice features extracted: speaker_id=%d, spk_dim=%d", spk_id, (int)assets_.hidden_dim());
    }

    return synthesize_internal(text, emb, nullptr, 0, p, result);
}

tts_result Qwen3TTS::synthesize_design(
    const std::string & text,
    const std::string & instruct,
    const tts_params & params_in) {
    tts_result result;
    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }
    if (!supports_mode("design")) {
        result.error_msg = "Current model does not support design mode synthesis";
        return result;
    }
    if (!profile_.instruct_support && !instruct.empty()) {
        result.error_msg = "This model does not support instruct in design mode";
        return result;
    }

    if (instruct.empty()) {
        result.error_msg = "Voice Design mode requires non-empty instruct text";
        return result;
    }

    // Design mode: no speaker embedding, only instruct
    tts_params p = params_in;
    p.instruct = instruct;

    if (params_in.print_progress) {
        LOG_INFO("Voice design features: instruct=\"%s\"", instruct.c_str());
    }

    return synthesize_internal(text, nullptr, nullptr, 0, p, result);
}

tts_result Qwen3TTS::synthesize_internal(
    const std::string & text,
    const float * speaker_embedding,
    const int32_t * ref_codes,
    int32_t n_ref_frames,
    const tts_params & params,
    tts_result & result) {
    int64_t t_total_start = get_time_ms();
    result.ort_provider_speaker_encoder = speaker_encoder_loaded_ ? speaker_encoder_.provider_summary() : "not_loaded";
    result.ort_provider_codec_encoder = codec_encoder_loaded_ ? codec_encoder_.provider_summary() : "not_loaded";
    result.ort_provider_decoder = decoder_loaded_ ? decoder_.provider_summary() : "not_loaded";

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
            LOG_DEBUG("  [mem] %-24s rss=%s  phys=%s",
                    stage,
                    format_bytes(mem.rss_bytes).c_str(),
                    format_bytes(mem.phys_footprint_bytes).c_str());
        }
    };

    sample_memory("synth/start");

    if (profile_.model_type == "base" && !params.instruct.empty()) {
        result.error_msg = "Base model forbids instruct in synthesis path";
        return result;
    }
    if (!profile_.instruct_support && !params.instruct.empty()) {
        result.error_msg = "This model does not support instruct";
        return result;
    }

    if (!preload_hot_embedding_rows()) {
        result.error_msg = "Failed to preload embedding hot rows: " + error_msg_;
        return result;
    }

    int64_t t_tok = get_time_ms();
    const std::vector<int32_t> assistant_prefix_tokens = tokenizer_.encode("<|im_start|>assistant\n");
    const std::vector<int32_t> user_prefix_tokens = tokenizer_.encode("<|im_start|>user\n");
    const std::vector<int32_t> end_newline_tokens = tokenizer_.encode("<|im_end|>\n");
    if (assistant_prefix_tokens.empty() || user_prefix_tokens.empty() || end_newline_tokens.empty()) {
        result.error_msg = "Failed to build ChatML boundary tokens from tokenizer";
        return result;
    }
    // Match qwen3-tts-gguf semantics: raw text/ref_text are encoded directly.
    std::vector<int32_t> text_tokens = tokenizer_.encode(text);

    std::vector<int32_t> ref_text_tokens;
    if (!params.ref_text.empty()) {
        ref_text_tokens = tokenizer_.encode(params.ref_text);
    }
    std::vector<int32_t> role_prefix_tokens = assistant_prefix_tokens;

    // Build instruct block explicitly via ChatML boundaries:
    // <|im_start|>user\n + encode(instruct) + <|im_end|>\n
    std::vector<int32_t> instruct_tokens;
    if (!params.instruct.empty()) {
        std::vector<int32_t> inner = tokenizer_.encode(params.instruct);
        instruct_tokens.reserve(user_prefix_tokens.size() + inner.size() + end_newline_tokens.size());
        instruct_tokens.insert(instruct_tokens.end(), user_prefix_tokens.begin(), user_prefix_tokens.end());
        instruct_tokens.insert(instruct_tokens.end(), inner.begin(), inner.end());
        instruct_tokens.insert(instruct_tokens.end(), end_newline_tokens.begin(), end_newline_tokens.end());

        if (params.print_timing) {
            LOG_DEBUG("Instruct tokens: %d tokens for block: %s",
                    (int)instruct_tokens.size(), params.instruct.c_str());
        }
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

    int32_t effective_language_id = params.language_id;
    result.effective_language_id = effective_language_id;
    result.used_auto_language = false;
    if (params.print_timing || params.print_progress) {
        const std::string lang_desc = (effective_language_id < 0)
                                          ? std::string("Auto(None): no language token")
                                          : (std::string("explicit codec id=") + std::to_string(effective_language_id));
        LOG_INFO("Language selection: %s", lang_desc.c_str());
    }

    int32_t effective_max_audio_tokens = params.max_audio_tokens;
    if (effective_max_audio_tokens <= 0) {
        effective_max_audio_tokens = profile_.default_max_new_tokens;
    }

    int64_t t_generate = get_time_ms();
    std::vector<int32_t> speech_codes;
    if (!talker_predictor_loaded_) {
        result.error_msg = "Talker/predictor runtime is not loaded";
        return result;
    }
    const bool gen_ok = talker_predictor_.generate(
            text_tokens,
            ref_text_tokens,
            role_prefix_tokens,
            instruct_tokens,
            speaker_embedding,
            ref_codes,
            n_ref_frames,
            effective_max_audio_tokens,
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
            speech_codes);
    result.ctx_required = talker_predictor_.last_ctx_required();
    result.ctx_allocated = talker_predictor_.last_ctx_allocated();
    result.ctx_cap = talker_predictor_.last_ctx_cap();
    result.ctx_overflow = talker_predictor_.last_ctx_overflow();
    if (!gen_ok) {
        result.error_msg = "Failed to generate speech codes: " + talker_predictor_.get_error();
        return result;
    }
    result.eos_step = talker_predictor_.last_eos_step();
    result.gen_code_frames = (int32_t) speech_codes.size() / profile_.codec_num_codebooks;
    result.gen_codebooks = profile_.codec_num_codebooks;
    minmax_i32(speech_codes.data(), (int32_t) speech_codes.size(), result.gen_code_min, result.gen_code_max);
    result.gen_codes_hash = fnv1a_u64(speech_codes.data(), speech_codes.size());
    result.t_generate_ms = get_time_ms() - t_generate;
    sample_memory("synth/after-generate");

    const int n_codebooks = profile_.codec_num_codebooks;
    int n_frames = (int) speech_codes.size() / n_codebooks;
    if (n_frames <= 0) {
        result.error_msg = "No speech codes generated";
        return result;
    }
    if ((int32_t) speech_codes.size() != n_frames * n_codebooks) {
        result.error_msg = "[generate] Invalid generated codes shape; expected T x codec_num_codebooks";
        return result;
    }
    const int32_t generated_code_upper_bound =
        (profile_.suppress_to > profile_.codec_id_end) ? profile_.suppress_to : profile_.codec_id_end;
    for (int32_t c : speech_codes) {
        if (c < profile_.codec_id_start || c >= generated_code_upper_bound) {
            result.error_msg =
                "[generate] Invalid generated code id out of runtime codec/special range from model_profile.json";
            return result;
        }
    }
    if (params.print_progress) {
        LOG_INFO("Speech codes generated: %d frames x %d codebooks", n_frames, n_codebooks);
    }

    int64_t t_decode = get_time_ms();
    if (!decoder_loaded_) {
        if (!decoder_.load_model(decoder_onnx_path_, params.n_threads)) {
            result.error_msg = "Failed to load decoder ONNX: " + decoder_.get_error();
            return result;
        }
        decoder_loaded_ = true;
        LOG_DEBUG("  Decoder providers: %s", decoder_.provider_summary().c_str());
    }
    result.ort_provider_decoder = decoder_.provider_summary();
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

    if (params.print_timing && false) { // Handled by main.cpp in a prettier way
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
    LOG_ERROR("ERROR: No data chunk found");
    return false;
}

bool save_audio_file(const std::string & path, const std::vector<float> & samples, int sample_rate) {
    try {
        fs::path p(path);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }
    } catch (...) {
        // Continue and let fopen report a concrete error if directory creation fails.
    }
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        LOG_ERROR("ERROR: Cannot create WAV file: %s", path.c_str());
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
