#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace qwen3_tts {

inline std::string profile_to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return s;
}

struct RuntimeModelProfile {
    int32_t version = 1;

    // Canonical runtime model family: "base", "custom", "design".
    std::string model_type;
    // Human-readable size label from source metadata, e.g. "1.7b", "0.6b".
    std::string model_size;

    // Whether this model supports user instruct control.
    bool instruct_support = false;

    // Talker runtime limits / token semantics.
    int32_t talker_n_ctx = 0;
    int32_t predictor_n_ctx = 0;
    int32_t codec_id_start = 0;
    int32_t codec_id_end = 2048; // exclusive
    int32_t codec_num_codebooks = 0;
    int32_t predictor_vocab_size = 0;

    int32_t codec_pad_id = 2148;
    int32_t codec_bos_id = 2149;
    int32_t codec_eos_id = 2150;
    int32_t codec_think_id = 2154;
    int32_t codec_nothink_id = 2155;
    int32_t codec_think_bos_id = 2156;
    int32_t codec_think_eos_id = 2157;

    int32_t tts_pad_id = 151671;
    int32_t tts_bos_id = 151672;
    int32_t tts_eos_id = 151673;

    int32_t min_new_tokens = 2;
    int32_t suppress_from = 2048;
    int32_t suppress_to = 3072; // exclusive

    // Generation defaults from official generation_config semantics.
    int32_t default_max_new_tokens = 2048;
    float default_temperature = 0.9f;
    float default_top_p = 1.0f;
    int32_t default_top_k = 50;
    float default_repetition_penalty = 1.05f;
    bool default_predictor_do_sample = true;
    float default_predictor_temperature = 0.9f;
    float default_predictor_top_p = 1.0f;
    int32_t default_predictor_top_k = 50;

    // Canonical maps (all keys lower-cased).
    std::unordered_map<std::string, int32_t> language_map;
    std::unordered_map<std::string, int32_t> speaker_map;
    std::unordered_map<std::string, std::string> speaker_dialect_map;

    bool is_valid(std::string * reason = nullptr) const {
        auto fail = [&](const char * msg) {
            if (reason) *reason = msg;
            return false;
        };

        if (model_type != "base" && model_type != "custom" && model_type != "design") {
            return fail("model_type must be one of: base/custom/design");
        }
        if (talker_n_ctx <= 0) {
            return fail("talker_n_ctx must be positive");
        }
        if (predictor_n_ctx <= 0) {
            return fail("predictor_n_ctx must be positive");
        }
        if (codec_id_end <= codec_id_start) {
            return fail("codec id range is invalid");
        }
        if (codec_num_codebooks <= 1) {
            return fail("codec_num_codebooks must be >= 2");
        }
        // Current runtime/decoder path is explicitly 16-codebook only.
        if (codec_num_codebooks != 16) {
            return fail("codec_num_codebooks must be 16 for current runtime");
        }
        if (predictor_vocab_size <= 0) {
            return fail("predictor_vocab_size must be positive");
        }
        const int32_t codebook_vocab_size = codec_id_end - codec_id_start;
        if ((codec_num_codebooks - 1) <= 0 || predictor_vocab_size % (codec_num_codebooks - 1) != 0) {
            return fail("predictor_vocab_size must be divisible by (codec_num_codebooks-1)");
        }
        if (predictor_vocab_size / (codec_num_codebooks - 1) != codebook_vocab_size) {
            return fail("predictor_vocab_size/codebook count mismatch with codec id range");
        }
        if (min_new_tokens < 0) {
            return fail("min_new_tokens must be non-negative");
        }
        if (default_max_new_tokens <= 0) {
            return fail("default_max_new_tokens must be positive");
        }
        if (default_temperature < 0.0f || default_predictor_temperature < 0.0f) {
            return fail("default temperature values must be non-negative");
        }
        if (default_top_p <= 0.0f || default_top_p > 1.0f ||
            default_predictor_top_p <= 0.0f || default_predictor_top_p > 1.0f) {
            return fail("default top_p values must be in (0, 1]");
        }
        if (default_top_k < 0 || default_predictor_top_k < 0) {
            return fail("default top_k values must be non-negative");
        }
        if (default_repetition_penalty <= 0.0f) {
            return fail("default_repetition_penalty must be positive");
        }
        return true;
    }

    int32_t resolve_language_id(const std::string & name_or_alias) const {
        if (name_or_alias.empty()) {
            return -1;
        }
        const std::string k = profile_to_lower_ascii(name_or_alias);
        if (k == "none" || k == "auto" || k == "unspecified") {
            return -1;
        }
        auto it = language_map.find(k);
        if (it == language_map.end()) {
            return -2;
        }
        return it->second;
    }

    int32_t resolve_speaker_id(const std::string & speaker_name) const {
        if (speaker_name.empty()) {
            return -1;
        }
        const std::string k = profile_to_lower_ascii(speaker_name);
        auto it = speaker_map.find(k);
        if (it == speaker_map.end()) {
            return -1;
        }
        return it->second;
    }

    std::string speaker_dialect(const std::string & speaker_name) const {
        if (speaker_name.empty()) {
            return std::string();
        }
        const std::string k = profile_to_lower_ascii(speaker_name);
        auto it = speaker_dialect_map.find(k);
        if (it == speaker_dialect_map.end()) {
            return std::string();
        }
        return it->second;
    }

    std::vector<std::string> speaker_names_sorted() const {
        std::vector<std::string> names;
        names.reserve(speaker_map.size());
        for (const auto & kv : speaker_map) {
            names.push_back(kv.first);
        }
        std::sort(names.begin(), names.end());
        return names;
    }
};

} // namespace qwen3_tts
