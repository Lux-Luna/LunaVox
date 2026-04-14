#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "platform_utils.h"

namespace lunavox {

enum class npy_storage_kind : uint8_t {
    none = 0,
    mmap_f32 = 1,
    mmap_f16 = 2,
    owned_f32 = 3,
};

struct npy_matrix {
    int32_t rows = 0;
    int32_t cols = 0;
    npy_storage_kind storage = npy_storage_kind::none;

    // Owned path (small buffers or intermediate compatibility payloads).
    std::vector<float> owned_f32;

    // Underlying mapped file; lifetime owned by this matrix.
    platform::MmapFile mapping;

    // Decoded payload pointers into `mapping.data` (or owned_f32).
    const float *   mapped_f32 = nullptr;
    const uint8_t * mapped_f16_bytes = nullptr;

    // Per-row decode cache for mmap_f16 (lazy decode, no full table expansion).
    mutable std::unordered_map<int32_t, std::vector<float>> f16_row_cache;
    mutable std::mutex f16_cache_mu;

    npy_matrix() = default;
    ~npy_matrix();
    npy_matrix(const npy_matrix &) = delete;
    npy_matrix & operator=(const npy_matrix &) = delete;
    npy_matrix(npy_matrix &&) = delete;
    npy_matrix & operator=(npy_matrix &&) = delete;

    bool empty() const;
    const float * row(int32_t r) const;
    void clear();
};

class AssetsManager {
public:
    bool load(const std::string & model_dir, int32_t codec_num_codebooks);
    void clear();

    bool is_loaded() const { return loaded_; }
    const std::string & get_error() const { return error_msg_; }

    int32_t hidden_dim() const { return text_table_.cols; }
    int32_t predictor_dim() const { return has_proj_ ? proj_weight_.rows : hidden_dim(); }

    const npy_matrix & text_table() const { return text_table_; }
    const float * tts_pad() const { return text_table_.row(151671); }
    const float * text_row(int32_t token_id) const { return text_table_.row(token_id); }
    const float * codec_row(int32_t q, int32_t code) const;

    bool has_projection() const { return has_proj_; }
    bool project_to_predictor(const float * in_hidden, std::vector<float> & out) const;
    bool codec_row_predictor(int32_t q, int32_t code, std::vector<float> & out) const;

private:
    bool load_npy_mmap_2d(const std::string & path, npy_matrix & out);
    bool parse_npy_header(const std::string & path, std::string & descr, std::vector<int64_t> & shape, size_t & data_offset);

    bool loaded_ = false;
    bool has_proj_ = false;
    std::string error_msg_;
    std::string model_dir_;

    npy_matrix text_table_;
    npy_matrix proj_weight_;
    npy_matrix proj_bias_;
    int32_t codec_num_codebooks_ = 0;
    npy_matrix codec_table_[16];
};

} // namespace lunavox

