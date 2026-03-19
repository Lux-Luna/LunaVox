#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qwen3_tts {

struct npy_matrix {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<float> data;

    bool empty() const { return data.empty() || rows <= 0 || cols <= 0; }
    const float * row(int32_t r) const {
        if (r < 0 || r >= rows) return nullptr;
        return data.data() + (size_t) r * (size_t) cols;
    }
};

class AssetsManager {
public:
    bool load(const std::string & model_dir);
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
    bool load_npy_f32_2d(const std::string & path, npy_matrix & out);
    bool parse_npy_header(const std::string & path, std::string & descr, std::vector<int64_t> & shape, size_t & data_offset);
    static float fp16_to_f32(uint16_t h);

    bool loaded_ = false;
    bool has_proj_ = false;
    std::string error_msg_;
    std::string model_dir_;

    npy_matrix text_table_;
    npy_matrix proj_weight_;
    npy_matrix proj_bias_;
    npy_matrix codec_table_[16];
};

} // namespace qwen3_tts

