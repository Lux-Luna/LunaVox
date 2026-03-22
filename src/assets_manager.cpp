#include "assets_manager.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <numeric>

namespace qwen3_tts {

namespace {

static std::string trim_copy(const std::string & s) {
    size_t l = 0;
    while (l < s.size() && std::isspace((unsigned char) s[l])) ++l;
    size_t r = s.size();
    while (r > l && std::isspace((unsigned char) s[r - 1])) --r;
    return s.substr(l, r - l);
}

static bool file_exists(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

} // namespace

bool AssetsManager::load(const std::string & model_dir) {
    clear();
    model_dir_ = model_dir;

    const std::string emb_dir = model_dir + "/embeddings";
    const std::string text_path = emb_dir + "/text_embedding_projected.npy";
    if (!load_npy_f32_2d(text_path, text_table_)) {
        error_msg_ = "Failed to load text embedding table: " + text_path + " : " + error_msg_;
        return false;
    }

    for (int i = 0; i < 16; ++i) {
        const std::string path = emb_dir + "/codec_embedding_" + std::to_string(i) + ".npy";
        if (!load_npy_f32_2d(path, codec_table_[i])) {
            error_msg_ = "Failed to load codec embedding table: " + path + " : " + error_msg_;
            return false;
        }
    }

    const std::string proj_w = emb_dir + "/proj_weight.npy";
    const std::string proj_b = emb_dir + "/proj_bias.npy";
    if (file_exists(proj_w) && file_exists(proj_b)) {
        if (!load_npy_f32_2d(proj_w, proj_weight_)) {
            error_msg_ = "Failed to load projection weight: " + error_msg_;
            return false;
        }
        if (!load_npy_f32_2d(proj_b, proj_bias_)) {
            error_msg_ = "Failed to load projection bias: " + error_msg_;
            return false;
        }
        if (proj_bias_.rows != 1 || proj_bias_.cols != proj_weight_.rows) {
            error_msg_ = "Projection bias shape mismatch";
            return false;
        }
        has_proj_ = true;
    }

    loaded_ = true;
    return true;
}

void AssetsManager::clear() {
    loaded_ = false;
    has_proj_ = false;
    error_msg_.clear();
    model_dir_.clear();
    text_table_ = {};
    proj_weight_ = {};
    proj_bias_ = {};
    for (auto & t : codec_table_) {
        t = {};
    }
}

const float * AssetsManager::codec_row(int32_t q, int32_t code) const {
    if (q < 0 || q >= 16) return nullptr;
    return codec_table_[q].row(code);
}

bool AssetsManager::project_to_predictor(const float * in_hidden, std::vector<float> & out) const {
    if (!in_hidden) return false;
    if (!has_proj_) {
        out.assign(in_hidden, in_hidden + hidden_dim());
        return true;
    }
    if (proj_weight_.cols != hidden_dim()) {
        return false;
    }
    out.assign((size_t) proj_weight_.rows, 0.0f);
    for (int32_t r = 0; r < proj_weight_.rows; ++r) {
        const float * w = proj_weight_.row(r);
        float v = proj_bias_.data[(size_t) r];
        for (int32_t c = 0; c < proj_weight_.cols; ++c) {
            v += w[c] * in_hidden[c];
        }
        out[(size_t) r] = v;
    }
    return true;
}

bool AssetsManager::codec_row_predictor(int32_t q, int32_t code, std::vector<float> & out) const {
    const float * raw = codec_row(q, code);
    if (!raw) return false;
    return project_to_predictor(raw, out);
}

bool AssetsManager::parse_npy_header(const std::string & path, std::string & descr, std::vector<int64_t> & shape, size_t & data_offset) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        error_msg_ = "Cannot open npy: " + path;
        return false;
    }

    char magic[6] = {};
    fin.read(magic, 6);
    if (!fin || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        error_msg_ = "Invalid npy magic";
        return false;
    }

    uint8_t ver_major = 0, ver_minor = 0;
    fin.read((char *) &ver_major, 1);
    fin.read((char *) &ver_minor, 1);
    if (!fin) {
        error_msg_ = "Invalid npy version bytes";
        return false;
    }

    uint32_t header_len = 0;
    if (ver_major <= 1) {
        uint16_t h16 = 0;
        fin.read((char *) &h16, 2);
        header_len = h16;
    } else {
        fin.read((char *) &header_len, 4);
    }
    if (!fin || header_len == 0) {
        error_msg_ = "Invalid npy header length";
        return false;
    }

    std::string header(header_len, '\0');
    fin.read(&header[0], header_len);
    if (!fin) {
        error_msg_ = "Failed to read npy header";
        return false;
    }

    auto find_quoted = [&](const std::string & key, std::string & out_val) -> bool {
        size_t k = header.find("'" + key + "'");
        if (k == std::string::npos) {
            k = header.find("\"" + key + "\"");
        }
        if (k == std::string::npos) {
            return false;
        }

        size_t colon = header.find(':', k);
        if (colon == std::string::npos) return false;

        size_t q1 = header.find_first_of("'\"", colon + 1);
        if (q1 == std::string::npos) return false;

        const char quote = header[q1];
        size_t q2 = header.find(quote, q1 + 1);
        if (q2 == std::string::npos) return false;

        out_val = trim_copy(header.substr(q1 + 1, q2 - q1 - 1));
        return true;
    };

    if (!find_quoted("descr", descr)) {
        error_msg_ = "npy header missing descr";
        return false;
    }

    size_t s_key = header.find("shape");
    if (s_key == std::string::npos) {
        error_msg_ = "npy header missing shape";
        return false;
    }
    size_t p1 = header.find('(', s_key);
    size_t p2 = header.find(')', p1);
    if (p1 == std::string::npos || p2 == std::string::npos || p2 <= p1) {
        error_msg_ = "npy header invalid shape tuple";
        return false;
    }

    std::string inside = header.substr(p1 + 1, p2 - p1 - 1);
    shape.clear();
    size_t st = 0;
    while (st < inside.size()) {
        size_t comma = inside.find(',', st);
        std::string part = (comma == std::string::npos) ? inside.substr(st) : inside.substr(st, comma - st);
        part = trim_copy(part);
        if (!part.empty()) {
            shape.push_back(std::stoll(part));
        }
        if (comma == std::string::npos) break;
        st = comma + 1;
    }

    data_offset = (size_t) fin.tellg();
    return true;
}

float AssetsManager::fp16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x03FF;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            uint32_t e = 127 - 15 + 1;
            uint32_t m = mant;
            while ((m & 0x0400) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x03FF;
            bits = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

bool AssetsManager::load_npy_f32_2d(const std::string & path, npy_matrix & out) {
    out = {};
    std::string descr;
    std::vector<int64_t> shape;
    size_t data_offset = 0;
    if (!parse_npy_header(path, descr, shape, data_offset)) {
        return false;
    }
    if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0) {
        error_msg_ = "Only 2D npy arrays are supported";
        return false;
    }
    out.rows = (int32_t) shape[0];
    out.cols = (int32_t) shape[1];
    const size_t elem_count = (size_t) out.rows * (size_t) out.cols;
    out.data.resize(elem_count);

    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        error_msg_ = "Cannot reopen npy for payload read";
        return false;
    }
    fin.seekg((std::streamoff) data_offset, std::ios::beg);

    if (descr == "<f4" || descr == "|f4" || descr == "f4") {
        fin.read((char *) out.data.data(), (std::streamsize) (elem_count * sizeof(float)));
        if (!fin) {
            error_msg_ = "Failed reading float32 payload";
            return false;
        }
        return true;
    }
    if (descr == "<f2" || descr == "|f2" || descr == "f2") {
        std::vector<uint16_t> tmp(elem_count);
        fin.read((char *) tmp.data(), (std::streamsize) (elem_count * sizeof(uint16_t)));
        if (!fin) {
            error_msg_ = "Failed reading float16 payload";
            return false;
        }
        for (size_t i = 0; i < elem_count; ++i) {
            out.data[i] = fp16_to_f32(tmp[i]);
        }
        return true;
    }

    error_msg_ = "Unsupported npy dtype: " + descr;
    return false;
}

} // namespace qwen3_tts
