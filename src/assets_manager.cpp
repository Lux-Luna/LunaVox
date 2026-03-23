#include "assets_manager.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace qwen3_tts {

namespace {

static constexpr intptr_t kInvalidNativeHandle = static_cast<intptr_t>(-1);

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

static float fp16_to_f32(uint16_t h) {
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

#ifndef _WIN32
static std::string errno_to_string(int err_no) {
    char buf[128] = {};
#if ((_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600) && !_GNU_SOURCE) || defined(__APPLE__)
    if (strerror_r(err_no, buf, sizeof(buf)) == 0) {
        return std::string(buf);
    }
    return "errno=" + std::to_string(err_no);
#else
    return std::string(strerror_r(err_no, buf, sizeof(buf)));
#endif
}
#endif

} // namespace

npy_matrix::~npy_matrix() {
    clear();
}

bool npy_matrix::empty() const {
    if (rows <= 0 || cols <= 0) {
        return true;
    }
    switch (storage) {
        case npy_storage_kind::mmap_f32:
            return mapped_f32 == nullptr;
        case npy_storage_kind::mmap_f16:
            return mapped_f16_bytes == nullptr;
        case npy_storage_kind::owned_f32:
            return owned_f32.empty();
        default:
            return true;
    }
}

const float * npy_matrix::row(int32_t r) const {
    if (r < 0 || r >= rows || cols <= 0) {
        return nullptr;
    }

    switch (storage) {
        case npy_storage_kind::mmap_f32: {
            if (!mapped_f32) {
                return nullptr;
            }
            return mapped_f32 + (size_t) r * (size_t) cols;
        }
        case npy_storage_kind::owned_f32: {
            if (owned_f32.empty()) {
                return nullptr;
            }
            return owned_f32.data() + (size_t) r * (size_t) cols;
        }
        case npy_storage_kind::mmap_f16: {
            if (!mapped_f16_bytes) {
                return nullptr;
            }
            std::lock_guard<std::mutex> lock(f16_cache_mu);
            auto it = f16_row_cache.find(r);
            if (it == f16_row_cache.end()) {
                std::vector<float> decoded((size_t) cols, 0.0f);
                const uint8_t * src = mapped_f16_bytes + (size_t) r * (size_t) cols * sizeof(uint16_t);
                for (int32_t i = 0; i < cols; ++i) {
                    uint16_t h = 0;
                    std::memcpy(&h, src + (size_t) i * sizeof(uint16_t), sizeof(uint16_t));
                    decoded[(size_t) i] = fp16_to_f32(h);
                }
                it = f16_row_cache.emplace(r, std::move(decoded)).first;
            }
            return it->second.data();
        }
        default:
            return nullptr;
    }
}

void npy_matrix::clear() {
#ifdef _WIN32
    if (mapped_base) {
        UnmapViewOfFile(mapped_base);
    }
    if (mapping_handle != kInvalidNativeHandle) {
        CloseHandle(reinterpret_cast<HANDLE>(mapping_handle));
    }
    if (file_handle != kInvalidNativeHandle && file_handle != reinterpret_cast<intptr_t>(INVALID_HANDLE_VALUE)) {
        CloseHandle(reinterpret_cast<HANDLE>(file_handle));
    }
#else
    if (mapped_base && mapped_bytes > 0) {
        munmap(const_cast<uint8_t *>(mapped_base), mapped_bytes);
    }
    if (file_handle >= 0) {
        close((int) file_handle);
    }
#endif

    rows = 0;
    cols = 0;
    storage = npy_storage_kind::none;
    owned_f32.clear();
    mapped_base = nullptr;
    mapped_bytes = 0;
    mapped_f32 = nullptr;
    mapped_f16_bytes = nullptr;
    file_handle = kInvalidNativeHandle;
    mapping_handle = kInvalidNativeHandle;

    {
        std::lock_guard<std::mutex> lock(f16_cache_mu);
        f16_row_cache.clear();
    }
}

bool AssetsManager::load(const std::string & model_dir) {
    clear();
    model_dir_ = model_dir;

    const std::string emb_dir = model_dir + "/embeddings";
    const std::string text_path = emb_dir + "/text_embedding_projected.npy";
    if (!load_npy_mmap_2d(text_path, text_table_)) {
        error_msg_ = "Failed to load text embedding table: " + text_path + " : " + error_msg_;
        return false;
    }

    for (int i = 0; i < 16; ++i) {
        const std::string path = emb_dir + "/codec_embedding_" + std::to_string(i) + ".npy";
        if (!load_npy_mmap_2d(path, codec_table_[i])) {
            error_msg_ = "Failed to load codec embedding table: " + path + " : " + error_msg_;
            return false;
        }
    }

    const std::string proj_w = emb_dir + "/proj_weight.npy";
    const std::string proj_b = emb_dir + "/proj_bias.npy";
    if (file_exists(proj_w) && file_exists(proj_b)) {
        if (!load_npy_mmap_2d(proj_w, proj_weight_)) {
            error_msg_ = "Failed to load projection weight: " + error_msg_;
            return false;
        }
        if (!load_npy_mmap_2d(proj_b, proj_bias_)) {
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
    text_table_.clear();
    proj_weight_.clear();
    proj_bias_.clear();
    for (auto & t : codec_table_) {
        t.clear();
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
    const float * bias = proj_bias_.row(0);
    if (!bias) {
        return false;
    }
    out.assign((size_t) proj_weight_.rows, 0.0f);
    for (int32_t r = 0; r < proj_weight_.rows; ++r) {
        const float * w = proj_weight_.row(r);
        if (!w) {
            return false;
        }
        float v = bias[(size_t) r];
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

bool AssetsManager::map_file_readonly(const std::string & path, npy_matrix & out, size_t & file_size) {
    file_size = 0;

#ifdef _WIN32
    HANDLE file = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        error_msg_ = "CreateFile failed";
        return false;
    }

    LARGE_INTEGER size_li = {};
    if (!GetFileSizeEx(file, &size_li) || size_li.QuadPart <= 0) {
        CloseHandle(file);
        error_msg_ = "GetFileSizeEx failed";
        return false;
    }
    if ((uint64_t) size_li.QuadPart > (uint64_t) std::numeric_limits<size_t>::max()) {
        CloseHandle(file);
        error_msg_ = "File too large for mmap on this platform";
        return false;
    }
    file_size = (size_t) size_li.QuadPart;

    HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping) {
        CloseHandle(file);
        error_msg_ = "CreateFileMapping failed";
        return false;
    }

    void * view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        CloseHandle(mapping);
        CloseHandle(file);
        error_msg_ = "MapViewOfFile failed";
        return false;
    }

    out.file_handle = reinterpret_cast<intptr_t>(file);
    out.mapping_handle = reinterpret_cast<intptr_t>(mapping);
    out.mapped_base = reinterpret_cast<const uint8_t *>(view);
    out.mapped_bytes = file_size;
    return true;
#else
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        error_msg_ = "open failed: " + errno_to_string(errno);
        return false;
    }

    struct stat st = {};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        error_msg_ = "fstat failed: " + errno_to_string(errno);
        return false;
    }
    if ((uint64_t) st.st_size > (uint64_t) std::numeric_limits<size_t>::max()) {
        close(fd);
        error_msg_ = "File too large for mmap on this platform";
        return false;
    }
    file_size = (size_t) st.st_size;

    void * view = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (view == MAP_FAILED) {
        close(fd);
        error_msg_ = "mmap failed: " + errno_to_string(errno);
        return false;
    }

    out.file_handle = (intptr_t) fd;
    out.mapping_handle = kInvalidNativeHandle;
    out.mapped_base = reinterpret_cast<const uint8_t *>(view);
    out.mapped_bytes = file_size;
    return true;
#endif
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

    uint8_t ver_major = 0;
    uint8_t ver_minor = 0;
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
            try {
                shape.push_back(std::stoll(part));
            } catch (...) {
                error_msg_ = "npy header has invalid shape entry: " + part;
                return false;
            }
        }
        if (comma == std::string::npos) break;
        st = comma + 1;
    }

    const std::streamoff off = fin.tellg();
    if (off < 0) {
        error_msg_ = "npy payload offset is invalid";
        return false;
    }
    data_offset = (size_t) off;
    return true;
}

bool AssetsManager::load_npy_mmap_2d(const std::string & path, npy_matrix & out) {
    out.clear();

    std::string descr;
    std::vector<int64_t> shape;
    size_t data_offset = 0;
    if (!parse_npy_header(path, descr, shape, data_offset)) {
        return false;
    }

    int32_t rows = 0;
    int32_t cols = 0;
    if (shape.size() == 1 && shape[0] > 0) {
        rows = 1;
        cols = (int32_t) shape[0];
    } else if (shape.size() == 2 && shape[0] > 0 && shape[1] > 0) {
        rows = (int32_t) shape[0];
        cols = (int32_t) shape[1];
    } else {
        error_msg_ = "Only 1D or 2D npy arrays are supported";
        return false;
    }

    const bool is_f32 = (descr == "<f4" || descr == "|f4" || descr == "f4");
    const bool is_f16 = (descr == "<f2" || descr == "|f2" || descr == "f2");
    if (!is_f32 && !is_f16) {
        error_msg_ = "Unsupported npy dtype for mmap loader: " + descr;
        return false;
    }

    size_t file_size = 0;
    if (!map_file_readonly(path, out, file_size)) {
        error_msg_ = "mmap failed for " + path + ": " + error_msg_;
        out.clear();
        return false;
    }

    if (data_offset >= file_size) {
        error_msg_ = "npy payload offset out of file range";
        out.clear();
        return false;
    }

    const size_t elem_size = is_f32 ? sizeof(float) : sizeof(uint16_t);
    const size_t elem_count = (size_t) rows * (size_t) cols;
    if (elem_count == 0 || elem_count > (std::numeric_limits<size_t>::max() / elem_size)) {
        error_msg_ = "npy element count overflow";
        out.clear();
        return false;
    }
    const size_t payload_bytes = elem_count * elem_size;
    if (payload_bytes > file_size - data_offset) {
        error_msg_ = "npy payload truncated";
        out.clear();
        return false;
    }

    const uint8_t * payload = out.mapped_base + data_offset;
    out.rows = rows;
    out.cols = cols;

    if (is_f32) {
        const uintptr_t addr = reinterpret_cast<uintptr_t>(payload);
        if ((addr % alignof(float)) == 0) {
            out.storage = npy_storage_kind::mmap_f32;
            out.mapped_f32 = reinterpret_cast<const float *>(payload);
            return true;
        }

        // Keep compatibility for rare unaligned payloads.
        out.owned_f32.resize(elem_count);
        std::memcpy(out.owned_f32.data(), payload, payload_bytes);
        out.storage = npy_storage_kind::owned_f32;
        return true;
    }

    out.storage = npy_storage_kind::mmap_f16;
    out.mapped_f16_bytes = payload;
    return true;
}

} // namespace qwen3_tts
