#include "gguf_loader.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <algorithm>
#include <cstdlib>
#include <cctype>

#if !defined(_WIN32)
#include <sys/types.h>
#endif

namespace qwen3_tts {

namespace {
bool seek_file_absolute(FILE * f, uint64_t offset) {
#if defined(_WIN32)
    return _fseeki64(f, static_cast<__int64>(offset), SEEK_SET) == 0;
#else
    return fseeko(f, static_cast<off_t>(offset), SEEK_SET) == 0;
#endif
}

static std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return s;
}

static std::vector<enum ggml_backend_dev_type> get_backend_try_order_from_env() {
    const char * env = std::getenv("QWEN3_TTS_BACKEND");
    if (!env || env[0] == '\0') {
        // Default to dGPU first, then iGPU/other accelerators, then CPU fallback.
        return {
            GGML_BACKEND_DEVICE_TYPE_GPU,
            GGML_BACKEND_DEVICE_TYPE_IGPU,
            GGML_BACKEND_DEVICE_TYPE_ACCEL,
            GGML_BACKEND_DEVICE_TYPE_CPU,
        };
    }

    const std::string pref = to_lower_ascii(env);
    if (pref == "auto") {
        return {
            GGML_BACKEND_DEVICE_TYPE_GPU,
            GGML_BACKEND_DEVICE_TYPE_IGPU,
            GGML_BACKEND_DEVICE_TYPE_ACCEL,
            GGML_BACKEND_DEVICE_TYPE_CPU,
        };
    }
    if (pref == "gpu" || pref == "cuda" || pref == "vulkan" || pref == "metal") {
        return {
            GGML_BACKEND_DEVICE_TYPE_GPU,
            GGML_BACKEND_DEVICE_TYPE_IGPU,
            GGML_BACKEND_DEVICE_TYPE_ACCEL,
            GGML_BACKEND_DEVICE_TYPE_CPU,
        };
    }
    if (pref == "igpu") {
        return {
            GGML_BACKEND_DEVICE_TYPE_IGPU,
            GGML_BACKEND_DEVICE_TYPE_GPU,
            GGML_BACKEND_DEVICE_TYPE_ACCEL,
            GGML_BACKEND_DEVICE_TYPE_CPU,
        };
    }
    if (pref == "accel") {
        return {
            GGML_BACKEND_DEVICE_TYPE_ACCEL,
            GGML_BACKEND_DEVICE_TYPE_GPU,
            GGML_BACKEND_DEVICE_TYPE_IGPU,
            GGML_BACKEND_DEVICE_TYPE_CPU,
        };
    }
    if (pref == "cpu") {
        return {
            GGML_BACKEND_DEVICE_TYPE_CPU,
        };
    }

    // Unknown override: keep a safe default.
    return {
        GGML_BACKEND_DEVICE_TYPE_GPU,
        GGML_BACKEND_DEVICE_TYPE_IGPU,
        GGML_BACKEND_DEVICE_TYPE_ACCEL,
        GGML_BACKEND_DEVICE_TYPE_CPU,
    };
}

static ggml_backend_t init_backend_from_order(enum ggml_backend_dev_type * selected_type = nullptr) {
    const auto order = get_backend_try_order_from_env();
    for (auto type : order) {
        ggml_backend_t backend = ggml_backend_init_by_type(type, nullptr);
        if (backend) {
            if (selected_type) {
                *selected_type = type;
            }
            return backend;
        }
    }

    if (selected_type) {
        *selected_type = GGML_BACKEND_DEVICE_TYPE_CPU;
    }
    return nullptr;
}
}

GGUFLoader::GGUFLoader() = default;

GGUFLoader::~GGUFLoader() {
    close();
}

ggml_backend_t init_preferred_backend(const char * component_name, std::string * error_msg) {
    if (error_msg) error_msg->clear();

    ggml_backend_t backend = init_backend_from_order();

    if (!backend && error_msg) {
        const char * name = component_name ? component_name : "component";
        *error_msg = "Failed to initialize backend (respecting QWEN3_TTS_BACKEND) for " + std::string(name);
    }

    return backend;
}

void release_preferred_backend(ggml_backend_t backend) {
    if (backend) {
        ggml_backend_free(backend);
    }
}

enum ggml_backend_dev_type detect_preferred_backend_type() {
    enum ggml_backend_dev_type selected = GGML_BACKEND_DEVICE_TYPE_CPU;
    ggml_backend_t backend = init_backend_from_order(&selected);
    if (backend) {
        ggml_backend_free(backend);
    }
    return selected;
}

void apply_backend_n_threads(ggml_backend_t backend, int32_t n_threads) {
    if (!backend || n_threads <= 0) {
        return;
    }

    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (!device) {
        return;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if (reg) {
        auto set_threads = (ggml_backend_set_n_threads_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_threads) {
            set_threads(backend, n_threads);
            return;
        }
    }

    // CPU fallback: some builds may expose CPU setter only through ggml-cpu.h.
    if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }
}

bool GGUFLoader::open(const std::string & path) {
    close();  // Close any previously opened file
    
    file_path_ = path;
    
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx_,
    };
    
    ctx_ = gguf_init_from_file(path.c_str(), params);
    if (!ctx_) {
        error_msg_ = "Failed to open GGUF file: " + path;
        return false;
    }
    
    return true;
}

void GGUFLoader::close() {
    if (ctx_) {
        gguf_free(ctx_);
        ctx_ = nullptr;
    }
    if (meta_ctx_) {
        ggml_free(meta_ctx_);
        meta_ctx_ = nullptr;
    }
    file_path_.clear();
}

int64_t GGUFLoader::get_n_tensors() const {
    if (!ctx_) return 0;
    return gguf_get_n_tensors(ctx_);
}

const char * GGUFLoader::get_tensor_name(int64_t idx) const {
    if (!ctx_) return nullptr;
    return gguf_get_tensor_name(ctx_, idx);
}

enum ggml_type GGUFLoader::get_tensor_type(int64_t idx) const {
    if (!ctx_) return GGML_TYPE_F32;
    return gguf_get_tensor_type(ctx_, idx);
}

size_t GGUFLoader::get_tensor_offset(int64_t idx) const {
    if (!ctx_) return 0;
    return gguf_get_tensor_offset(ctx_, idx);
}

size_t GGUFLoader::get_tensor_size(int64_t idx) const {
    if (!ctx_) return 0;
    return gguf_get_tensor_size(ctx_, idx);
}

int32_t GGUFLoader::get_u32(const char * key, int32_t default_val) const {
    if (!ctx_) return default_val;
    int64_t idx = gguf_find_key(ctx_, key);
    if (idx < 0) return default_val;
    return (int32_t)gguf_get_val_u32(ctx_, idx);
}

float GGUFLoader::get_f32(const char * key, float default_val) const {
    if (!ctx_) return default_val;
    int64_t idx = gguf_find_key(ctx_, key);
    if (idx < 0) return default_val;
    return gguf_get_val_f32(ctx_, idx);
}

size_t GGUFLoader::get_data_offset() const {
    if (!ctx_) return 0;
    return gguf_get_data_offset(ctx_);
}

bool load_tensor_data_from_file(
    const std::string & path,
    struct gguf_context * ctx,
    struct ggml_context * model_ctx,
    const std::map<std::string, struct ggml_tensor *> & tensors,
    ggml_backend_buffer_t & buffer,
    std::string & error_msg,
    enum ggml_backend_dev_type preferred_backend_type
) {
    ggml_backend_t backend = ggml_backend_init_by_type(preferred_backend_type, nullptr);
    if (!backend && preferred_backend_type != GGML_BACKEND_DEVICE_TYPE_CPU) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (!backend) {
        error_msg = "Failed to initialize backend for GGUF tensor loader";
        return false;
    }
    
    // Allocate buffer for all tensors
    buffer = ggml_backend_alloc_ctx_tensors(model_ctx, backend);
    if (!buffer) {
        error_msg = "Failed to allocate tensor buffer";
        ggml_backend_free(backend);
        return false;
    }
    
    // Open file for reading tensor data
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        error_msg = "Failed to open file for reading: " + path;
        ggml_backend_free(backend);
        return false;
    }
    
    const size_t data_offset = gguf_get_data_offset(ctx);
    const uint64_t data_offset64 = static_cast<uint64_t>(data_offset);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    std::vector<uint8_t> read_buf;
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);
        
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            continue;  // Skip tensors not in our map
        }
        
        struct ggml_tensor * tensor = it->second;
        size_t nbytes = ggml_nbytes(tensor);
        
        read_buf.resize(nbytes);
        
        const uint64_t offset64 = static_cast<uint64_t>(offset);
        if (offset64 > (std::numeric_limits<uint64_t>::max)() - data_offset64) {
            error_msg = "Tensor offset overflow: " + std::string(name);
            fclose(f);
            ggml_backend_free(backend);
            return false;
        }
        const uint64_t absolute_offset = data_offset64 + offset64;

        if (!seek_file_absolute(f, absolute_offset)) {
            error_msg = "Failed to seek to tensor data: " + std::string(name);
            fclose(f);
            ggml_backend_free(backend);
            return false;
        }
        
        if (fread(read_buf.data(), 1, nbytes, f) != nbytes) {
            error_msg = "Failed to read tensor data: " + std::string(name);
            fclose(f);
            ggml_backend_free(backend);
            return false;
        }
        
        ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);
    }
    
    fclose(f);
    ggml_backend_free(backend);
    
    return true;
}

void free_ggml_resources(struct ggml_context * ctx, ggml_backend_buffer_t buffer) {
    if (buffer) {
        ggml_backend_buffer_free(buffer);
    }
    if (ctx) {
        ggml_free(ctx);
    }
}

} // namespace qwen3_tts
