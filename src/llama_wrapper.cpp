#include "llama_wrapper.h"
#include "logger.h"

#include <cstdio>
#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace qwen3_tts {

namespace {

template <typename T>
bool load_fn(void * handle, const char * name, T & fn_ptr, std::string & err) {
#ifdef _WIN32
    auto sym = GetProcAddress((HMODULE) handle, name);
#else
    auto sym = dlsym(handle, name);
#endif
    if (!sym) {
        err = std::string("Missing symbol in llama runtime: ") + name;
        return false;
    }
    fn_ptr = reinterpret_cast<T>(sym);
    return true;
}

template <typename T>
bool try_load_fn(void * handle, const char * name, T & fn_ptr) {
    if (!handle) {
        fn_ptr = nullptr;
        return false;
    }
#ifdef _WIN32
    auto sym = GetProcAddress((HMODULE) handle, name);
#else
    auto sym = dlsym(handle, name);
#endif
    if (!sym) {
        fn_ptr = nullptr;
        return false;
    }
    fn_ptr = reinterpret_cast<T>(sym);
    return true;
}

} // namespace

LlamaLibrary & LlamaLibrary::instance() {
    static LlamaLibrary lib;
    return lib;
}

bool LlamaLibrary::ensure_loaded(const std::string & lib_dir, std::string & err) {
    if (loaded_) {
        return true;
    }

    std::string ggml_base_path;
    std::string llama_path;
#ifdef _WIN32
    handle_ggml_base_ = (void *) GetModuleHandleA("ggml.dll");
    ggml_base_path = lib_dir + "\\ggml.dll";
    if (!handle_ggml_base_) {
        handle_ggml_base_ = (void *) LoadLibraryA(ggml_base_path.c_str());
    }
    if (!handle_ggml_base_) {
        handle_ggml_base_ = (void *) GetModuleHandleA("ggml-base.dll");
    }
    if (!handle_ggml_base_) {
        ggml_base_path = lib_dir + "\\ggml-base.dll";
        handle_ggml_base_ = (void *) LoadLibraryA(ggml_base_path.c_str());
    }
    llama_path = lib_dir + "\\llama.dll";
    handle_llama_ = (void *) LoadLibraryA(llama_path.c_str());
#else
    ggml_base_path = lib_dir + "/libggml.so";
    handle_ggml_base_ = dlopen(ggml_base_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_ggml_base_) {
        ggml_base_path = lib_dir + "/libggml-base.so";
        handle_ggml_base_ = dlopen(ggml_base_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    }
    llama_path = lib_dir + "/libllama.so";
    handle_llama_ = dlopen(llama_path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (!handle_llama_) {
        err = "Failed to load llama runtime: " + llama_path;
        return false;
    }

    if (!load_symbol_table(err)) {
#ifdef _WIN32
        FreeLibrary((HMODULE) handle_llama_);
        if (handle_ggml_base_) {
            FreeLibrary((HMODULE) handle_ggml_base_);
        }
#else
        dlclose(handle_llama_);
        if (handle_ggml_base_) {
            dlclose(handle_ggml_base_);
        }
#endif
        handle_llama_ = nullptr;
        handle_ggml_base_ = nullptr;
        return false;
    }

    // Register logger early to capture backend discovery logs
    if (llama_log_set) {
        llama_log_set([](enum llama_log_level level, const char * text, void * user_data) {
            (void) user_data;
            // Translate llama_log_level to our Logger
            Logger::instance().log_backend((int)level, text);
        }, nullptr);
    }

    if (ggml_backend_load_all_from_path) {
        ggml_backend_load_all_from_path(lib_dir.c_str());
    } else if (ggml_backend_load_all) {
        ggml_backend_load_all();
    }
    llama_backend_init();
    
    loaded_ = true;
    return true;
}

bool LlamaLibrary::load_symbol_table(std::string & err) {
    if (!load_fn(handle_llama_, "llama_model_default_params", llama_model_default_params, err)) return false;
    if (!load_fn(handle_llama_, "llama_model_load_from_file", llama_model_load_from_file, err)) return false;
    if (!load_fn(handle_llama_, "llama_model_free", llama_model_free, err)) return false;
    if (!load_fn(handle_llama_, "llama_model_get_vocab", llama_model_get_vocab, err)) return false;
    if (!load_fn(handle_llama_, "llama_model_n_embd", llama_model_n_embd, err)) return false;
    if (!load_fn(handle_llama_, "llama_model_n_ctx_train", llama_model_n_ctx_train, err)) return false;

    if (!load_fn(handle_llama_, "llama_context_default_params", llama_context_default_params, err)) return false;
    if (!load_fn(handle_llama_, "llama_init_from_model", llama_init_from_model, err)) return false;
    if (!load_fn(handle_llama_, "llama_free", llama_free, err)) return false;

    if (!load_fn(handle_llama_, "llama_backend_init", llama_backend_init, err)) return false;
    if (!load_fn(handle_llama_, "llama_batch_init", llama_batch_init, err)) return false;
    if (!load_fn(handle_llama_, "llama_batch_free", llama_batch_free, err)) return false;
    if (!load_fn(handle_llama_, "llama_decode", llama_decode, err)) return false;
    if (!load_fn(handle_llama_, "llama_get_logits_ith", llama_get_logits_ith, err)) return false;
    if (!load_fn(handle_llama_, "llama_get_embeddings", llama_get_embeddings, err)) return false;
    try_load_fn(handle_llama_, "llama_get_embeddings_ith", llama_get_embeddings_ith);
    if (!load_fn(handle_llama_, "llama_vocab_n_tokens", llama_vocab_n_tokens, err)) return false;
    if (!load_fn(handle_llama_, "llama_vocab_eos", llama_vocab_eos, err)) return false;
    if (!load_fn(handle_llama_, "llama_get_memory", llama_get_memory, err)) return false;
    if (!load_fn(handle_llama_, "llama_memory_clear", llama_memory_clear, err)) return false;

    if (!load_fn(handle_llama_, "llama_sampler_chain_default_params", llama_sampler_chain_default_params, err)) return false;
    if (!load_fn(handle_llama_, "llama_sampler_chain_init", llama_sampler_chain_init, err)) return false;
    if (!load_fn(handle_llama_, "llama_sampler_chain_add", llama_sampler_chain_add, err)) return false;
    if (!load_fn(handle_llama_, "llama_sampler_init_greedy", llama_sampler_init_greedy, err)) return false;
    if (!load_fn(handle_llama_, "llama_sampler_init_dist", llama_sampler_init_dist, err)) return false;
    if (!load_fn(handle_llama_, "llama_sampler_init_temp", llama_sampler_init_temp, err)) return false;
    if (!load_fn(handle_llama_, "llama_sampler_init_top_k", llama_sampler_init_top_k, err)) return false;
    if (!load_fn(handle_llama_, "llama_sampler_init_top_p", llama_sampler_init_top_p, err)) return false;
    if (!load_fn(handle_llama_, "llama_sampler_init_min_p", llama_sampler_init_min_p, err)) return false;
    if (!load_fn(handle_llama_, "llama_sampler_init_penalties", llama_sampler_init_penalties, err)) return false;
    if (!load_fn(handle_llama_, "llama_sampler_sample", llama_sampler_sample, err)) return false;
    if (!load_fn(handle_llama_, "llama_sampler_accept", llama_sampler_accept, err)) return false;
    if (!load_fn(handle_llama_, "llama_sampler_free", llama_sampler_free, err)) return false;

    if (!load_fn(handle_llama_, "llama_log_set", llama_log_set, err)) return false;

    if (!try_load_fn(handle_llama_, "ggml_backend_load_all_from_path", ggml_backend_load_all_from_path)) {
        try_load_fn(handle_ggml_base_, "ggml_backend_load_all_from_path", ggml_backend_load_all_from_path);
    }
    if (!try_load_fn(handle_llama_, "ggml_backend_load_all", ggml_backend_load_all)) {
        try_load_fn(handle_ggml_base_, "ggml_backend_load_all", ggml_backend_load_all);
    }

    return true;
}

LlamaModel::~LlamaModel() {
    free();
}

bool LlamaModel::load(const std::string & path, int32_t n_gpu_layers, std::string & err) {
    free();
    auto & lib = LlamaLibrary::instance();
    llama_model_params params = lib.llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    params.vocab_only = false;
    params.use_mmap = true;
    params.use_mlock = false;
    params.no_alloc = false;
    model_ = lib.llama_model_load_from_file(path.c_str(), params);
    if (!model_) {
        err = "llama_model_load_from_file failed: " + path;
        return false;
    }
    vocab_ = lib.llama_model_get_vocab(model_);
    n_embd_ = lib.llama_model_n_embd(model_);
    n_vocab_ = lib.llama_vocab_n_tokens(vocab_);
    eos_id_ = (int32_t) lib.llama_vocab_eos(vocab_);
    n_ctx_train_ = lib.llama_model_n_ctx_train(model_);
    return true;
}

void LlamaModel::free() {
    if (model_) {
        LlamaLibrary::instance().llama_model_free(model_);
        model_ = nullptr;
    }
    vocab_ = nullptr;
    n_embd_ = 0;
    n_vocab_ = 0;
    eos_id_ = -1;
    n_ctx_train_ = 0;
}

LlamaContext::~LlamaContext() {
    free();
}

bool LlamaContext::init(LlamaModel & model, int32_t n_ctx, int32_t n_threads, bool embeddings, std::string & err) {
    free();
    auto & lib = LlamaLibrary::instance();
    llama_context_params params = lib.llama_context_default_params();
    params.n_ctx = (uint32_t) n_ctx;
    params.n_batch = (uint32_t) n_ctx;
    params.n_ubatch = (uint32_t) (n_ctx > 512 ? 512 : n_ctx);
    params.n_seq_max = 1;
    params.embeddings = embeddings;
    params.flash_attn_type = 1;
    params.offload_kqv = true;
    params.no_perf = true;
    params.n_threads = n_threads > 0 ? n_threads : 4;
    params.n_threads_batch = n_threads > 0 ? n_threads : 4;
    ctx_ = lib.llama_init_from_model(model.raw_model(), params);
    if (!ctx_) {
        err = "llama_init_from_model failed";
        return false;
    }
    n_ctx_ = n_ctx;
    n_embd_ = model.n_embd();
    return true;
}

void LlamaContext::free() {
    if (ctx_) {
        LlamaLibrary::instance().llama_free(ctx_);
        ctx_ = nullptr;
    }
    n_ctx_ = 0;
    n_embd_ = 0;
}

int32_t LlamaContext::decode(llama_batch batch) const {
    return LlamaLibrary::instance().llama_decode(ctx_, batch);
}

float * LlamaContext::get_logits_ith(int32_t i) const {
    return LlamaLibrary::instance().llama_get_logits_ith(ctx_, i);
}

float * LlamaContext::get_embeddings() const {
    return LlamaLibrary::instance().llama_get_embeddings(ctx_);
}

float * LlamaContext::get_embeddings_ith(int32_t i) const {
    auto & lib = LlamaLibrary::instance();
    if (lib.llama_get_embeddings_ith) {
        return lib.llama_get_embeddings_ith(ctx_, i);
    }
    if (i < 0 || n_embd_ <= 0) {
        return nullptr;
    }
    float * all = lib.llama_get_embeddings(ctx_);
    if (!all) {
        return nullptr;
    }
    return all + (size_t) i * (size_t) n_embd_;
}

void LlamaContext::clear_kv_cache() const {
    auto & lib = LlamaLibrary::instance();
    void * mem = lib.llama_get_memory(ctx_);
    lib.llama_memory_clear(mem, true);
}

LlamaBatch::~LlamaBatch() {
    free();
}

bool LlamaBatch::init(int32_t n_tokens, int32_t embd_dim, int32_t n_seq_max, std::string & err) {
    free();
    if (n_tokens <= 0 || embd_dim <= 0) {
        err = "Invalid batch shape";
        return false;
    }
    auto & lib = LlamaLibrary::instance();
    batch_ = lib.llama_batch_init(n_tokens, embd_dim, n_seq_max);
    max_tokens_ = n_tokens;
    embd_dim_ = embd_dim;
    inited_ = true;
    return true;
}

void LlamaBatch::free() {
    if (inited_) {
        LlamaLibrary::instance().llama_batch_free(batch_);
        std::memset(&batch_, 0, sizeof(batch_));
    }
    max_tokens_ = 0;
    embd_dim_ = 0;
    inited_ = false;
}

bool LlamaBatch::set_embeddings(
    const float * embd,
    int32_t n_tokens,
    int32_t embd_dim,
    const int32_t * pos,
    int32_t pos_count,
    int32_t seq_id,
    std::string & err) {
    if (!inited_) {
        err = "Batch is not initialized";
        return false;
    }
    if (!embd || !pos) {
        err = "Null embedding or position buffer";
        return false;
    }
    if (n_tokens <= 0 || n_tokens > max_tokens_) {
        err = "n_tokens out of range";
        return false;
    }
    if (pos_count <= 0) {
        err = "pos_count out of range";
        return false;
    }
    if (pos_count > max_tokens_) {
        err = "pos_count exceeds batch capacity";
        return false;
    }
    if (embd_dim != embd_dim_) {
        err = "Embedding dim mismatch";
        return false;
    }
    std::memcpy(batch_.embd, embd, (size_t) n_tokens * (size_t) embd_dim * sizeof(float));
    std::memcpy(batch_.pos, pos, (size_t) pos_count * sizeof(int32_t));

    batch_.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch_.n_seq_id[i] = 1;
        batch_.seq_id[i][0] = seq_id;
        batch_.logits[i] = (i == n_tokens - 1) ? 1 : 0;
    }
    return true;
}

LlamaSampler::~LlamaSampler() {
    free();
}

bool LlamaSampler::init(
    float temperature,
    float top_p,
    int32_t top_k,
    float min_p,
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty,
    int32_t penalty_last_n,
    uint32_t seed,
    std::string & err) {
    free();
    auto & lib = LlamaLibrary::instance();

    llama_sampler_chain_params params = lib.llama_sampler_chain_default_params();
    sampler_ = lib.llama_sampler_chain_init(params);
    if (!sampler_) {
        err = "llama_sampler_chain_init failed";
        return false;
    }

    const bool has_penalty =
        repeat_penalty != 1.0f || frequency_penalty != 0.0f || presence_penalty != 0.0f;
    if (has_penalty) {
        lib.llama_sampler_chain_add(
            sampler_,
            lib.llama_sampler_init_penalties(
                penalty_last_n,
                repeat_penalty,
                frequency_penalty,
                presence_penalty));
    }

    if (temperature <= 0.0f) {
        lib.llama_sampler_chain_add(sampler_, lib.llama_sampler_init_greedy());
        return true;
    }

    if (top_k > 0) {
        lib.llama_sampler_chain_add(sampler_, lib.llama_sampler_init_top_k(top_k));
    }
    if (top_p < 1.0f) {
        lib.llama_sampler_chain_add(sampler_, lib.llama_sampler_init_top_p(top_p, 1));
    }
    if (min_p > 0.0f && min_p < 1.0f) {
        lib.llama_sampler_chain_add(sampler_, lib.llama_sampler_init_min_p(min_p, 1));
    }
    lib.llama_sampler_chain_add(sampler_, lib.llama_sampler_init_temp(temperature));
    lib.llama_sampler_chain_add(sampler_, lib.llama_sampler_init_dist(seed));
    return true;
}

void LlamaSampler::free() {
    if (sampler_) {
        LlamaLibrary::instance().llama_sampler_free(sampler_);
        sampler_ = nullptr;
    }
}

int32_t LlamaSampler::sample(LlamaContext & ctx, int32_t idx) const {
    return (int32_t) LlamaLibrary::instance().llama_sampler_sample(sampler_, ctx.raw(), idx);
}

void LlamaSampler::accept(int32_t token) const {
    LlamaLibrary::instance().llama_sampler_accept(sampler_, (llama_token) token);
}

} // namespace qwen3_tts
