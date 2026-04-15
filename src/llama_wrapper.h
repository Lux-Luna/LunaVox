#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lunavox {

// Minimal llama.cpp C API mirror used by the new talker/predictor runtime.
using llama_token = int32_t;
using llama_pos = int32_t;
using llama_seq_id = int32_t;

enum llama_log_level {
    LLAMA_LOG_LEVEL_ERROR = 2,
    LLAMA_LOG_LEVEL_WARN  = 3,
    LLAMA_LOG_LEVEL_INFO  = 4,
    LLAMA_LOG_LEVEL_DEBUG = 5
};

typedef void (*llama_log_callback)(enum llama_log_level level, const char * text, void * user_data);

struct llama_model_params {
    void * devices;
    void * tensor_buft_overrides;
    int32_t n_gpu_layers;
    int32_t split_mode;
    int32_t main_gpu;
    float * tensor_split;
    void * progress_callback;
    void * progress_callback_user_data;
    void * kv_overrides;
    bool vocab_only;
    bool use_mmap;
    bool use_direct_io;
    bool use_mlock;
    bool check_tensors;
    bool use_extra_bufts;
    bool no_host;
    bool no_alloc;
};

struct llama_context_params {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t n_threads;
    int32_t n_threads_batch;
    int32_t rope_scaling_type;
    int32_t pooling_type;
    int32_t attention_type;
    int32_t flash_attn_type;
    float rope_freq_base;
    float rope_freq_scale;
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    uint32_t yarn_orig_ctx;
    float defrag_thold;
    void * cb_eval;
    void * cb_eval_user_data;
    int32_t type_k;
    int32_t type_v;
    void * abort_callback;
    void * abort_callback_data;
    bool embeddings;
    bool offload_kqv;
    bool no_perf;
    bool op_offload;
    bool swa_full;
    bool kv_unified;
    void * samplers;
    size_t n_samplers;
};

struct llama_sampler_chain_params {
    bool no_perf;
};

struct llama_batch {
    int32_t n_tokens;
    llama_token * token;
    float * embd;
    llama_pos * pos;
    int32_t * n_seq_id;
    llama_seq_id ** seq_id;
    int8_t * logits;
};

struct llama_logit_bias {
    llama_token token;
    float bias;
};

class LlamaLibrary {
public:
    static LlamaLibrary & instance();

    bool ensure_loaded(const std::string & lib_dir, std::string & err);
    bool is_loaded() const { return loaded_; }

    // model lifecycle
    llama_model_params (*llama_model_default_params)();
    void * (*llama_model_load_from_file)(const char * path, llama_model_params params);
    void (*llama_model_free)(void * model);
    void * (*llama_model_get_vocab)(void * model);
    int32_t (*llama_model_n_embd)(void * model);
    int32_t (*llama_model_n_ctx_train)(void * model);

    // context lifecycle
    llama_context_params (*llama_context_default_params)();
    void * (*llama_init_from_model)(void * model, llama_context_params params);
    void (*llama_free)(void * ctx);

    // backend
    void (*llama_backend_init)();
    void (*ggml_backend_load_all)();
    void (*ggml_backend_load_all_from_path)(const char * dir_path);

    // batch and decode
    llama_batch (*llama_batch_init)(int32_t n_tokens, int32_t embd_dim, int32_t n_seq_max);
    void (*llama_batch_free)(llama_batch batch);
    int32_t (*llama_decode)(void * ctx, llama_batch batch);
    float * (*llama_get_logits_ith)(void * ctx, int32_t i);
    float * (*llama_get_embeddings)(void * ctx);
    float * (*llama_get_embeddings_ith)(void * ctx, int32_t i);

    // vocab / memory
    int32_t (*llama_vocab_n_tokens)(void * vocab);
    llama_token (*llama_vocab_eos)(void * vocab);
    void * (*llama_get_memory)(void * ctx);
    void (*llama_memory_clear)(void * mem, bool set);

    // samplers
    llama_sampler_chain_params (*llama_sampler_chain_default_params)();
    void * (*llama_sampler_chain_init)(llama_sampler_chain_params params);
    void (*llama_sampler_chain_add)(void * chain, void * sampler);
    void * (*llama_sampler_init_greedy)();
    void * (*llama_sampler_init_dist)(uint32_t seed);
    void * (*llama_sampler_init_temp)(float t);
    void * (*llama_sampler_init_top_k)(int32_t k);
    void * (*llama_sampler_init_top_p)(float p, size_t min_keep);
    void * (*llama_sampler_init_min_p)(float p, size_t min_keep);
    void * (*llama_sampler_init_penalties)(int32_t penalty_last_n, float repeat_penalty, float freq_penalty, float presence_penalty);
    llama_token (*llama_sampler_sample)(void * sampler, void * ctx, int32_t idx);
    void (*llama_sampler_accept)(void * sampler, llama_token token);
    void (*llama_sampler_free)(void * sampler);

    // logger
    void (*llama_log_set)(llama_log_callback log_callback, void * user_data);

private:
    LlamaLibrary() = default;
    ~LlamaLibrary() = default;
    LlamaLibrary(const LlamaLibrary &) = delete;
    LlamaLibrary & operator=(const LlamaLibrary &) = delete;

    bool load_symbol_table(std::string & err);
    bool loaded_ = false;
    void * handle_llama_ = nullptr;
    void * handle_ggml_base_ = nullptr;
};

class LlamaModel {
public:
    LlamaModel() = default;
    ~LlamaModel();

    bool load(const std::string & path, int32_t n_gpu_layers, std::string & err);
    void free();
    bool is_loaded() const { return model_ != nullptr; }

    void * raw_model() const { return model_; }
    void * vocab() const { return vocab_; }
    int32_t n_embd() const { return n_embd_; }
    int32_t n_vocab() const { return n_vocab_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t n_ctx_train() const { return n_ctx_train_; }

private:
    void * model_ = nullptr;
    void * vocab_ = nullptr;
    int32_t n_embd_ = 0;
    int32_t n_vocab_ = 0;
    int32_t eos_id_ = -1;
    int32_t n_ctx_train_ = 0;
};

class LlamaContext {
public:
    LlamaContext() = default;
    ~LlamaContext();

    bool init(
        LlamaModel & model,
        int32_t n_ctx,
        int32_t n_threads,
        bool embeddings,
        int32_t n_batch_cap,
        int32_t n_ubatch_cap,
        const char * tag,
        std::string & err,
        int32_t type_k = 1,
        int32_t type_v = 1);
    void free();
    bool is_ready() const { return ctx_ != nullptr; }

    int32_t decode(llama_batch batch) const;
    float * get_logits_ith(int32_t i) const;
    float * get_embeddings() const;
    float * get_embeddings_ith(int32_t i) const;
    void clear_kv_cache() const;
    void * raw() const { return ctx_; }
    int32_t n_ctx() const { return n_ctx_; }
    int32_t n_embd() const { return n_embd_; }

private:
    void * ctx_ = nullptr;
    int32_t n_ctx_ = 0;
    int32_t n_embd_ = 0;
};

class LlamaBatch {
public:
    LlamaBatch() = default;
    ~LlamaBatch();

    bool init(int32_t n_tokens, int32_t embd_dim, int32_t n_seq_max, std::string & err);
    void free();
    bool is_ready() const { return inited_; }

    bool set_embeddings(
        const float * embd,
        int32_t n_tokens,
        int32_t embd_dim,
        const int32_t * pos,
        int32_t pos_count,
        int32_t seq_id,
        std::string & err);

    llama_batch & raw() { return batch_; }
    const llama_batch & raw() const { return batch_; }
    int32_t capacity() const { return max_tokens_; }

private:
    llama_batch batch_{};
    int32_t max_tokens_ = 0;
    int32_t embd_dim_ = 0;
    bool inited_ = false;
};

class LlamaSampler {
public:
    LlamaSampler() = default;
    ~LlamaSampler();

    bool init(
        float temperature,
        float top_p,
        int32_t top_k,
        float min_p,
        float repeat_penalty,
        float frequency_penalty,
        float presence_penalty,
        int32_t penalty_last_n,
        uint32_t seed,
        std::string & err);

    void free();
    bool is_ready() const { return sampler_ != nullptr; }

    int32_t sample(LlamaContext & ctx, int32_t idx) const;
    void accept(int32_t token) const;

private:
    void * sampler_ = nullptr;
};

} // namespace lunavox
