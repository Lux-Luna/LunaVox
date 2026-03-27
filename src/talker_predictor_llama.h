#pragma once

#include "assets_manager.h"
#include "llama_wrapper.h"
#include "model_profile.h"

#include <cstdint>
#include <string>
#include <vector>

namespace qwen3_tts {

class TalkerPredictorLlama {
public:
    bool load(
        const std::string & lib_dir,
        const std::string & talker_model_path,
        const std::string & predictor_model_path,
        const AssetsManager & assets,
        const RuntimeModelProfile & profile,
        int32_t n_threads);

    void unload();
    bool is_loaded() const { return loaded_; }
    const std::string & get_error() const { return error_msg_; }
    int32_t hidden_dim() const { return hidden_dim_; }

    bool generate(
        const std::vector<int32_t> & text_tokens,
        const std::vector<int32_t> & ref_text_tokens,
        const std::vector<int32_t> & role_prefix_tokens,
        const std::vector<int32_t> & instruct_tokens,
        const float * speaker_embedding,
        const int32_t * ref_codes,
        int32_t n_ref_frames,
        int32_t max_frames,
        int32_t language_id,
        float repetition_penalty,
        float talker_temperature,
        float talker_top_p,
        int32_t talker_top_k,
        bool predictor_do_sample,
        float predictor_temperature,
        float predictor_top_p,
        int32_t predictor_top_k,
        int32_t talker_seed,
        int32_t predictor_seed,
        std::vector<int32_t> & output_codes);

    int32_t last_eos_step() const { return last_eos_step_; } // -1 means reached max_frames without EOS
    int32_t last_ctx_required() const { return last_ctx_required_; }
    int32_t last_ctx_allocated() const { return last_ctx_allocated_; }
    int32_t last_ctx_cap() const { return last_ctx_cap_; }
    bool last_ctx_overflow() const { return last_ctx_overflow_; }

private:
    bool ensure_talker_runtime(int32_t request_ctx);
    int32_t estimate_prompt_tokens(
        const std::vector<int32_t> & text_tokens,
        const std::vector<int32_t> & ref_text_tokens,
        const std::vector<int32_t> & role_prefix_tokens,
        const std::vector<int32_t> & instruct_tokens,
        bool has_speaker_embedding,
        bool use_clone_icl,
        int32_t n_ref_frames,
        int32_t language_id) const;

    bool run_prefill(
        const std::vector<int32_t> & text_tokens,
        const std::vector<int32_t> & ref_text_tokens,
        const std::vector<int32_t> & role_prefix_tokens,
        const std::vector<int32_t> & instruct_tokens,
        const float * speaker_embedding,
        const int32_t * ref_codes,
        int32_t n_ref_frames,
        int32_t language_id,
        std::vector<float> & hidden_out);

    bool predict_frame(
        const std::vector<float> & master_hidden,
        int32_t code0,
        LlamaSampler & predictor_sampler,
        std::vector<int32_t> & frame_codes,
        std::vector<float> & audio_sum);

    bool run_decode_step(const std::vector<float> & audio_sum, std::vector<float> & hidden_out);
    bool sample_with_mask(
        LlamaContext & ctx,
        const LlamaModel & model,
        LlamaSampler & sampler,
        int32_t limit_start,
        int32_t limit_end,
        const int32_t * allow_tokens,
        int32_t n_allow,
        int32_t & sampled_token);

    bool loaded_ = false;
    std::string error_msg_;
    int32_t hidden_dim_ = 0;
    int32_t predictor_dim_ = 0;
    int32_t codebook_vocab_size_ = 2048;
    int32_t cur_pos_ = 0;
    int32_t last_eos_step_ = -1;
    int32_t n_threads_ = 4;
    int32_t talker_ctx_cap_ = 0;
    int32_t talker_ctx_train_ = 0;
    int32_t talker_batch_cap_ = 0;
    int32_t last_ctx_required_ = 0;
    int32_t last_ctx_allocated_ = 0;
    int32_t last_ctx_cap_ = 0;
    bool last_ctx_overflow_ = false;

    const AssetsManager * assets_ = nullptr;
    RuntimeModelProfile profile_;

    LlamaModel talker_model_;
    LlamaModel predictor_model_;
    LlamaContext talker_ctx_;
    LlamaContext predictor_ctx_;
    LlamaBatch talker_batch_;
    LlamaBatch predictor_batch_;
};

} // namespace qwen3_tts
