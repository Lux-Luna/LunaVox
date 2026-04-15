#pragma once

#include "assets_manager.h"
#include "llama_wrapper.h"
#include "model_profile.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace lunavox {

class TalkerPredictor {
public:
    bool load(
        const std::string & lib_dir,
        const std::string & talker_model_path,
        const std::string & predictor_model_path,
        const AssetsManager & assets,
        const ModelProfile & profile,
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

    // Streaming hook: fired inside the generate() loop after each frame
    // (16 codes) is appended to output_codes. The callback receives a pointer
    // into output_codes for the newly appended frame and the number of frames
    // (currently always 1, left as a parameter for future batched hooks).
    // Empty callback = disabled (zero overhead).
    using on_frames_ready_cb = std::function<void(const int32_t * new_frame_codes, int32_t n_new_frames)>;
    void set_on_frames_ready(on_frames_ready_cb cb) { on_frames_ready_ = std::move(cb); }
    void clear_on_frames_ready() { on_frames_ready_ = nullptr; }

    int32_t last_eos_step() const { return last_eos_step_; } // -1 means reached max_frames without EOS
    int32_t last_ctx_required() const { return last_ctx_required_; }
    int32_t last_ctx_allocated() const { return last_ctx_allocated_; }
    int32_t last_ctx_cap() const { return last_ctx_cap_; }
    bool last_ctx_overflow() const { return last_ctx_overflow_; }

    int64_t last_t_prefill_ms() const { return t_prefill_ms_; }
    int64_t last_t_decode_loop_ms() const { return t_decode_loop_ms_; }
    int64_t last_t_talker_post_ms() const { return t_talker_post_ms_; }
    int64_t last_t_predictor_sample_ms() const { return t_predictor_sample_ms_; }
    // Talker-post breakdown (subset of talker_post)
    int64_t last_t_talker_decode_ms() const { return t_talker_decode_ms_; }
    int64_t last_t_talker_post_prep_ms() const { return t_talker_post_prep_ms_; }
    int64_t last_t_talker_post_copy_ms() const { return t_talker_post_copy_ms_; }

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

    int64_t t_prefill_ms_ = 0;
    int64_t t_decode_loop_ms_ = 0;
    int64_t t_talker_post_ms_ = 0;
    int64_t t_predictor_sample_ms_ = 0;
    int64_t t_talker_decode_ms_ = 0;
    int64_t t_talker_post_prep_ms_ = 0;
    int64_t t_talker_post_copy_ms_ = 0;

    const AssetsManager * assets_ = nullptr;
    ModelProfile profile_;

    LlamaModel talker_model_;
    LlamaModel predictor_model_;
    LlamaContext talker_ctx_;
    LlamaContext predictor_ctx_;
    LlamaBatch talker_batch_;
    LlamaBatch predictor_batch_;

    // Reused scratch buffers to avoid per-frame heap churn in predict_frame().
    std::vector<float> scratch_m_pred_;
    std::vector<float> scratch_emb0_pred_;
    std::vector<float> scratch_prefill_;
    std::vector<float> scratch_emb_next_;

    on_frames_ready_cb on_frames_ready_;
};

} // namespace lunavox
