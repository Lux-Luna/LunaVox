#pragma once

#include "assets_manager.h"
#include "llama_wrapper.h"

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
        int32_t n_threads);

    void unload();
    bool is_loaded() const { return loaded_; }
    const std::string & get_error() const { return error_msg_; }
    int32_t hidden_dim() const { return hidden_dim_; }

    bool generate(
        const std::vector<int32_t> & text_tokens,
        const std::vector<int32_t> & role_prefix_tokens,
        const float * speaker_embedding,
        int32_t max_frames,
        int32_t language_id,
        float repetition_penalty,
        float temperature,
        float top_p,
        int32_t top_k,
        std::vector<int32_t> & output_codes);

private:
    bool run_prefill(
        const std::vector<int32_t> & text_tokens,
        const std::vector<int32_t> & role_prefix_tokens,
        const float * speaker_embedding,
        int32_t language_id,
        std::vector<float> & hidden_out);

    bool predict_frame(
        const std::vector<float> & master_hidden,
        int32_t code0,
        float temperature,
        float top_p,
        int32_t top_k,
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
    int32_t cur_pos_ = 0;
    int32_t step_idx_ = 0;
    int32_t trailing_count_ = 0;

    const AssetsManager * assets_ = nullptr;
    std::vector<float> trailing_text_pool_;

    LlamaModel talker_model_;
    LlamaModel predictor_model_;
    LlamaContext talker_ctx_;
    LlamaContext predictor_ctx_;
    LlamaBatch talker_batch_;
    LlamaBatch predictor_batch_;
};

} // namespace qwen3_tts
