#include "talker_predictor_llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace qwen3_tts {

namespace {

static constexpr int32_t kCodecPad = 2148;
static constexpr int32_t kCodecBos = 2149;
static constexpr int32_t kCodecEos = 2150;
static constexpr int32_t kThink = 2154;
static constexpr int32_t kNoThink = 2155;
static constexpr int32_t kThinkBos = 2156;
static constexpr int32_t kThinkEos = 2157;
static constexpr int32_t kTtsBos = 151672;
static constexpr int32_t kTtsEos = 151673;

static inline bool is_valid_language_id(int32_t id) {
    return id >= 2048 && id <= 2147;
}

static inline void add_vec(float * dst, const float * src, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}

} // namespace

bool TalkerPredictorLlama::load(
    const std::string & lib_dir,
    const std::string & talker_model_path,
    const std::string & predictor_model_path,
    const AssetsManager & assets,
    int32_t n_threads) {
    unload();
    assets_ = &assets;

    auto & lib = LlamaLibrary::instance();
    if (!lib.ensure_loaded(lib_dir, error_msg_)) {
        return false;
    }

    if (!talker_model_.load(talker_model_path, -1, error_msg_)) {
        return false;
    }
    if (!predictor_model_.load(predictor_model_path, -1, error_msg_)) {
        return false;
    }

    hidden_dim_ = assets.hidden_dim();
    predictor_dim_ = assets.predictor_dim();
    if (hidden_dim_ <= 0 || predictor_dim_ <= 0) {
        error_msg_ = "Invalid embedding dimensions in assets";
        return false;
    }
    if (talker_model_.n_embd() != hidden_dim_) {
        error_msg_ = "Talker model hidden dim mismatch with embeddings table";
        return false;
    }
    if (predictor_model_.n_embd() != predictor_dim_) {
        error_msg_ = "Predictor model hidden dim mismatch with predictor projection";
        return false;
    }

    if (!talker_ctx_.init(talker_model_, 4096, n_threads, true, error_msg_)) {
        return false;
    }
    if (!predictor_ctx_.init(predictor_model_, 64, n_threads, false, error_msg_)) {
        return false;
    }

    if (!talker_batch_.init(4096, hidden_dim_, 1, error_msg_)) {
        return false;
    }
    if (!predictor_batch_.init(2, predictor_dim_, 1, error_msg_)) {
        return false;
    }

    loaded_ = true;
    cur_pos_ = 0;
    step_idx_ = 0;
    trailing_count_ = 0;
    return true;
}

void TalkerPredictorLlama::unload() {
    loaded_ = false;
    error_msg_.clear();
    cur_pos_ = 0;
    step_idx_ = 0;
    trailing_count_ = 0;
    trailing_text_pool_.clear();
    predictor_batch_.free();
    talker_batch_.free();
    predictor_ctx_.free();
    talker_ctx_.free();
    predictor_model_.free();
    talker_model_.free();
    assets_ = nullptr;
    hidden_dim_ = 0;
    predictor_dim_ = 0;
}

bool TalkerPredictorLlama::sample_with_mask(
    LlamaContext & ctx,
    const LlamaModel & model,
    LlamaSampler & sampler,
    int32_t limit_start,
    int32_t limit_end,
    const int32_t * allow_tokens,
    int32_t n_allow,
    int32_t & sampled_token) {
    float * logits = ctx.get_logits_ith(-1);
    if (!logits) {
        error_msg_ = "llama_get_logits_ith returned null";
        return false;
    }
    const int32_t n_vocab = model.n_vocab();
    if (n_vocab <= 0) {
        error_msg_ = "Invalid vocab size";
        return false;
    }

    const float neg_inf = -1e10f;
    for (int32_t i = 0; i < n_vocab; ++i) {
        bool keep = (i >= limit_start && i < limit_end);
        if (!keep && allow_tokens) {
            for (int32_t j = 0; j < n_allow; ++j) {
                if (i == allow_tokens[j]) {
                    keep = true;
                    break;
                }
            }
        }
        if (!keep) {
            logits[i] = neg_inf;
        }
    }

    sampled_token = sampler.sample(ctx, -1);
    return true;
}

bool TalkerPredictorLlama::run_prefill(
    const std::vector<int32_t> & text_tokens,
    const std::vector<int32_t> & role_prefix_tokens,
    const float * speaker_embedding,
    int32_t language_id,
    std::vector<float> & hidden_out) {
    if (!assets_ || text_tokens.empty()) {
        error_msg_ = "run_prefill invalid input";
        return false;
    }
    const float * tts_pad = assets_->tts_pad();
    if (!tts_pad) {
        error_msg_ = "Missing tts_pad row in text embedding table";
        return false;
    }

    // Follow the streaming prompt style used by Qwen3-TTS-GGUF:
    // prefix + first fused text/audio token, then keep remaining text in trailing pool.
    std::vector<float> prompt_flat;
    std::vector<float> tmp_vec((size_t) hidden_dim_, 0.0f);
    auto append_sum = [&](const float * a, const float * b) {
        for (int32_t i = 0; i < hidden_dim_; ++i) {
            tmp_vec[(size_t) i] = a[i] + b[i];
        }
        prompt_flat.insert(prompt_flat.end(), tmp_vec.begin(), tmp_vec.end());
    };

    const float * think = assets_->codec_row(0, kThink);
    const float * nothink = assets_->codec_row(0, kNoThink);
    const float * think_bos = assets_->codec_row(0, kThinkBos);
    const float * think_eos = assets_->codec_row(0, kThinkEos);
    if (!think || !nothink || !think_bos || !think_eos) {
        error_msg_ = "Missing protocol embedding rows";
        return false;
    }

    for (int32_t tid : role_prefix_tokens) {
        const float * role = assets_->text_row(tid);
        if (!role) {
            error_msg_ = "Role prefix token embedding out of range";
            return false;
        }
        prompt_flat.insert(prompt_flat.end(), role, role + hidden_dim_);
    }

    if (is_valid_language_id(language_id)) {
        append_sum(tts_pad, think);
        append_sum(tts_pad, think_bos);
        const float * lang = assets_->codec_row(0, language_id);
        if (!lang) {
            error_msg_ = "Language embedding not found in codec table";
            return false;
        }
        append_sum(tts_pad, lang);
        append_sum(tts_pad, think_eos);
    } else {
        append_sum(tts_pad, nothink);
        append_sum(tts_pad, think_bos);
        append_sum(tts_pad, think_eos);
    }

    if (speaker_embedding) {
        for (int32_t i = 0; i < hidden_dim_; ++i) {
            tmp_vec[(size_t) i] = tts_pad[i] + speaker_embedding[i];
        }
        prompt_flat.insert(prompt_flat.end(), tmp_vec.begin(), tmp_vec.end());
    }

    const float * tts_bos = assets_->text_row(kTtsBos);
    const float * codec_pad = assets_->codec_row(0, kCodecPad);
    if (!tts_bos || !codec_pad) {
        error_msg_ = "Missing TTS_BOS text or codec PAD embedding";
        return false;
    }
    append_sum(tts_bos, codec_pad);

    const float * text0 = assets_->text_row(text_tokens[0]);
    const float * codec_bos = assets_->codec_row(0, kCodecBos);
    if (!text0 || !codec_bos) {
        error_msg_ = "Missing first text or codec BOS embedding";
        return false;
    }
    append_sum(text0, codec_bos);

    const int32_t n_prompt_tokens = (int32_t) (prompt_flat.size() / (size_t) hidden_dim_);
    std::vector<int32_t> pos((size_t) n_prompt_tokens * 4);
    for (int32_t i = 0; i < n_prompt_tokens; ++i) {
        pos[(size_t) i] = cur_pos_ + i;
        pos[(size_t) n_prompt_tokens + (size_t) i] = cur_pos_ + i;
        pos[(size_t) 2 * (size_t) n_prompt_tokens + (size_t) i] = cur_pos_ + i;
        pos[(size_t) 3 * (size_t) n_prompt_tokens + (size_t) i] = 0;
    }

    if (!talker_batch_.set_embeddings(
            prompt_flat.data(),
            n_prompt_tokens,
            hidden_dim_,
            pos.data(),
            (int32_t) pos.size(),
            0,
            error_msg_)) {
        return false;
    }

    if (talker_ctx_.decode(talker_batch_.raw()) != 0) {
        error_msg_ = "Talker prefill decode failed";
        return false;
    }

    hidden_out.resize((size_t) hidden_dim_);
    const float * embd_all = talker_ctx_.get_embeddings();
    if (!embd_all) {
        error_msg_ = "Talker prefill embeddings pointer is null";
        return false;
    }
    const float * last = embd_all + (size_t) (n_prompt_tokens - 1) * (size_t) hidden_dim_;
    std::memcpy(hidden_out.data(), last, (size_t) hidden_dim_ * sizeof(float));
    cur_pos_ += n_prompt_tokens;

    trailing_text_pool_.clear();
    if (text_tokens.size() > 1) {
        trailing_text_pool_.reserve((text_tokens.size()) * (size_t) hidden_dim_);
        for (size_t i = 1; i < text_tokens.size(); ++i) {
            const float * row = assets_->text_row(text_tokens[i]);
            if (!row) {
                error_msg_ = "Text embedding row out of range";
                return false;
            }
            trailing_text_pool_.insert(trailing_text_pool_.end(), row, row + hidden_dim_);
        }
    }
    const float * tts_eos = assets_->text_row(kTtsEos);
    if (!tts_eos) {
        error_msg_ = "Missing TTS_EOS text embedding row";
        return false;
    }
    trailing_text_pool_.insert(trailing_text_pool_.end(), tts_eos, tts_eos + hidden_dim_);
    trailing_count_ = (int32_t) (trailing_text_pool_.size() / (size_t) hidden_dim_);
    step_idx_ = 0;
    return true;
}

bool TalkerPredictorLlama::predict_frame(
    const std::vector<float> & master_hidden,
    int32_t code0,
    float temperature,
    float top_p,
    int32_t top_k,
    std::vector<int32_t> & frame_codes,
    std::vector<float> & audio_sum) {
    frame_codes.clear();
    frame_codes.reserve(16);
    audio_sum.assign((size_t) hidden_dim_, 0.0f);

    predictor_ctx_.clear_kv_cache();

    std::vector<float> m_pred;
    if (!assets_->project_to_predictor(master_hidden.data(), m_pred)) {
        error_msg_ = "Failed to project master hidden for predictor";
        return false;
    }
    std::vector<float> emb0_pred;
    if (!assets_->codec_row_predictor(0, code0, emb0_pred)) {
        error_msg_ = "Failed to fetch predictor embedding for code_0";
        return false;
    }

    const float * raw0 = assets_->codec_row(0, code0);
    if (!raw0) {
        error_msg_ = "Missing raw codec embedding for code_0";
        return false;
    }
    add_vec(audio_sum.data(), raw0, hidden_dim_);
    frame_codes.push_back(code0);

    std::vector<float> prefill((size_t) predictor_dim_ * 2);
    std::memcpy(prefill.data(), m_pred.data(), (size_t) predictor_dim_ * sizeof(float));
    std::memcpy(prefill.data() + predictor_dim_, emb0_pred.data(), (size_t) predictor_dim_ * sizeof(float));
    int32_t ppos[2] = {0, 1};
    if (!predictor_batch_.set_embeddings(
            prefill.data(),
            2,
            predictor_dim_,
            ppos,
            2,
            0,
            error_msg_)) {
        return false;
    }
    if (predictor_ctx_.decode(predictor_batch_.raw()) != 0) {
        error_msg_ = "Predictor prefill decode failed";
        return false;
    }

    std::string sampler_err;
    LlamaSampler sampler;
    if (!sampler.init(
            temperature,
            top_p,
            top_k,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            64,
            (uint32_t) std::chrono::high_resolution_clock::now().time_since_epoch().count(),
            sampler_err)) {
        error_msg_ = sampler_err;
        return false;
    }

    for (int32_t cs = 1; cs < 16; ++cs) {
        const int32_t start = (cs - 1) * 2048;
        const int32_t end = cs * 2048;
        int32_t token_id = -1;
        if (!sample_with_mask(predictor_ctx_, predictor_model_, sampler, start, end, nullptr, 0, token_id)) {
            return false;
        }
        const int32_t code = token_id - start;
        if (code < 0 || code >= 2048) {
            error_msg_ = "Predictor sampled token out of expected codebook range";
            return false;
        }
        frame_codes.push_back(code);

        const float * raw = assets_->codec_row(cs, code);
        if (!raw) {
            error_msg_ = "Missing raw codec embedding row";
            return false;
        }
        add_vec(audio_sum.data(), raw, hidden_dim_);

        if (cs < 15) {
            std::vector<float> emb_next;
            if (!assets_->codec_row_predictor(cs, code, emb_next)) {
                error_msg_ = "Failed to project predictor codec embedding";
                return false;
            }
            int32_t pos = cs + 1;
            if (!predictor_batch_.set_embeddings(
                    emb_next.data(),
                    1,
                    predictor_dim_,
                    &pos,
                    1,
                    0,
                    error_msg_)) {
                return false;
            }
            if (predictor_ctx_.decode(predictor_batch_.raw()) != 0) {
                error_msg_ = "Predictor step decode failed";
                return false;
            }
        }
    }

    return true;
}

bool TalkerPredictorLlama::run_decode_step(const std::vector<float> & audio_sum, std::vector<float> & hidden_out) {
    if (!assets_) return false;
    if ((int32_t) audio_sum.size() != hidden_dim_) {
        error_msg_ = "audio_sum dim mismatch";
        return false;
    }

    const float * text_vec = nullptr;
    if (step_idx_ < trailing_count_) {
        text_vec = trailing_text_pool_.data() + (size_t) step_idx_ * (size_t) hidden_dim_;
    } else {
        text_vec = assets_->tts_pad();
    }
    if (!text_vec) {
        error_msg_ = "Failed to resolve trailing text vector";
        return false;
    }

    std::vector<float> fused((size_t) hidden_dim_);
    for (int32_t i = 0; i < hidden_dim_; ++i) {
        fused[(size_t) i] = audio_sum[(size_t) i] + text_vec[i];
    }

    int32_t pos4[4] = {cur_pos_, cur_pos_, cur_pos_, 0};
    if (!talker_batch_.set_embeddings(
            fused.data(),
            1,
            hidden_dim_,
            pos4,
            4,
            0,
            error_msg_)) {
        return false;
    }
    if (talker_ctx_.decode(talker_batch_.raw()) != 0) {
        error_msg_ = "Talker decode_step failed";
        return false;
    }

    const float * embd = talker_ctx_.get_embeddings();
    if (!embd) {
        error_msg_ = "Talker step embeddings pointer is null";
        return false;
    }
    hidden_out.resize((size_t) hidden_dim_);
    std::memcpy(hidden_out.data(), embd, (size_t) hidden_dim_ * sizeof(float));
    ++cur_pos_;
    ++step_idx_;
    return true;
}

bool TalkerPredictorLlama::generate(
    const std::vector<int32_t> & text_tokens,
    const std::vector<int32_t> & role_prefix_tokens,
    const float * speaker_embedding,
    int32_t max_frames,
    int32_t language_id,
    float repetition_penalty,
    float temperature,
    float top_p,
    int32_t top_k,
    std::vector<int32_t> & output_codes) {
    output_codes.clear();
    if (!loaded_) {
        error_msg_ = "TalkerPredictorLlama is not loaded";
        return false;
    }
    if (text_tokens.empty()) {
        error_msg_ = "Text token list is empty";
        return false;
    }
    if (max_frames <= 0) {
        error_msg_ = "max_frames must be positive";
        return false;
    }

    talker_ctx_.clear_kv_cache();
    predictor_ctx_.clear_kv_cache();
    cur_pos_ = 0;
    step_idx_ = 0;
    trailing_count_ = 0;

    std::vector<float> master_hidden;
    if (!run_prefill(text_tokens, role_prefix_tokens, speaker_embedding, language_id, master_hidden)) {
        return false;
    }

    std::string sampler_err;
    LlamaSampler talker_sampler;
    if (!talker_sampler.init(
            temperature,
            top_p,
            top_k,
            0.0f,
            repetition_penalty,
            0.0f,
            0.0f,
            128,
            (uint32_t) std::chrono::high_resolution_clock::now().time_since_epoch().count(),
            sampler_err)) {
        error_msg_ = sampler_err;
        return false;
    }

    const int32_t allow_tokens[3] = {kCodecEos, kCodecPad, kCodecBos};
    for (int32_t step = 0; step < max_frames; ++step) {
        int32_t code0_token = -1;
        if (!sample_with_mask(
                talker_ctx_,
                talker_model_,
                talker_sampler,
                0,
                2048,
                allow_tokens,
                3,
                code0_token)) {
            return false;
        }
        talker_sampler.accept(code0_token);

        if (code0_token == kCodecEos || code0_token == talker_model_.eos_id()) {
            break;
        }
        if (code0_token < 0 || code0_token >= 2048) {
            error_msg_ = "Talker sampled invalid code_0 token";
            return false;
        }

        std::vector<int32_t> frame_codes;
        std::vector<float> audio_sum;
        if (!predict_frame(master_hidden, code0_token, temperature, top_p, top_k, frame_codes, audio_sum)) {
            return false;
        }
        if ((int32_t) frame_codes.size() != 16) {
            error_msg_ = "Predictor did not produce 16-code frame";
            return false;
        }

        output_codes.insert(output_codes.end(), frame_codes.begin(), frame_codes.end());
        if (!run_decode_step(audio_sum, master_hidden)) {
            return false;
        }
    }

    if (output_codes.empty()) {
        error_msg_ = "No speech codes generated";
        return false;
    }
    return true;
}

} // namespace qwen3_tts
