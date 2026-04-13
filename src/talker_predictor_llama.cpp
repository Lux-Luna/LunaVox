#include "talker_predictor_llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>

namespace qwen3_tts {

namespace {

static inline void add_vec(float * dst, const float * src, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}

static inline uint32_t choose_seed(int32_t requested_seed, uint32_t salt = 0) {
    if (requested_seed >= 0) {
        return (uint32_t) requested_seed + salt;
    }
    return (uint32_t) std::chrono::high_resolution_clock::now().time_since_epoch().count() + salt;
}

static inline int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

} // namespace

bool TalkerPredictorLlama::load(
    const std::string & lib_dir,
    const std::string & talker_model_path,
    const std::string & predictor_model_path,
    const AssetsManager & assets,
    const RuntimeModelProfile & profile,
    int32_t n_threads) {
    unload();
    assets_ = &assets;
    profile_ = profile;

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

    std::string profile_err;
    if (!profile_.is_valid(&profile_err)) {
        error_msg_ = "Invalid runtime profile: " + profile_err;
        return false;
    }
    codebook_vocab_size_ = profile_.codec_id_end - profile_.codec_id_start;
    if (codebook_vocab_size_ <= 0) {
        error_msg_ = "Invalid codec vocabulary size in runtime profile";
        return false;
    }
    if (predictor_model_.n_vocab() != profile_.predictor_vocab_size) {
        error_msg_ = "Predictor vocab size mismatch with runtime profile";
        return false;
    }

    talker_ctx_cap_ = profile_.talker_n_ctx;
    talker_ctx_train_ = profile_.talker_n_ctx_train;
    n_threads_ = n_threads > 0 ? n_threads : 4;
    if (talker_ctx_cap_ <= 0) {
        error_msg_ = "Talker context size is invalid in runtime profile";
        return false;
    }
    if (talker_ctx_train_ <= 0) {
        error_msg_ = "Talker train context size is invalid in runtime profile";
        return false;
    }
    if (talker_ctx_cap_ > talker_ctx_train_) {
        error_msg_ = "talker_n_ctx in profile exceeds talker_n_ctx_train";
        return false;
    }
    if (talker_model_.n_ctx_train() > 0) {
        if (talker_model_.n_ctx_train() < talker_ctx_cap_) {
            error_msg_ = "Talker context cap in profile exceeds model metadata";
            return false;
        }
        if (talker_model_.n_ctx_train() < talker_ctx_train_) {
            error_msg_ = "Talker train context in profile exceeds model metadata";
            return false;
        }
    }

    const int32_t predictor_n_ctx = profile_.predictor_n_ctx;
    if (predictor_n_ctx <= 0) {
        error_msg_ = "Predictor context size is invalid in runtime profile";
        return false;
    }
    if (predictor_model_.n_ctx_train() > 0 && predictor_model_.n_ctx_train() < predictor_n_ctx) {
        error_msg_ = "Predictor context in profile exceeds model metadata";
        return false;
    }

    if (!predictor_ctx_.init(
            predictor_model_,
            predictor_n_ctx,
            n_threads_,
            false,
            64,
            64,
            "predictor",
            error_msg_)) {
        return false;
    }

    if (!predictor_batch_.init(2, predictor_dim_, 1, error_msg_)) {
        return false;
    }

    loaded_ = true;
    cur_pos_ = 0;
    last_eos_step_ = -1;
    last_ctx_required_ = 0;
    last_ctx_allocated_ = 0;
    last_ctx_cap_ = talker_ctx_cap_;
    last_ctx_overflow_ = false;
    return true;
}

void TalkerPredictorLlama::unload() {
    loaded_ = false;
    error_msg_.clear();
    cur_pos_ = 0;
    last_eos_step_ = -1;
    predictor_batch_.free();
    talker_batch_.free();
    predictor_ctx_.free();
    talker_ctx_.free();
    predictor_model_.free();
    talker_model_.free();
    assets_ = nullptr;
    hidden_dim_ = 0;
    predictor_dim_ = 0;
    codebook_vocab_size_ = 2048;
    n_threads_ = 4;
    talker_ctx_cap_ = 0;
    talker_ctx_train_ = 0;
    talker_batch_cap_ = 0;
    last_ctx_required_ = 0;
    last_ctx_allocated_ = 0;
    last_ctx_cap_ = 0;
    last_ctx_overflow_ = false;
    profile_ = RuntimeModelProfile{};
}

int32_t TalkerPredictorLlama::estimate_prompt_tokens(
    const std::vector<int32_t> & text_tokens,
    const std::vector<int32_t> & ref_text_tokens,
    const std::vector<int32_t> & role_prefix_tokens,
    const std::vector<int32_t> & instruct_tokens,
    bool has_speaker_embedding,
    bool use_clone_icl,
    int32_t n_ref_frames,
    int32_t language_id) const {
    int64_t total = 0;
    total += (int64_t) instruct_tokens.size();
    total += (int64_t) role_prefix_tokens.size();
    total += (language_id >= 0) ? 4 : 3; // protocol tokens
    total += has_speaker_embedding ? 1 : 0;
    total += 1; // tts_bos + codec_pad bridge

    if (use_clone_icl) {
        total += (int64_t) ref_text_tokens.size();
        total += (int64_t) text_tokens.size();
        total += 1; // tts_eos + codec_pad
        total += 1; // tts_pad + codec_bos
        total += std::max(0, n_ref_frames);
    } else {
        total += (int64_t) text_tokens.size();
        total += 1; // tts_eos + codec_pad
        total += 1; // tts_pad + codec_bos
    }

    if (total <= 0) {
        return 0;
    }
    if (total > (int64_t) std::numeric_limits<int32_t>::max()) {
        return std::numeric_limits<int32_t>::max();
    }
    return (int32_t) total;
}

bool TalkerPredictorLlama::ensure_talker_runtime(int32_t request_ctx) {
    if (request_ctx <= 0) {
        error_msg_ = "Invalid talker request context";
        return false;
    }
    if (talker_ctx_cap_ <= 0) {
        error_msg_ = "Invalid talker context cap";
        return false;
    }
    if (request_ctx > talker_ctx_cap_) {
        error_msg_ = "Requested talker context exceeds configured cap";
        return false;
    }

    const bool need_new_ctx = !talker_ctx_.is_ready() || talker_ctx_.n_ctx() < request_ctx;
    if (need_new_ctx) {
        talker_batch_.free();
        talker_batch_cap_ = 0;
        talker_ctx_.free();
        if (!talker_ctx_.init(
                talker_model_,
                request_ctx,
                n_threads_,
                true,
                2048,
                512,
                "talker",
                error_msg_)) {
            return false;
        }
    }

    const int64_t desired_batch_cap64 = (int64_t) talker_ctx_.n_ctx() * 4LL;
    if (desired_batch_cap64 <= 0 || desired_batch_cap64 > (int64_t) std::numeric_limits<int32_t>::max()) {
        error_msg_ = "Talker batch capacity overflow";
        return false;
    }
    const int32_t desired_batch_cap = (int32_t) desired_batch_cap64;
    if (!talker_batch_.is_ready() || talker_batch_cap_ < desired_batch_cap) {
        talker_batch_.free();
        if (!talker_batch_.init(desired_batch_cap, hidden_dim_, 1, error_msg_)) {
            return false;
        }
        talker_batch_cap_ = desired_batch_cap;
    }
    return true;
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
    const int32_t lo = std::max(0, std::min(n_vocab, limit_start));
    const int32_t hi = std::max(lo, std::min(n_vocab, limit_end));
    // Snapshot allow-list logit values before masking so we can restore them.
    // n_allow is O(1) in practice (typically 0 or a handful).
    float allow_values[16];
    const int32_t n_allow_clamped = (n_allow > 0 && n_allow <= 16) ? n_allow : 0;
    for (int32_t j = 0; j < n_allow_clamped; ++j) {
        const int32_t t = allow_tokens[j];
        allow_values[j] = (t >= 0 && t < n_vocab) ? logits[t] : neg_inf;
    }
    std::fill(logits, logits + lo, neg_inf);
    std::fill(logits + hi, logits + n_vocab, neg_inf);
    for (int32_t j = 0; j < n_allow_clamped; ++j) {
        const int32_t t = allow_tokens[j];
        if (t >= 0 && t < n_vocab && (t < lo || t >= hi)) {
            logits[t] = allow_values[j];
        }
    }

    sampled_token = sampler.sample(ctx, -1);
    return true;
}

bool TalkerPredictorLlama::run_prefill(
    const std::vector<int32_t> & text_tokens,
    const std::vector<int32_t> & ref_text_tokens,
    const std::vector<int32_t> & role_prefix_tokens,
    const std::vector<int32_t> & instruct_tokens,
    const float * speaker_embedding,
    const int32_t * ref_codes,
    int32_t n_ref_frames,
    int32_t language_id,
    std::vector<float> & hidden_out) {
    if (!assets_ || text_tokens.empty()) {
        error_msg_ = "run_prefill invalid input";
        return false;
    }
    const float * tts_pad = assets_->text_row(profile_.tts_pad_id);
    if (!tts_pad) {
        error_msg_ = "Missing tts_pad row in text embedding table (profile token id)";
        return false;
    }

    // Official non-streaming prompt:
    // [instruct(optional)] + assistant prefix + protocol + (speaker) + text/clone body.
    std::vector<float> prompt_flat;
    std::vector<float> tmp_vec((size_t) hidden_dim_, 0.0f);
    auto append_sum = [&](const float * a, const float * b) {
        for (int32_t i = 0; i < hidden_dim_; ++i) {
            tmp_vec[(size_t) i] = a[i] + b[i];
        }
        prompt_flat.insert(prompt_flat.end(), tmp_vec.begin(), tmp_vec.end());
    };

    const float * think = assets_->codec_row(0, profile_.codec_think_id);
    const float * nothink = assets_->codec_row(0, profile_.codec_nothink_id);
    const float * think_bos = assets_->codec_row(0, profile_.codec_think_bos_id);
    const float * think_eos = assets_->codec_row(0, profile_.codec_think_eos_id);
    if (!think || !nothink || !think_bos || !think_eos) {
        error_msg_ = "Missing protocol embedding rows";
        return false;
    }

    // Instruct is injected before assistant role:
    // <|im_start|>user\n{instruct}<|im_end|>\n
    if (!instruct_tokens.empty()) {
        for (int32_t tid : instruct_tokens) {
            const float * row = assets_->text_row(tid);
            if (!row) {
                error_msg_ = "Instruct token embedding out of range";
                return false;
            }
            prompt_flat.insert(prompt_flat.end(), row, row + hidden_dim_);
        }
    }

    for (int32_t tid : role_prefix_tokens) {
        const float * role = assets_->text_row(tid);
        if (!role) {
            error_msg_ = "Role prefix token embedding out of range";
            return false;
        }
        prompt_flat.insert(prompt_flat.end(), role, role + hidden_dim_);
    }

    if (language_id >= 0) {
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

    const float * codec_pad = assets_->codec_row(0, profile_.codec_pad_id);
    const float * codec_bos = assets_->codec_row(0, profile_.codec_bos_id);
    const float * tts_eos = assets_->text_row(profile_.tts_eos_id);
    if (!codec_pad || !codec_bos || !tts_eos) {
        error_msg_ = "Missing codec/text embeddings for non-streaming prompt";
        return false;
    }

    // Official non-streaming prompt keeps a tts_bos+codec_pad bridge token
    // before text/ICL body.
    const float * tts_bos = assets_->text_row(profile_.tts_bos_id);
    if (!tts_bos) {
        error_msg_ = "Missing tts_bos row in text embedding table";
        return false;
    }
    append_sum(tts_bos, codec_pad);

    const bool use_clone_icl = ref_codes && n_ref_frames > 0 && !ref_text_tokens.empty();
    if (use_clone_icl) {
        // ICL body follows official non-streaming semantics:
        // text_embed = (ref_text + target_text + tts_eos) + codec_pad
        // audio_embed = [codec_bos, ref_code_sum...] + tts_pad
        for (int32_t tid : ref_text_tokens) {
            const float * row = assets_->text_row(tid);
            if (!row) {
                error_msg_ = "Reference text embedding row out of range in clone ICL";
                return false;
            }
            append_sum(row, codec_pad);
        }
        for (int32_t tid : text_tokens) {
            const float * row = assets_->text_row(tid);
            if (!row) {
                error_msg_ = "Target text embedding row out of range in clone ICL";
                return false;
            }
            append_sum(row, codec_pad);
        }
        append_sum(tts_eos, codec_pad);

        append_sum(tts_pad, codec_bos);
        for (int32_t t = 0; t < n_ref_frames; ++t) {
            std::fill(tmp_vec.begin(), tmp_vec.end(), 0.0f);
            for (int32_t q = 0; q < profile_.codec_num_codebooks; ++q) {
                const int32_t code = ref_codes[(size_t) t * (size_t) profile_.codec_num_codebooks + (size_t) q];
                const float * row = assets_->codec_row(q, code);
                if (!row) {
                    error_msg_ = "Reference code out of range in clone ICL";
                    return false;
                }
                for (int32_t i = 0; i < hidden_dim_; ++i) {
                    tmp_vec[(size_t) i] += row[i];
                }
            }
            for (int32_t i = 0; i < hidden_dim_; ++i) {
                tmp_vec[(size_t) i] += tts_pad[i];
            }
            prompt_flat.insert(prompt_flat.end(), tmp_vec.begin(), tmp_vec.end());
        }
    } else {
        for (int32_t tid : text_tokens) {
            const float * row = assets_->text_row(tid);
            if (!row) {
                error_msg_ = "Text embedding row out of range";
                return false;
            }
            append_sum(row, codec_pad);
        }
        append_sum(tts_eos, codec_pad);
        append_sum(tts_pad, codec_bos);
    }

    const int32_t n_prompt_tokens = (int32_t) (prompt_flat.size() / (size_t) hidden_dim_);
    const int32_t ctx_cap = talker_ctx_.n_ctx();
    if (ctx_cap <= 0) {
        error_msg_ = "Invalid talker context size";
        return false;
    }
    if (n_prompt_tokens >= ctx_cap) {
        error_msg_ = "Prompt is too long for talker context";
        return false;
    }
    std::vector<int32_t> pos((size_t) n_prompt_tokens * 4u);
    for (int32_t i = 0; i < n_prompt_tokens; ++i) {
        const int32_t p = cur_pos_ + i;
        pos[(size_t) i] = p;
        pos[(size_t) n_prompt_tokens + (size_t) i] = p;
        pos[(size_t) n_prompt_tokens * 2u + (size_t) i] = p;
        pos[(size_t) n_prompt_tokens * 3u + (size_t) i] = 0;
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
    const float * last = talker_ctx_.get_embeddings_ith(n_prompt_tokens - 1);
    if (!last) {
        error_msg_ = "Talker prefill embeddings pointer is null";
        return false;
    }
    std::memcpy(hidden_out.data(), last, (size_t) hidden_dim_ * sizeof(float));
    cur_pos_ += n_prompt_tokens;
    return true;
}

bool TalkerPredictorLlama::predict_frame(
    const std::vector<float> & master_hidden,
    int32_t code0,
    LlamaSampler & predictor_sampler,
    std::vector<int32_t> & frame_codes,
    std::vector<float> & audio_sum) {
    frame_codes.clear();
    frame_codes.reserve((size_t) profile_.codec_num_codebooks);
    audio_sum.assign((size_t) hidden_dim_, 0.0f);

    predictor_ctx_.clear_kv_cache();

    if (!assets_->project_to_predictor(master_hidden.data(), scratch_m_pred_)) {
        error_msg_ = "Failed to project master hidden for predictor";
        return false;
    }
    if (!assets_->codec_row_predictor(0, code0, scratch_emb0_pred_)) {
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

    if ((int32_t) scratch_prefill_.size() < predictor_dim_ * 2) {
        scratch_prefill_.resize((size_t) predictor_dim_ * 2);
    }
    std::memcpy(scratch_prefill_.data(), scratch_m_pred_.data(), (size_t) predictor_dim_ * sizeof(float));
    std::memcpy(scratch_prefill_.data() + predictor_dim_, scratch_emb0_pred_.data(), (size_t) predictor_dim_ * sizeof(float));
    int32_t ppos[2] = {0, 1};
    if (!predictor_batch_.set_embeddings(
            scratch_prefill_.data(),
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

    for (int32_t cs = 1; cs < profile_.codec_num_codebooks; ++cs) {
        const int32_t start = (cs - 1) * codebook_vocab_size_;
        const int32_t end = cs * codebook_vocab_size_;
        int32_t token_id = -1;
        if (!sample_with_mask(
                predictor_ctx_,
                predictor_model_,
                predictor_sampler,
                start,
                end,
                nullptr,
                0,
                token_id)) {
            return false;
        }
        const int32_t code = profile_.codec_id_start + (token_id - start);
        if (code < profile_.codec_id_start || code >= profile_.codec_id_end) {
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

        if (cs < profile_.codec_num_codebooks - 1) {
            if (!assets_->codec_row_predictor(cs, code, scratch_emb_next_)) {
                error_msg_ = "Failed to project predictor codec embedding";
                return false;
            }
            int32_t pos = cs + 1;
            if (!predictor_batch_.set_embeddings(
                    scratch_emb_next_.data(),
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

    const int64_t t_prep_start = now_ms();
    const float * tts_pad = assets_->text_row(profile_.tts_pad_id);
    if (!tts_pad) {
        error_msg_ = "Missing tts_pad row in text embedding table";
        return false;
    }

    std::vector<float> fused((size_t) hidden_dim_);
    for (int32_t i = 0; i < hidden_dim_; ++i) {
        fused[(size_t) i] = audio_sum[(size_t) i] + tts_pad[i];
    }

    int32_t pos[4] = {cur_pos_, cur_pos_, cur_pos_, 0};
    if (!talker_batch_.set_embeddings(
            fused.data(),
            1,
            hidden_dim_,
            pos,
            4,
            0,
            error_msg_)) {
        return false;
    }
    t_talker_post_prep_ms_ += now_ms() - t_prep_start;

    const int64_t t_dec_start = now_ms();
    if (talker_ctx_.decode(talker_batch_.raw()) != 0) {
        error_msg_ = "Talker decode_step failed";
        return false;
    }
    t_talker_decode_ms_ += now_ms() - t_dec_start;

    const int64_t t_copy_start = now_ms();
    const float * embd = talker_ctx_.get_embeddings_ith(0);
    if (!embd) {
        error_msg_ = "Talker step embeddings pointer is null";
        return false;
    }
    hidden_out.resize((size_t) hidden_dim_);
    std::memcpy(hidden_out.data(), embd, (size_t) hidden_dim_ * sizeof(float));
    ++cur_pos_;
    t_talker_post_copy_ms_ += now_ms() - t_copy_start;
    return true;
}

bool TalkerPredictorLlama::generate(
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
    std::vector<int32_t> & output_codes) {
    output_codes.clear();
    last_ctx_required_ = 0;
    last_ctx_allocated_ = 0;
    last_ctx_cap_ = talker_ctx_cap_;
    last_ctx_overflow_ = false;
    t_prefill_ms_ = 0;
    t_decode_loop_ms_ = 0;
    t_talker_post_ms_ = 0;
    t_predictor_sample_ms_ = 0;
    t_talker_decode_ms_ = 0;
    t_talker_post_prep_ms_ = 0;
    t_talker_post_copy_ms_ = 0;

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
    if (n_ref_frames < 0) {
        error_msg_ = "n_ref_frames must be non-negative";
        return false;
    }
    if ((ref_codes == nullptr) != (n_ref_frames == 0)) {
        error_msg_ = "ref_codes/ref_frames mismatch";
        return false;
    }

    const bool use_clone_icl = ref_codes && n_ref_frames > 0 && !ref_text_tokens.empty();
    const int32_t prompt_tokens = estimate_prompt_tokens(
        text_tokens,
        ref_text_tokens,
        role_prefix_tokens,
        instruct_tokens,
        speaker_embedding != nullptr,
        use_clone_icl,
        n_ref_frames,
        language_id);
    if (prompt_tokens <= 0) {
        error_msg_ = "Failed to estimate prompt token budget";
        return false;
    }
    const int64_t required_ctx64 = (int64_t) prompt_tokens + (int64_t) max_frames + 32LL;
    if (required_ctx64 > (int64_t) std::numeric_limits<int32_t>::max()) {
        error_msg_ = "Context requirement overflow";
        return false;
    }
    const int32_t required_ctx = (int32_t) required_ctx64;
    last_ctx_required_ = required_ctx;
    last_ctx_cap_ = talker_ctx_cap_;
    if (required_ctx > talker_ctx_cap_) {
        last_ctx_overflow_ = true;
        char buf[256];
        std::snprintf(
            buf,
            sizeof(buf),
            "Talker context overflow: required=%d exceeds cap=%d (prompt=%d, requested_max_frames=%d)",
            required_ctx,
            talker_ctx_cap_,
            prompt_tokens,
            max_frames);
        error_msg_ = buf;
        return false;
    }

    int32_t request_ctx = ((required_ctx + 255) / 256) * 256;
    request_ctx = std::max(512, request_ctx);
    request_ctx = std::min(talker_ctx_cap_, request_ctx);
    if (request_ctx < required_ctx) {
        last_ctx_overflow_ = true;
        char buf[256];
        std::snprintf(
            buf,
            sizeof(buf),
            "Talker context allocation failed to satisfy request: allocated=%d required=%d cap=%d",
            request_ctx,
            required_ctx,
            talker_ctx_cap_);
        error_msg_ = buf;
        return false;
    }

    if (!ensure_talker_runtime(request_ctx)) {
        return false;
    }
    last_ctx_allocated_ = talker_ctx_.n_ctx();

    talker_ctx_.clear_kv_cache();
    predictor_ctx_.clear_kv_cache();
    cur_pos_ = 0;
    last_eos_step_ = -1;

    std::vector<float> master_hidden;
    const int64_t t_prefill_start = now_ms();
    if (!run_prefill(
            text_tokens,
            ref_text_tokens,
            role_prefix_tokens,
            instruct_tokens,
            speaker_embedding,
            ref_codes,
            n_ref_frames,
            language_id,
            master_hidden)) {
        return false;
    }
    t_prefill_ms_ = now_ms() - t_prefill_start;

    const int32_t decode_budget = talker_ctx_.n_ctx() - cur_pos_;
    if (decode_budget <= 0) {
        error_msg_ = "No decode budget left after prompt prefill";
        return false;
    }
    if (max_frames > decode_budget) {
        char buf[256];
        std::snprintf(
            buf,
            sizeof(buf),
            "Talker decode budget overflow: requested_max_frames=%d exceeds remaining_ctx=%d",
            max_frames,
            decode_budget);
        error_msg_ = buf;
        return false;
    }

    std::string sampler_err;
    const uint32_t talker_rng_seed = choose_seed(talker_seed);
    const uint32_t predictor_rng_seed = (predictor_seed >= 0)
                                            ? (uint32_t) predictor_seed + 1337u
                                            : talker_rng_seed + 1337u;
    LlamaSampler talker_sampler;
    if (!talker_sampler.init(
            talker_temperature,
            talker_top_p,
            talker_top_k,
            0.0f,
            repetition_penalty,
            0.0f,
            0.0f,
            128,
            talker_rng_seed,
            sampler_err)) {
        error_msg_ = sampler_err;
        return false;
    }

    LlamaSampler predictor_sampler;
    const float pred_temp = predictor_do_sample ? predictor_temperature : 0.0f;
    const float pred_top_p = predictor_do_sample ? predictor_top_p : 1.0f;
    const int32_t pred_top_k = predictor_do_sample ? predictor_top_k : 0;
    if (!predictor_sampler.init(
            pred_temp,
            pred_top_p,
            pred_top_k,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            64,
            predictor_rng_seed,
            sampler_err)) {
        error_msg_ = sampler_err;
        return false;
    }

    int32_t min_new_frames_before_eos = std::max(0, profile_.min_new_tokens);
    const int32_t allow_special[3] = {
        profile_.codec_eos_id,
        profile_.codec_pad_id,
        profile_.codec_bos_id,
    };
    const int32_t n_allow_special = 3;
    int32_t code_limit_end = profile_.codec_id_end;
    if (profile_.suppress_from > profile_.codec_id_start) {
        code_limit_end = std::min(code_limit_end, profile_.suppress_from);
    }
    if (code_limit_end <= profile_.codec_id_start) {
        error_msg_ = "Invalid codec sampling range derived from runtime profile";
        return false;
    }

    int32_t generated_frames = 0;
    const int64_t t_loop_start = now_ms();
    for (int32_t step = 0; step < max_frames; ++step) {
        int32_t code0_token = -1;
        const int32_t * allow_tokens = (generated_frames >= min_new_frames_before_eos) ? allow_special : nullptr;
        const int32_t n_allow = (generated_frames >= min_new_frames_before_eos) ? n_allow_special : 0;
        if (!sample_with_mask(
                talker_ctx_,
                talker_model_,
                talker_sampler,
                profile_.codec_id_start,
                code_limit_end,
                allow_tokens,
                n_allow,
                code0_token)) {
            return false;
        }
        talker_sampler.accept(code0_token);

        const bool hit_codec_eos = (code0_token == profile_.codec_eos_id);
        if (hit_codec_eos && generated_frames >= min_new_frames_before_eos) {
            last_eos_step_ = step;
            break;
        }
        if (hit_codec_eos) {
            continue;
        }
        const bool is_codec_base = code0_token >= profile_.codec_id_start && code0_token < profile_.codec_id_end;
        const bool is_codec_special = code0_token == profile_.codec_pad_id || code0_token == profile_.codec_bos_id;
        if (!is_codec_base && !is_codec_special) {
            error_msg_ = "Talker sampled invalid code_0 token";
            return false;
        }

        std::vector<int32_t> frame_codes;
        std::vector<float> audio_sum;
        const int64_t t_pred_start = now_ms();
        if (!predict_frame(
                master_hidden,
                code0_token,
                predictor_sampler,
                frame_codes,
                audio_sum)) {
            return false;
        }
        t_predictor_sample_ms_ += now_ms() - t_pred_start;
        if ((int32_t) frame_codes.size() != profile_.codec_num_codebooks) {
            error_msg_ = "Predictor did not produce codec_num_codebooks frame";
            return false;
        }

        output_codes.insert(output_codes.end(), frame_codes.begin(), frame_codes.end());
        ++generated_frames;
        const int64_t t_talk_start = now_ms();
        if (!run_decode_step(audio_sum, master_hidden)) {
            return false;
        }
        t_talker_post_ms_ += now_ms() - t_talk_start;
    }
    t_decode_loop_ms_ = now_ms() - t_loop_start;

    if (output_codes.empty()) {
        error_msg_ = "No speech codes generated";
        return false;
    }
    return true;
}

} // namespace qwen3_tts
