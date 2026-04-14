/* lunavox_c_api.cpp — C API wrapper for Nim/Python FFI.
 *
 * Wraps lunavox::Engine C++ class in a C-linkage API.
 * Synthesis calls wrap each public entry point in a
 * platform::AutoreleasePoolScope so the macOS build drains Metal
 * Objective-C temporaries when called from background threads. */

#include "lunavox_engine.h"
#include "lunavox_c_api.h"
#include "audio_io.h"
#include "logger.h"
#include "platform_utils.h"

#include <cstring>
#include <cstdlib>
#include <mutex>
#include <string>

// Last error captured when lunavox_create() fails and there is no engine
// handle to attach it to. Protected because multiple Python threads can
// call create() concurrently and the error should survive until read.
static std::mutex          g_last_create_error_mu;
static std::string         g_last_create_error;

static void set_create_error(const std::string & msg) {
    std::lock_guard<std::mutex> guard(g_last_create_error_mu);
    g_last_create_error = msg;
}

// Opaque handle — backs the C typedef
struct LunavoxEngine {
    lunavox::Engine engine;
    std::string last_error;
};

// Helper: convert C params to C++ params
static lunavox::tts_params to_cpp_params(const LunavoxSynthesisParams * p) {
    lunavox::tts_params params;
    if (p) {
        params.max_audio_tokens  = p->max_audio_tokens;
        params.temperature       = p->temperature;
        params.top_p             = p->top_p;
        params.top_k             = p->top_k;
        params.n_threads         = p->n_threads;
        params.repetition_penalty = p->repetition_penalty;
        params.language_id       = p->language_id;
        if (p->ref_text) {
            params.ref_text = p->ref_text;
        }
    }
    return params;
}

// Helper: convert C++ result to heap-allocated C audio struct with embedded stats.
static LunavoxAudio * to_c_audio(const lunavox::tts_result & result) {
    if (!result.success || result.audio.empty()) {
        return nullptr;
    }
    auto * out = new LunavoxAudio;
    auto * buf = new float[result.audio.size()];
    std::memcpy(buf, result.audio.data(), result.audio.size() * sizeof(float));
    out->samples        = buf;
    out->n_samples      = (int32_t) result.audio.size();
    out->sample_rate    = result.sample_rate;
    out->t_tokenize_ms  = result.t_tokenize_ms;
    out->t_encode_ms    = result.t_encode_ms;
    out->t_generate_ms  = result.t_generate_ms;
    out->t_decode_ms    = result.t_decode_ms;
    out->t_total_ms     = result.t_total_ms;
    // audio_duration_ms is derived from sample count so it stays correct
    // even if someone post-trims the PCM buffer from the caller side.
    out->audio_duration_ms = (result.sample_rate > 0)
        ? (int64_t) ((double) out->n_samples * 1000.0 / (double) result.sample_rate)
        : 0;
    out->rtf = (out->audio_duration_ms > 0)
        ? (float) ((double) result.t_total_ms / (double) out->audio_duration_ms)
        : 0.0f;
    out->rss_peak_bytes = result.mem_rss_peak_bytes;
    out->rss_end_bytes  = result.mem_rss_end_bytes;
    return out;
}

// ============================================================
// C API implementation
// ============================================================

extern "C" {

void lunavox_default_params(LunavoxSynthesisParams * params) {
    if (!params) return;
    params->max_audio_tokens  = 0;
    params->temperature       = 0.9f;
    params->top_p             = 1.0f;
    params->top_k             = 50;
    params->n_threads         = 4;
    params->repetition_penalty = 1.05f;
    params->language_id       = -1; // Auto(None)
    params->ref_text          = nullptr;
}

LunavoxEngine * lunavox_create(const char * model_dir, int32_t n_threads) {
    if (!model_dir) {
        set_create_error("model_dir is null");
        return nullptr;
    }
    auto * tts = new LunavoxEngine;
    const int32_t load_threads = n_threads > 0 ? n_threads : 4;
    if (!tts->engine.load_models(model_dir, load_threads)) {
        const std::string err = tts->engine.get_error();
        set_create_error(err);
        // Also log so consumers that installed a log callback but haven't
        // checked get_error() still see something.
        lunavox::Logger::instance().log(
            lunavox::LogLevel::ERROR_LOG,
            "lunavox_create failed for %s: %s",
            model_dir,
            err.c_str());
        delete tts;
        return nullptr;
    }
    return tts;
}

int lunavox_is_loaded(const LunavoxEngine * tts) {
    return (tts && tts->engine.is_loaded()) ? 1 : 0;
}

LunavoxAudio * lunavox_synthesize(
        LunavoxEngine * tts, const char * text,
        const LunavoxSynthesisParams * params) {
    if (!tts || !text) return nullptr;
    lunavox::platform::AutoreleasePoolScope _pool_scope;
    auto cpp_params = to_cpp_params(params);
    auto result = tts->engine.synthesize(text, cpp_params);
    if (!result.success) {
        tts->last_error = result.error_msg;
    }
    return to_c_audio(result);
}

int32_t lunavox_sample_rate(const LunavoxEngine * tts) {
    (void)tts;
    return 24000;
}

void lunavox_free_audio(LunavoxAudio * audio) {
    if (!audio) return;
    delete[] audio->samples;
    delete audio;
}

void lunavox_destroy(LunavoxEngine * tts) {
    delete tts;
}

LunavoxAudio * lunavox_synthesize_with_voice_file(
        LunavoxEngine * tts, const char * text,
        const char * reference_audio_path,
        const LunavoxSynthesisParams * params) {
    if (!tts || !text || !reference_audio_path) return nullptr;
    lunavox::platform::AutoreleasePoolScope _pool_scope;
    auto cpp_params = to_cpp_params(params);
    auto result = tts->engine.synthesize_with_voice(text, reference_audio_path, cpp_params);
    if (!result.success) {
        tts->last_error = result.error_msg;
    }
    return to_c_audio(result);
}

LunavoxAudio * lunavox_synthesize_with_voice_samples(
        LunavoxEngine * tts, const char * text,
        const float * ref_samples, int32_t n_ref_samples,
        const LunavoxSynthesisParams * params) {
    if (!tts || !text || !ref_samples || n_ref_samples <= 0) return nullptr;
    lunavox::platform::AutoreleasePoolScope _pool_scope;
    auto cpp_params = to_cpp_params(params);
    auto result = tts->engine.synthesize_with_voice(text, ref_samples, n_ref_samples, cpp_params);
    if (!result.success) {
        tts->last_error = result.error_msg;
    }
    return to_c_audio(result);
}

int32_t lunavox_extract_embedding_file(
        LunavoxEngine * tts, const char * reference_audio_path,
        float * embedding_out, int32_t max_size) {
    if (!tts || !reference_audio_path || !embedding_out || max_size <= 0) return -1;

    // Load WAV and resample to 24kHz
    std::vector<float> ref_samples;
    int ref_sample_rate;
    if (!lunavox::load_audio_file(reference_audio_path, ref_samples, ref_sample_rate)) {
        tts->last_error = "Failed to load reference audio: " + std::string(reference_audio_path);
        return -1;
    }

    // Resample if needed (same quality path as synthesize_with_voice)
    if (ref_sample_rate != 24000) {
        std::vector<float> resampled;
        if (!lunavox::resample_windowed_sinc(
                ref_samples.data(),
                (int32_t) ref_samples.size(),
                ref_sample_rate,
                resampled,
                24000)) {
            tts->last_error = "Failed to resample reference audio to 24kHz";
            return -1;
        }
        ref_samples = std::move(resampled);
    }

    std::vector<float> embedding;
    {
        lunavox::platform::AutoreleasePoolScope _pool_scope;
        if (!tts->engine.extract_speaker_embedding(ref_samples.data(), (int32_t)ref_samples.size(), embedding)) {
            tts->last_error = tts->engine.get_error();
            return -1;
        }
    }

    int32_t emb_size = (int32_t)embedding.size();
    if (emb_size > max_size) emb_size = max_size;
    std::memcpy(embedding_out, embedding.data(), emb_size * sizeof(float));
    return emb_size;
}

LunavoxAudio * lunavox_synthesize_with_embedding(
        LunavoxEngine * tts, const char * text,
        const float * embedding, int32_t embedding_size,
        const LunavoxSynthesisParams * params) {
    if (!tts || !text || !embedding || embedding_size <= 0) return nullptr;
    lunavox::platform::AutoreleasePoolScope _pool_scope;
    auto cpp_params = to_cpp_params(params);
    auto result = tts->engine.synthesize_with_embedding(text, embedding, embedding_size, cpp_params);
    if (!result.success) {
        tts->last_error = result.error_msg;
    }
    return to_c_audio(result);
}

LunavoxAudio * lunavox_synthesize_custom(
        LunavoxEngine * tts, const char * text,
        const char * speaker, const char * instruct,
        const LunavoxSynthesisParams * params) {
    if (!tts || !text || !speaker) return nullptr;
    lunavox::platform::AutoreleasePoolScope _pool_scope;
    auto cpp_params = to_cpp_params(params);
    auto result = tts->engine.synthesize_custom(
        text,
        std::string(speaker),
        std::string(instruct ? instruct : ""),
        cpp_params);
    if (!result.success) {
        tts->last_error = result.error_msg;
    }
    return to_c_audio(result);
}

LunavoxAudio * lunavox_synthesize_design(
        LunavoxEngine * tts, const char * text,
        const char * instruct,
        const LunavoxSynthesisParams * params) {
    if (!tts || !text || !instruct) return nullptr;
    lunavox::platform::AutoreleasePoolScope _pool_scope;
    auto cpp_params = to_cpp_params(params);
    auto result = tts->engine.synthesize_design(text, std::string(instruct), cpp_params);
    if (!result.success) {
        tts->last_error = result.error_msg;
    }
    return to_c_audio(result);
}

const char * lunavox_get_error(const LunavoxEngine * tts) {
    if (tts) return tts->last_error.c_str();
    // Null handle means the caller is asking about the most recent
    // lunavox_create failure. Return the stashed string directly — we do
    // not lock because the caller is single-threaded with respect to its
    // own create() call (and the buffer is stable across reads).
    return g_last_create_error.c_str();
}

int64_t lunavox_last_load_ms(const LunavoxEngine * tts) {
    return tts ? tts->engine.last_load_ms() : 0;
}

int64_t lunavox_last_warmup_ms(const LunavoxEngine * tts) {
    return tts ? tts->engine.last_warmup_ms() : 0;
}

// The logger only knows about a narrow C-style sink with an int level; we
// store the caller's callback + user_data in process-scope statics and use
// a trampoline to translate the internal LogLevel enum into the C API
// enum. Values are contiguous 0..4 so they already line up, but we switch
// explicitly to stay decoupled from reordering.
static LunavoxLogCallback g_log_cb      = nullptr;
static void *             g_log_cb_data = nullptr;

static void lunavox_c_api_log_trampoline(int internal_level, const char * msg) {
    auto cb = g_log_cb;
    if (!cb || !msg) return;
    LunavoxLogLevel lvl;
    switch ((lunavox::LogLevel) internal_level) {
        case lunavox::LogLevel::DEBUG_LOG: lvl = LUNAVOX_LOG_DEBUG; break;
        case lunavox::LogLevel::INFO_LOG:  lvl = LUNAVOX_LOG_INFO;  break;
        case lunavox::LogLevel::WARN_LOG:  lvl = LUNAVOX_LOG_WARN;  break;
        case lunavox::LogLevel::ERROR_LOG: lvl = LUNAVOX_LOG_ERROR; break;
        case lunavox::LogLevel::USER_LOG:  lvl = LUNAVOX_LOG_USER;  break;
        default:                           lvl = LUNAVOX_LOG_INFO;  break;
    }
    cb(lvl, msg, g_log_cb_data);
}

void lunavox_set_log_callback(LunavoxLogCallback cb, void * user_data) {
    g_log_cb      = cb;
    g_log_cb_data = user_data;
    lunavox::Logger::instance().set_external_sink(cb ? lunavox_c_api_log_trampoline : nullptr);
}

} // extern "C"
