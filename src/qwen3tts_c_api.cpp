/* qwen3tts_c_api.cpp — C API wrapper for Nim FFI.
 *
 * Wraps qwen3_tts::Qwen3TTS C++ class in a C-linkage API.
 * Synthesis calls use @autoreleasepool on macOS to drain Metal
 * Objective-C objects when called from background threads. */

#include "qwen3_tts.h"

#ifdef __APPLE__
#include <objc/objc.h>
#include <objc/message.h>
// Minimal autorelease pool without importing Foundation
static void * new_autorelease_pool() {
    id pool = ((id(*)(id, SEL))objc_msgSend)(
        (id)objc_getClass("NSAutoreleasePool"),
        sel_registerName("new"));
    return (void *)pool;
}
static void drain_autorelease_pool(void * pool) {
    ((void(*)(id, SEL))objc_msgSend)((id)pool, sel_registerName("drain"));
}
#define AUTORELEASE_BEGIN void * _pool = new_autorelease_pool();
#define AUTORELEASE_END   drain_autorelease_pool(_pool);
#else
#define AUTORELEASE_BEGIN
#define AUTORELEASE_END
#endif

#include <cstring>
#include <cstdlib>

// Match the C API header types (qwen3tts_c_api.h)
struct Qwen3TtsParams {
    int32_t max_audio_tokens;
    float   temperature;
    float   top_p;
    int32_t top_k;
    int32_t n_threads;
    float   repetition_penalty;
    int32_t language_id;
};

struct Qwen3TtsAudio {
    const float * samples;
    int32_t n_samples;
    int32_t sample_rate;
};

// Opaque handle — backs the C typedef
struct Qwen3Tts {
    qwen3_tts::Qwen3TTS engine;
    std::string last_error;
};

// Helper: convert C params to C++ params
static qwen3_tts::tts_params to_cpp_params(const Qwen3TtsParams * p) {
    qwen3_tts::tts_params params;
    if (p) {
        params.max_audio_tokens  = p->max_audio_tokens;
        params.temperature       = p->temperature;
        params.top_p             = p->top_p;
        params.top_k             = p->top_k;
        params.n_threads         = p->n_threads;
        params.repetition_penalty = p->repetition_penalty;
        params.language_id       = p->language_id;
    }
    return params;
}

// Helper: convert C++ result to heap-allocated C audio struct
static Qwen3TtsAudio * to_c_audio(const qwen3_tts::tts_result & result) {
    if (!result.success || result.audio.empty()) {
        return nullptr;
    }
    auto * out = new Qwen3TtsAudio;
    auto * buf = new float[result.audio.size()];
    std::memcpy(buf, result.audio.data(), result.audio.size() * sizeof(float));
    out->samples     = buf;
    out->n_samples   = (int32_t)result.audio.size();
    out->sample_rate = result.sample_rate;
    return out;
}

// ============================================================
// C API implementation
// ============================================================

extern "C" {

void qwen3_tts_default_params(Qwen3TtsParams * params) {
    if (!params) return;
    params->max_audio_tokens  = 4096;
    params->temperature       = 0.9f;
    params->top_p             = 1.0f;
    params->top_k             = 50;
    params->n_threads         = 4;
    params->repetition_penalty = 1.05f;
    params->language_id       = 2050; // en
}

Qwen3Tts * qwen3_tts_create(const char * model_dir, int32_t n_threads) {
    if (!model_dir) return nullptr;
    auto * tts = new Qwen3Tts;
    (void)n_threads; // thread count is set per-call via params
    if (!tts->engine.load_models(model_dir)) {
        tts->last_error = tts->engine.get_error();
        delete tts;
        return nullptr;
    }
    return tts;
}

int qwen3_tts_is_loaded(const Qwen3Tts * tts) {
    return (tts && tts->engine.is_loaded()) ? 1 : 0;
}

Qwen3TtsAudio * qwen3_tts_synthesize(
        Qwen3Tts * tts, const char * text,
        const Qwen3TtsParams * params) {
    if (!tts || !text) return nullptr;
    AUTORELEASE_BEGIN
    auto cpp_params = to_cpp_params(params);
    auto result = tts->engine.synthesize(text, cpp_params);
    if (!result.success) {
        tts->last_error = result.error_msg;
    }
    auto * out = to_c_audio(result);
    AUTORELEASE_END
    return out;
}

int32_t qwen3_tts_sample_rate(const Qwen3Tts * tts) {
    (void)tts;
    return 24000;
}

void qwen3_tts_free_audio(Qwen3TtsAudio * audio) {
    if (!audio) return;
    delete[] audio->samples;
    delete audio;
}

void qwen3_tts_destroy(Qwen3Tts * tts) {
    delete tts;
}

Qwen3TtsAudio * qwen3_tts_synthesize_with_voice_file(
        Qwen3Tts * tts, const char * text,
        const char * reference_audio_path,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !reference_audio_path) return nullptr;
    AUTORELEASE_BEGIN
    auto cpp_params = to_cpp_params(params);
    auto result = tts->engine.synthesize_with_voice(text, reference_audio_path, cpp_params);
    if (!result.success) {
        tts->last_error = result.error_msg;
    }
    auto * out = to_c_audio(result);
    AUTORELEASE_END
    return out;
}

Qwen3TtsAudio * qwen3_tts_synthesize_with_voice_samples(
        Qwen3Tts * tts, const char * text,
        const float * ref_samples, int32_t n_ref_samples,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !ref_samples || n_ref_samples <= 0) return nullptr;
    AUTORELEASE_BEGIN
    auto cpp_params = to_cpp_params(params);
    auto result = tts->engine.synthesize_with_voice(text, ref_samples, n_ref_samples, cpp_params);
    if (!result.success) {
        tts->last_error = result.error_msg;
    }
    auto * out = to_c_audio(result);
    AUTORELEASE_END
    return out;
}

int32_t qwen3_tts_extract_embedding_file(
        Qwen3Tts * tts, const char * reference_audio_path,
        float * embedding_out, int32_t max_size) {
    if (!tts || !reference_audio_path || !embedding_out || max_size <= 0) return -1;

    // Load WAV and resample to 24kHz
    std::vector<float> ref_samples;
    int ref_sample_rate;
    if (!qwen3_tts::load_audio_file(reference_audio_path, ref_samples, ref_sample_rate)) {
        tts->last_error = "Failed to load reference audio: " + std::string(reference_audio_path);
        return -1;
    }

    // Resample if needed (same logic as synthesize_with_voice)
    if (ref_sample_rate != 24000) {
        // Simple linear resampling
        double ratio = (double)ref_sample_rate / 24000;
        int output_len = (int)((double)ref_samples.size() / ratio);
        std::vector<float> resampled(output_len);
        for (int i = 0; i < output_len; ++i) {
            double src_idx = i * ratio;
            int idx0 = (int)src_idx;
            int idx1 = idx0 + 1;
            double frac = src_idx - idx0;
            if (idx1 >= (int)ref_samples.size()) {
                resampled[i] = ref_samples.back();
            } else {
                resampled[i] = (float)((1.0 - frac) * ref_samples[idx0] + frac * ref_samples[idx1]);
            }
        }
        ref_samples = std::move(resampled);
    }

    AUTORELEASE_BEGIN
    std::vector<float> embedding;
    if (!tts->engine.extract_speaker_embedding(ref_samples.data(), (int32_t)ref_samples.size(), embedding)) {
        tts->last_error = tts->engine.get_error();
        AUTORELEASE_END
        return -1;
    }
    AUTORELEASE_END

    int32_t emb_size = (int32_t)embedding.size();
    if (emb_size > max_size) emb_size = max_size;
    std::memcpy(embedding_out, embedding.data(), emb_size * sizeof(float));
    return emb_size;
}

Qwen3TtsAudio * qwen3_tts_synthesize_with_embedding(
        Qwen3Tts * tts, const char * text,
        const float * embedding, int32_t embedding_size,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !embedding || embedding_size <= 0) return nullptr;
    AUTORELEASE_BEGIN
    auto cpp_params = to_cpp_params(params);
    auto result = tts->engine.synthesize_with_embedding(text, embedding, embedding_size, cpp_params);
    if (!result.success) {
        tts->last_error = result.error_msg;
    }
    auto * out = to_c_audio(result);
    AUTORELEASE_END
    return out;
}

const char * qwen3_tts_get_error(const Qwen3Tts * tts) {
    if (!tts) return "";
    return tts->last_error.c_str();
}

} // extern "C"
