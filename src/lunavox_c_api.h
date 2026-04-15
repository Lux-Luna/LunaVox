/* lunavox_c_api.h — stable C ABI facade around lunavox::Engine (Nim/Python FFI). */
#ifndef LUNAVOX_C_API_H
#define LUNAVOX_C_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct LunavoxEngine LunavoxEngine;

/* Generation parameters */
typedef struct LunavoxSynthesisParams {
    int32_t max_audio_tokens;    /* default: 0 (use model_profile default_max_new_tokens) */
    float   temperature;         /* default: 0.9, 0=greedy */
    float   top_p;               /* default: 1.0 */
    int32_t top_k;               /* default: 50, 0=disabled */
    int32_t n_threads;           /* default: 4 */
    float   repetition_penalty;  /* default: 1.05 */
    int32_t language_id;         /* 2050=en, 2058=ja, 2055=zh, etc. */
    const char* ref_text;        /* Optional reference text for cloned voice */
} LunavoxSynthesisParams;

/* Generated audio + per-run stats. Each synthesize call returns a fresh
 * LunavoxAudio; all timing values are milliseconds, memory values are
 * bytes. `audio_duration_ms` is derived from n_samples / sample_rate.
 * `rtf` is total_ms / audio_duration_ms (lower is better; < 1.0 is
 * faster than real time). Fields the backend does not measure are 0. */
typedef struct LunavoxAudio {
    const float* samples;          /* PCM float32 mono */
    int32_t n_samples;
    int32_t sample_rate;           /* always 24000 */

    /* Timing (ms) */
    int64_t t_tokenize_ms;
    int64_t t_encode_ms;
    int64_t t_generate_ms;
    int64_t t_decode_ms;
    int64_t t_total_ms;

    /* Derived */
    int64_t audio_duration_ms;
    float   rtf;

    /* Memory snapshots (bytes) */
    uint64_t rss_peak_bytes;
    uint64_t rss_end_bytes;
} LunavoxAudio;

/* Fill params with defaults */
void lunavox_default_params(LunavoxSynthesisParams* params);

/* Engine-level cold-path timings. Valid after lunavox_create() returns.
 * Both are milliseconds. `warmup_ms` is 0 if warmup was disabled at load. */
int64_t lunavox_last_load_ms(const LunavoxEngine* tts);
int64_t lunavox_last_warmup_ms(const LunavoxEngine* tts);

/* Log level mirrors lunavox::LogLevel. USER is always printed. */
typedef enum LunavoxLogLevel {
    LUNAVOX_LOG_DEBUG = 0,
    LUNAVOX_LOG_INFO  = 1,
    LUNAVOX_LOG_WARN  = 2,
    LUNAVOX_LOG_ERROR = 3,
    LUNAVOX_LOG_USER  = 4
} LunavoxLogLevel;

typedef void (*LunavoxLogCallback)(LunavoxLogLevel level, const char* message, void* user_data);

/* Install a process-wide log callback. Passing NULL removes the callback.
 * The callback is invoked in addition to the normal file/console sinks —
 * consumers (e.g. the Python binding) can forward C++ logs into their own
 * logger without parsing stdout. */
void lunavox_set_log_callback(LunavoxLogCallback cb, void* user_data);

/* Streaming audio chunk callback. Fires from the decoder worker thread
 * as each PCM chunk becomes available during synthesis.
 *
 *   samples     — pointer to the newly produced float32 PCM slice.
 *                 Valid only for the duration of the callback; copy it
 *                 if you need to keep it past the callback return.
 *   n_samples   — number of samples in this slice. May be zero on the
 *                 terminal chunk (is_last=1) if no new audio was produced.
 *   is_last     — 1 on the final chunk of a synthesis call, 0 otherwise.
 *   user_data   — opaque pointer passed to lunavox_synthesize_streaming.
 *
 * The callback must not block for long: any stall here stalls synthesis
 * because it runs inline on the decoder worker thread. The Python
 * binding handles this by pushing slices into a bounded queue and
 * having the main thread drain them. */
typedef void (*LunavoxAudioChunkCallback)(
    const float* samples,
    int32_t n_samples,
    int32_t is_last,
    void* user_data);

/* Create TTS engine and load models from directory.
 * New default layout expects:
 *   lunavox_talker*.gguf
 *   lunavox_predictor*.gguf
 *   lunavox_decoder*.onnx
 * Optional:
 *   lunavox_speaker_encoder*.onnx
 *   lunavox_codec_encoder*.onnx
 *   embeddings/
 *   tokenizer.json
 * Returns NULL on failure. */
LunavoxEngine* lunavox_create(const char* model_dir, int32_t n_threads);

/* Check if models are loaded */
int lunavox_is_loaded(const LunavoxEngine* tts);

/* Synthesize text to audio. Returns NULL on failure.
 * Caller must free with lunavox_free_audio(). */
LunavoxAudio* lunavox_synthesize(
    LunavoxEngine* tts,
    const char* text,
    const LunavoxSynthesisParams* params);

/* Get sample rate (always 24000) */
int32_t lunavox_sample_rate(const LunavoxEngine* tts);

/* Free generated audio */
void lunavox_free_audio(LunavoxAudio* audio);

/* Destroy TTS engine */
void lunavox_destroy(LunavoxEngine* tts);

/* Synthesize with voice cloning from WAV file.
 * reference_audio_path: path to reference WAV (24kHz mono recommended).
 * Returns NULL on failure. Caller must free with lunavox_free_audio(). */
LunavoxAudio* lunavox_synthesize_with_voice_file(
    LunavoxEngine* tts,
    const char* text,
    const char* reference_audio_path,
    const LunavoxSynthesisParams* params);

/* Synthesize with voice cloning from raw samples.
 * ref_samples: 24kHz mono float32 normalized to [-1, 1].
 * Returns NULL on failure. Caller must free with lunavox_free_audio(). */
LunavoxAudio* lunavox_synthesize_with_voice_samples(
    LunavoxEngine* tts,
    const char* text,
    const float* ref_samples,
    int32_t n_ref_samples,
    const LunavoxSynthesisParams* params);

/* Extract speaker embedding from WAV file (for caching).
 * embedding_out: caller-allocated buffer for the embedding.
 * max_size: size of embedding_out in floats.
 * Returns the actual embedding size (typically 2048 for Qwen3-TTS), or -1 on failure. */
int32_t lunavox_extract_embedding_file(
    LunavoxEngine* tts,
    const char* reference_audio_path,
    float* embedding_out,
    int32_t max_size);

/* Synthesize with pre-computed speaker embedding (skips encoder).
 * embedding: speaker embedding from lunavox_extract_embedding_file().
 * embedding_size: must match the size returned by extract.
 * Returns NULL on failure. Caller must free with lunavox_free_audio(). */
LunavoxAudio* lunavox_synthesize_with_embedding(
    LunavoxEngine* tts,
    const char* text,
    const float* embedding,
    int32_t embedding_size,
    const LunavoxSynthesisParams* params);

/* Custom Voice mode: fixed speaker embedding + optional instruct.
 * speaker: speaker name from model_profile aliases (e.g. "Vivian").
 * instruct: optional style/emotion instruction; may be NULL or empty.
 * Returns NULL on failure. Caller must free with lunavox_free_audio(). */
LunavoxAudio* lunavox_synthesize_custom(
    LunavoxEngine* tts,
    const char* text,
    const char* speaker,
    const char* instruct,
    const LunavoxSynthesisParams* params);

/* Voice Design mode: instruct-only, no reference audio or speaker ID.
 * instruct: required non-empty voice design description.
 * Returns NULL on failure. Caller must free with lunavox_free_audio(). */
LunavoxAudio* lunavox_synthesize_design(
    LunavoxEngine* tts,
    const char* text,
    const char* instruct,
    const LunavoxSynthesisParams* params);

/* Streaming base-mode synthesis. Functionally equivalent to
 * lunavox_synthesize but fires `chunk_cb` from the decoder worker
 * thread as each PCM slice becomes available, giving callers a
 * progressive audio stream while still returning the cumulative
 * LunavoxAudio* on completion. The callback receives both the
 * intermediate chunks and (exactly once) a terminal chunk with
 * is_last=1 — callers can use that as the end-of-stream signal.
 *
 * Returns NULL on failure. Caller must free with lunavox_free_audio().
 * This is the streaming entry used by `lunavox serve` and the
 * FastAPI WebSocket endpoint. */
LunavoxAudio* lunavox_synthesize_streaming(
    LunavoxEngine* tts,
    const char* text,
    const LunavoxSynthesisParams* params,
    LunavoxAudioChunkCallback chunk_cb,
    void* user_data);

/* Streaming voice-clone synthesis (reference WAV or pre-computed
 * .json feature file). Same semantics as lunavox_synthesize_streaming;
 * clone pipeline runs first, then the decoder-chunk callback fires as
 * each PCM slice becomes available. */
LunavoxAudio* lunavox_synthesize_with_voice_file_streaming(
    LunavoxEngine* tts,
    const char* text,
    const char* reference_audio_path,
    const LunavoxSynthesisParams* params,
    LunavoxAudioChunkCallback chunk_cb,
    void* user_data);

/* Streaming custom-voice synthesis. speaker must resolve to a
 * catalog entry in the active model profile; instruct may be NULL
 * or empty for tone-neutral generation. */
LunavoxAudio* lunavox_synthesize_custom_streaming(
    LunavoxEngine* tts,
    const char* text,
    const char* speaker,
    const char* instruct,
    const LunavoxSynthesisParams* params,
    LunavoxAudioChunkCallback chunk_cb,
    void* user_data);

/* Streaming voice-design synthesis. instruct is the required
 * non-empty voice description; no speaker or reference audio is
 * accepted. */
LunavoxAudio* lunavox_synthesize_design_streaming(
    LunavoxEngine* tts,
    const char* text,
    const char* instruct,
    const LunavoxSynthesisParams* params,
    LunavoxAudioChunkCallback chunk_cb,
    void* user_data);

/* Get last error message (or empty string) */
const char* lunavox_get_error(const LunavoxEngine* tts);

#ifdef __cplusplus
}
#endif

#endif /* LUNAVOX_C_API_H */
