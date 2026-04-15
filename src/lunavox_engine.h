#pragma once

#include "text_tokenizer.h"
#include "audio_decoder.h"
#include "assets_manager.h"
#include "talker_predictor_llama.h"
#include "model_profile.h"

#include <string>
#include <vector>
#include <functional>
#include <cstdint>
#include <mutex>

namespace lunavox {

// TTS generation parameters
struct tts_params {
    // Maximum number of audio tokens to generate.
    // <= 0 means use model_profile default_max_new_tokens.
    int32_t max_audio_tokens = 0;
    
    // Temperature for sampling (0 = greedy)
    float temperature = 0.6f;
    
    // Top-p sampling
    float top_p = 1.0f;
    
    // Top-k sampling (0 = disabled)
    int32_t top_k = 50;

    // Predictor stage sampling controls (Q1..Q15 generation).
    bool predictor_do_sample = true;
    float predictor_temperature = 0.6f;
    float predictor_top_p = 1.0f;
    int32_t predictor_top_k = 50;

    // Sampling seeds (deterministic defaults; callers can override).
    int32_t seed = 42;
    int32_t predictor_seed = 45;
    
    // Number of threads
    int32_t n_threads = 4;

    // Enable verbose ORT runtime logs (warning+) for ONNX debugging.
    // Default false keeps runtime output quiet at error level.
    bool ort_debug_log = false;
    
    // Print progress during generation
    bool print_progress = false;
    
    // Print timing information
    bool print_timing = true;
    
    // Repetition penalty for CB0 token generation (HuggingFace style)
    float repetition_penalty = 1.05f;

    // Language ID for codec. -1 means "Auto(None)" (no language token injection).
    int32_t language_id = -1;

    // Instruct text for Custom Voice / Voice Design modes.
    // For Custom: emotion/style instructions (e.g. "用温柔的语气说")
    // For Design: full voice design description
    std::string instruct;

    // Speaker name for Custom Voice mode (e.g. "Vivian", "Ryan")
    std::string speaker_name;
    
    // Reference text for Voice Clone mode (matches reference audio)
    std::string ref_text;

    // Streaming pipeline: number of frames to accumulate before the first
    // decoder chunk fires. Smaller value = lower time-to-first-audio but
    // slightly worse steady-state decoder RTF (the first chunk has fixed
    // ORT overhead). Steady-state chunks use the decoder's internal
    // chunk size (see StatefulDecoderOnnx::decode_chunk_frames()).
    // Set to <=0 to use the default (8).
    int32_t first_chunk_frames = 8;

    // Optional audio chunk callback fired from the decoder worker thread
    // as each PCM chunk becomes available. When set, the caller receives
    // progressive chunks AND still gets the cumulative audio in
    // tts_result at the end — use one or the other. The slice points
    // into tts_result.audio and is valid only for the duration of the
    // callback; copy it if you need to keep it. Caller must not block
    // for long: this runs on the decoder worker and any stall here
    // stalls synthesis.
    //
    // Signature: (samples_ptr, n_samples, is_last_chunk)
    std::function<void(const float *, int32_t, bool)> chunk_callback;
};

// TTS generation result
struct tts_result {
    // Generated audio samples (24kHz, mono)
    std::vector<float> audio;

    // Number of samples originally produced. Mirrors audio.size() at the
    // moment the engine returns, but callers that release `audio` after
    // consuming it (e.g. CLI --repeat loops) can still read the count.
    int32_t n_samples = 0;

    // Sample rate
    int32_t sample_rate = 24000;
    
    // Success flag
    bool success = false;
    
    // Error message if failed
    std::string error_msg;
    
    // Timing info (in milliseconds)
    int64_t t_load_ms = 0;
    int64_t t_warmup_ms = 0;
    int64_t t_tokenize_ms = 0;
    int64_t t_encode_ms = 0;
    int64_t t_generate_ms = 0;
    int64_t t_llama_prefill_ms = 0;
    int64_t t_llama_decode_loop_ms = 0;
    int64_t t_talker_post_ms = 0;
    int64_t t_predictor_sample_ms = 0;
    int64_t t_talker_decode_ms = 0;
    int64_t t_talker_post_prep_ms = 0;
    int64_t t_talker_post_copy_ms = 0;
    int64_t t_decode_ms = 0;
    int64_t t_ort_decoder_run_ms = 0;
    int64_t t_decoder_tensor_prep_ms = 0;
    int64_t t_decoder_ort_run_ms = 0;
    int64_t t_decoder_tensor_extract_ms = 0;
    int64_t t_decoder_state_trim_ms = 0;
    int64_t t_pcm_gather_ms = 0;
    int64_t t_total_ms = 0;

    // Streaming pipeline diagnostics. t_first_audio_ms is the wall-clock time
    // (relative to synth start) at which the first decoder chunk completed
    // and PCM samples became available to the caller. first_chunk_frames_used
    // records the chunk size that was actually consumed for the first chunk.
    int64_t t_first_audio_ms = 0;
    int32_t first_chunk_frames_used = 0;

    // Language used for generation after explicit override (-1 means no language token).
    int32_t effective_language_id = -1;
    bool used_auto_language = false; // reserved for backward compatibility, always false.

    // Process memory snapshots (bytes)
    uint64_t mem_rss_start_bytes = 0;
    uint64_t mem_rss_end_bytes = 0;
    uint64_t mem_rss_peak_bytes = 0;
    uint64_t mem_phys_start_bytes = 0;
    uint64_t mem_phys_end_bytes = 0;
    uint64_t mem_phys_peak_bytes = 0;

    // Clone/generation diagnostics for alignment & noisy-audio triage.
    int32_t spk_emb_dim = 0;
    float spk_emb_l2 = 0.0f;
    int32_t spk_emb_nan_count = 0;
    int32_t spk_emb_inf_count = 0;

    int32_t ref_code_frames = 0;
    int32_t ref_codebooks = 0;
    int32_t ref_code_min = -1;
    int32_t ref_code_max = -1;

    int32_t gen_code_frames = 0;
    int32_t gen_codebooks = 0;
    int32_t gen_code_min = -1;
    int32_t gen_code_max = -1;
    uint64_t gen_codes_hash = 0;
    int32_t eos_step = -1;
    int32_t ctx_required = 0;
    int32_t ctx_allocated = 0;
    int32_t ctx_cap = 0;
    bool ctx_overflow = false;

    float pcm_peak = 0.0f;
    float pcm_rms = 0.0f;

    // ORT provider diagnostics
    std::string ort_provider_speaker_encoder = "not_loaded";
    std::string ort_provider_codec_encoder = "not_loaded";
    std::string ort_provider_decoder = "not_loaded";
    
};

// Progress callback type
using tts_progress_callback_t = std::function<void(int tokens_generated, int max_tokens)>;

// Main TTS class that orchestrates the full pipeline
class Engine {
public:
    Engine();
    ~Engine();
    
    // Load all models from directory.
    // Required layout:
    //   lunavox_talker*.gguf
    //   lunavox_predictor*.gguf
    //   lunavox_decoder*.onnx
    // Optional:
    //   lunavox_speaker_encoder*.onnx
    //   lunavox_codec_encoder*.onnx
    //   embeddings/
    //   tokenizer.json
    bool load_models(const std::string & model_dir, int32_t n_threads = 4);
    
    // Generate speech from text
    // text: input text to synthesize
    // params: generation parameters
    tts_result synthesize(const std::string & text,
                          const tts_params & params = tts_params());
    
    // Generate speech with voice cloning
    // text: input text to synthesize
    // reference_audio: path to reference audio file (WAV, 24kHz)
    // params: generation parameters
    tts_result synthesize_with_voice(const std::string & text,
                                      const std::string & reference_audio,
                                      const tts_params & params = tts_params());
    
    // Generate speech with voice cloning from samples
    // text: input text to synthesize
    // ref_samples: reference audio samples (24kHz, mono, normalized to [-1, 1])
    // n_ref_samples: number of reference samples
    // params: generation parameters
    tts_result synthesize_with_voice(const std::string & text,
                                      const float * ref_samples, int32_t n_ref_samples,
                                      const tts_params & params = tts_params());
    
    // Extract speaker embedding from raw audio samples (for caching)
    // ref_samples: 24kHz mono float32 normalized to [-1, 1]
    // embedding: output vector (typically 2048 for Qwen3-TTS)
    // Returns true on success
    bool extract_speaker_embedding(const float * ref_samples, int32_t n_ref_samples,
                                   std::vector<float> & embedding,
                                   const tts_params & params = tts_params());

    // Synthesize with pre-computed speaker embedding (skips encoder)
    // embedding: speaker embedding from extract_speaker_embedding()
    // embedding_size: must match runtime hidden size (typically 2048)
    tts_result synthesize_with_embedding(const std::string & text,
                                          const float * embedding, int32_t embedding_size,
                                          const tts_params & params = tts_params());

    // Custom Voice synthesis (uses built-in speaker + optional instruct)
    // speaker: speaker name (e.g. "Vivian", "Ryan", "Aiden")
    // instruct: optional style/emotion instruction
    tts_result synthesize_custom(const std::string & text,
                                  const std::string & speaker,
                                  const std::string & instruct,
                                  const tts_params & params = tts_params());

    // Voice Design synthesis (instruct-only, no speaker embedding)
    // instruct: full voice design description
    tts_result synthesize_design(const std::string & text,
                                  const std::string & instruct,
                                  const tts_params & params = tts_params());

    // Set progress callback
    void set_progress_callback(tts_progress_callback_t callback);
    
    // Get error message
    const std::string & get_error() const { return error_msg_; }
    
    // Check if models are loaded
    bool is_loaded() const { return models_loaded_; }

    // Enable/disable the post-load warmup pass. Default: enabled.
    // Warmup cost is accumulated into last_warmup_ms().
    void set_warmup_enabled(bool enabled) { warmup_enabled_ = enabled; }
    int64_t last_warmup_ms() const { return last_warmup_ms_; }

    // Wall time spent inside the most recent load_models() call
    // (includes warmup). 0 if no load has happened yet.
    int64_t last_load_ms() const { return last_load_ms_; }

    // Runtime profile accessors.
    const ModelProfile & profile() const { return profile_; }
    bool supports_mode(const std::string & mode) const;
    bool resolve_language_id(const std::string & name_or_alias, int32_t & language_id_out) const;
    std::vector<std::string> supported_speakers() const;
    std::vector<std::string> supported_languages() const;
    
private:
    bool preload_hot_embedding_rows();

    tts_result synthesize_internal(const std::string & text,
                                   const float * speaker_embedding,
                                   const int32_t * ref_codes,
                                   int32_t n_ref_frames,
                                   const tts_params & params,
                                   tts_result & result);

    bool load_model_profile(const std::string & path);

    bool load_models_new_layout(const std::string & model_dir, int32_t n_threads);
    
    TextTokenizer tokenizer_;
    TalkerPredictor talker_predictor_;
    AssetsManager assets_;
    SpeakerEncoderOnnx speaker_encoder_;
    CodecEncoderOnnx codec_encoder_;
    StatefulDecoderOnnx decoder_;
    
    bool models_loaded_ = false;
    bool speaker_encoder_loaded_ = false;
    bool codec_encoder_loaded_ = false;
    bool talker_predictor_loaded_ = false;
    bool assets_loaded_ = false;
    bool decoder_loaded_ = false;
    bool low_mem_mode_ = false;
    bool warmup_enabled_ = true;
    int64_t last_warmup_ms_ = 0;
    int64_t last_load_ms_ = 0;
    bool hot_rows_preloaded_ = false;
    std::mutex hot_rows_preload_mu_;
    std::string error_msg_;
    std::string speaker_onnx_path_;
    std::string codec_encoder_onnx_path_;
    std::string talker_model_path_;
    std::string predictor_model_path_;
    std::string embeddings_dir_path_;
    std::string decoder_onnx_path_;
    std::string tokenizer_json_path_;
    std::string model_profile_json_path_;
    ModelProfile profile_;
    tts_progress_callback_t progress_callback_;
};

} // namespace lunavox
