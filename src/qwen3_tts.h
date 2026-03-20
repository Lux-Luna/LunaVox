#pragma once

#include "text_tokenizer.h"
#include "onnx_audio_runtime.h"
#include "assets_manager.h"
#include "talker_predictor_llama.h"

#include <string>
#include <vector>
#include <functional>
#include <cstdint>

namespace qwen3_tts {

// TTS generation parameters
struct tts_params {
    // Maximum number of audio tokens to generate
    int32_t max_audio_tokens = 4096;
    
    // Temperature for sampling (0 = greedy)
    float temperature = 0.9f;
    
    // Top-p sampling
    float top_p = 1.0f;
    
    // Top-k sampling (0 = disabled)
    int32_t top_k = 50;

    // Predictor stage sampling controls (Q1..Q15 generation).
    bool predictor_do_sample = true;
    float predictor_temperature = 0.9f;
    float predictor_top_p = 1.0f;
    int32_t predictor_top_k = 50;

    // Sampling seeds (-1 means random seed from clock).
    int32_t seed = -1;
    int32_t predictor_seed = -1;
    
    // Number of threads
    int32_t n_threads = 4;
    
    // Print progress during generation
    bool print_progress = false;
    
    // Print timing information
    bool print_timing = true;
    
    // Repetition penalty for CB0 token generation (HuggingFace style)
    float repetition_penalty = 1.05f;

    // Auto-detect language from text when true.
    // If false, use language_id as-is.
    bool auto_language = true;

    // Language ID for codec (2050=en, 2069=ru, 2055=zh, 2058=ja, 2064=ko, 2053=de, 2061=fr, 2054=es)
    int32_t language_id = 2050;

};

// TTS generation result
struct tts_result {
    // Generated audio samples (24kHz, mono)
    std::vector<float> audio;
    
    // Sample rate
    int32_t sample_rate = 24000;
    
    // Success flag
    bool success = false;
    
    // Error message if failed
    std::string error_msg;
    
    // Timing info (in milliseconds)
    int64_t t_load_ms = 0;
    int64_t t_tokenize_ms = 0;
    int64_t t_encode_ms = 0;
    int64_t t_generate_ms = 0;
    int64_t t_decode_ms = 0;
    int64_t t_total_ms = 0;

    // Language used for generation after auto-detection / override.
    int32_t effective_language_id = 2050;
    bool used_auto_language = false;

    // Process memory snapshots (bytes)
    uint64_t mem_rss_start_bytes = 0;
    uint64_t mem_rss_end_bytes = 0;
    uint64_t mem_rss_peak_bytes = 0;
    uint64_t mem_phys_start_bytes = 0;
    uint64_t mem_phys_end_bytes = 0;
    uint64_t mem_phys_peak_bytes = 0;
    
};

// Progress callback type
using tts_progress_callback_t = std::function<void(int tokens_generated, int max_tokens)>;

// Main TTS class that orchestrates the full pipeline
class Qwen3TTS {
public:
    Qwen3TTS();
    ~Qwen3TTS();
    
    // Load all models from directory.
    // Required layout:
    //   qwen3_tts_talker.q5_k.gguf
    //   qwen3_tts_predictor.q8_0.gguf
    //   qwen3_tts_speaker_encoder.fp16.onnx
    //   qwen3_tts_codec_encoder.fp16.onnx
    //   qwen3_tts_decoder.fp16.onnx
    //   embeddings/
    //   tokenizer.json
    bool load_models(const std::string & model_dir);
    
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

    // Set progress callback
    void set_progress_callback(tts_progress_callback_t callback);
    
    // Get error message
    const std::string & get_error() const { return error_msg_; }
    
    // Check if models are loaded
    bool is_loaded() const { return models_loaded_; }
    
private:
    tts_result synthesize_internal(const std::string & text,
                                   const float * speaker_embedding,
                                   const int32_t * ref_codes,
                                   int32_t n_ref_frames,
                                   const tts_params & params,
                                   tts_result & result);

    bool load_models_new_layout(const std::string & model_dir, int32_t n_threads);
    
    TextTokenizer tokenizer_;
    TalkerPredictorLlama talker_predictor_;
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
    std::string error_msg_;
    std::string speaker_onnx_path_;
    std::string codec_encoder_onnx_path_;
    std::string talker_model_path_;
    std::string predictor_model_path_;
    std::string embeddings_dir_path_;
    std::string decoder_onnx_path_;
    std::string tokenizer_json_path_;
    tts_progress_callback_t progress_callback_;
};

// Utility: Load audio file (WAV format)
bool load_audio_file(const std::string & path, std::vector<float> & samples, 
                     int & sample_rate);

// Utility: Save audio file (WAV format)
bool save_audio_file(const std::string & path, const std::vector<float> & samples,
                     int sample_rate);

} // namespace qwen3_tts
