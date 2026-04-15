#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lunavox {

// Configure ORT runtime logging before creating any ONNX session.
// false (default): ERROR only; true: WARNING and above.
void set_ort_debug_log(bool enabled);
bool ort_debug_log_enabled();

struct mel_config {
    int32_t sample_rate = 24000;
    int32_t n_mels = 128;
    int32_t n_fft = 1024;
    int32_t hop_length = 256;
    int32_t win_length = 1024;
    float f_min = 0.0f;
    float f_max = 12000.0f;
};

class CodecEncoderOnnx {
public:
    bool load_model(const std::string & model_path, int32_t intra_threads = 1);
    void unload_model();
    bool is_loaded() const { return loaded_; }
    const std::string & get_error() const { return error_msg_; }
    const std::string & provider_summary() const { return provider_summary_; }

    bool encode(const float * samples, int32_t n_samples, std::vector<int32_t> & codes, int32_t & n_frames);

private:
    std::string error_msg_;
    std::string provider_summary_ = "not_loaded";
    bool loaded_ = false;
    void * session_impl_ = nullptr;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    // Dynamo-exported codec ONNX can require aligned input lengths.
    // Keep defaults compatible with current Qwen3-TTS tokenizer-12Hz exports.
    int32_t align_min_samples_ = 24000;
    int32_t align_mod_base_ = 960;
    int32_t align_mod_stride_ = 1920;
    bool align_grid_enabled_ = true;
};

class SpeakerEncoderOnnx {
public:
    bool load_model(const std::string & model_path, int32_t intra_threads = 1);
    void unload_model();
    bool is_loaded() const { return loaded_; }
    const std::string & get_error() const { return error_msg_; }
    const mel_config & get_config() const { return cfg_; }
    const std::string & provider_summary() const { return provider_summary_; }

    bool encode(const float * samples, int32_t n_samples, std::vector<float> & embedding);

private:
    bool compute_mel_spectrogram(const float * samples, int32_t n_samples, std::vector<float> & mel, int32_t & n_frames);
    bool ensure_mel_kernels();

    mel_config cfg_;
    std::string error_msg_;
    std::string provider_summary_ = "not_loaded";
    bool loaded_ = false;
    void * session_impl_ = nullptr;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<float> mel_filterbank_;
    std::vector<float> window_;
};

class StatefulDecoderOnnx {
public:
    // Caller-owned streaming state. Stage 1 of the streaming refactor:
    // decode() now internally calls begin_stream + decode_chunk in a loop
    // using a local instance; callers that want to pipeline talker output
    // against decoder work can hold their own instance and feed chunks as
    // frames become available.
    struct DecoderStreamState {
        struct tensor_state {
            int32_t elem_type = 1; // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            std::vector<float> f32;
            std::vector<uint16_t> f16;
            int64_t seq = 0;
        };

        tensor_state pre_conv_history;
        tensor_state latent_buffer;
        tensor_state conv_history;
        std::vector<tensor_state> past_keys;
        std::vector<tensor_state> past_values;

        // Cached I/O bookkeeping filled by begin_stream()
        std::vector<int32_t> past_key_input_idx;
        std::vector<int32_t> past_value_input_idx;
        std::vector<const char *> out_names_cache;

        // Per-chunk scratch reused across chunks inside one stream.
        std::vector<int64_t> codes_i64;
    };

    bool load_model(const std::string & model_path, int32_t intra_threads = 1);
    void unload_model();
    bool is_loaded() const { return loaded_; }
    const std::string & get_error() const { return error_msg_; }
    const std::string & provider_summary() const { return provider_summary_; }

    bool decode(const int32_t * codes, int32_t n_frames, std::vector<float> & audio);
    int32_t sample_rate() const { return sample_rate_; }
    int32_t decode_chunk_frames() const { return decode_chunk_frames_; }

    // Streaming API. begin_stream() initialises state; decode_chunk() runs one
    // chunk of frames through ORT and appends PCM to out_pcm_append.
    // Timings (last_t_*) accumulate across chunks; call reset_stream_timings()
    // at the boundary where you want a fresh counter (typically once per
    // synthesize call).
    bool begin_stream(DecoderStreamState & state);
    bool decode_chunk(DecoderStreamState & state,
                      const int32_t * codes,
                      int32_t chunk_frames,
                      bool is_last,
                      std::vector<float> & out_pcm_append);
    void reset_stream_timings();

    // Per-decode() call breakdown (reset at each decode() entry or on
    // reset_stream_timings()).
    int64_t last_t_tensor_prep_ms() const { return t_tensor_prep_ms_; }
    int64_t last_t_ort_run_ms() const { return t_ort_run_ms_; }
    int64_t last_t_tensor_extract_ms() const { return t_tensor_extract_ms_; }
    int64_t last_t_state_trim_ms() const { return t_state_trim_ms_; }

private:

    std::string error_msg_;
    std::string provider_summary_ = "not_loaded";
    bool loaded_ = false;
    int32_t sample_rate_ = 24000;
    int32_t num_layers_ = 0;
    int32_t num_heads_ = 0;
    int32_t head_dim_ = 0;
    // Chunk size trades per-Run() overhead (DML command submit + state tensor
    // copies + fence wait) against peak workspace. Sweep on RTX 3090 showed
    // decode time scaling: 8→984ms, 16→783ms, 24→518ms, 32→452ms (1.7B).
    // 32 halves decode vs 8 with negligible VRAM impact and imperceptible
    // numerical drift at chunk boundaries (PSNR 34-42 dB vs chunk=8).
    int32_t decode_chunk_frames_ = 32;
    int32_t pre_conv_channels_ = 512;
    int32_t latent_channels_ = 1024;
    int32_t conv_channels_ = 1024;
    int32_t pre_conv_window_ = 2;
    int32_t latent_window_ = 4;
    int32_t conv_window_ = 4;
    int32_t kv_cache_window_ = 72;
    int32_t state_elem_type_ = 1; // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    int32_t kv_elem_type_ = 1;    // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT

    void * session_impl_ = nullptr;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    int64_t t_tensor_prep_ms_ = 0;
    int64_t t_ort_run_ms_ = 0;
    int64_t t_tensor_extract_ms_ = 0;
    int64_t t_state_trim_ms_ = 0;
};

} // namespace lunavox
