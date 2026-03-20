#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qwen3_tts {

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

    bool encode(const float * samples, int32_t n_samples, std::vector<int32_t> & codes, int32_t & n_frames);

private:
    std::string error_msg_;
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

    bool encode(const float * samples, int32_t n_samples, std::vector<float> & embedding);

private:
    bool compute_mel_spectrogram(const float * samples, int32_t n_samples, std::vector<float> & mel, int32_t & n_frames);
    bool ensure_mel_kernels();

    mel_config cfg_;
    std::string error_msg_;
    bool loaded_ = false;
    void * session_impl_ = nullptr;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<float> mel_filterbank_;
    std::vector<float> window_;
};

class StatefulDecoderOnnx {
public:
    bool load_model(const std::string & model_path, int32_t intra_threads = 1);
    void unload_model();
    bool is_loaded() const { return loaded_; }
    const std::string & get_error() const { return error_msg_; }

    bool decode(const int32_t * codes, int32_t n_frames, std::vector<float> & audio);
    int32_t sample_rate() const { return sample_rate_; }

private:
    struct state_buffer {
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
    };

    std::string error_msg_;
    bool loaded_ = false;
    int32_t sample_rate_ = 24000;
    int32_t num_layers_ = 0;
    int32_t num_heads_ = 0;
    int32_t head_dim_ = 0;
    int32_t decode_chunk_frames_ = 12;
    int32_t pre_conv_channels_ = 512;
    int32_t latent_channels_ = 1024;
    int32_t conv_channels_ = 1024;
    int32_t state_elem_type_ = 1; // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    int32_t kv_elem_type_ = 1;    // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT

    void * session_impl_ = nullptr;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

// Windowed-sinc resampler (Kaiser window) for better quality than linear interpolation.
bool resample_windowed_sinc(
    const float * input,
    int32_t input_len,
    int32_t input_rate,
    std::vector<float> & output,
    int32_t output_rate);

} // namespace qwen3_tts
