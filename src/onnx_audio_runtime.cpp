#include "onnx_audio_runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "onnxruntime_cxx_api.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace qwen3_tts {

namespace {

static constexpr float kPi = 3.14159265358979323846f;
static constexpr float kEps = 1e-9f;

struct ort_env_holder {
    Ort::Env env;
    ort_env_holder() : env(ORT_LOGGING_LEVEL_WARNING, "lunavox-ort") {}
};

Ort::Env & get_ort_env() {
    static ort_env_holder holder;
    return holder.env;
}

#ifdef _WIN32
std::wstring utf8_to_wide(const std::string & s) {
    if (s.empty()) {
        return std::wstring();
    }
    int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int) s.size(), nullptr, 0);
    if (len <= 0) {
        return std::wstring();
    }
    std::wstring w((size_t) len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int) s.size(), &w[0], len);
    return w;
}
#endif

struct ort_session_data {
    Ort::Session session{nullptr};
    std::unique_ptr<std::wstring> wide_path; // keep lifetime for Windows constructor args
};

ort_session_data * as_session(void * ptr) {
    return reinterpret_cast<ort_session_data *>(ptr);
}

const ort_session_data * as_session_const(const void * ptr) {
    return reinterpret_cast<const ort_session_data *>(ptr);
}

bool create_session_impl(
    const std::string & model_path,
    int32_t intra_threads,
    std::string & error_msg,
    void *& out_ptr,
    std::vector<std::string> & input_names,
    std::vector<std::string> & output_names) {
    try {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (intra_threads > 0) {
            opts.SetIntraOpNumThreads((int) intra_threads);
        }
        opts.SetInterOpNumThreads(1);

        auto * impl = new ort_session_data();
#ifdef _WIN32
        impl->wide_path = std::make_unique<std::wstring>(utf8_to_wide(model_path));
        impl->session = Ort::Session(get_ort_env(), impl->wide_path->c_str(), opts);
#else
        impl->session = Ort::Session(get_ort_env(), model_path.c_str(), opts);
#endif

        Ort::AllocatorWithDefaultOptions allocator;
        size_t n_inputs = impl->session.GetInputCount();
        size_t n_outputs = impl->session.GetOutputCount();

        input_names.clear();
        output_names.clear();
        input_names.reserve(n_inputs);
        output_names.reserve(n_outputs);

        for (size_t i = 0; i < n_inputs; ++i) {
            auto name = impl->session.GetInputNameAllocated(i, allocator);
            input_names.emplace_back(name.get() ? name.get() : "");
        }
        for (size_t i = 0; i < n_outputs; ++i) {
            auto name = impl->session.GetOutputNameAllocated(i, allocator);
            output_names.emplace_back(name.get() ? name.get() : "");
        }

        out_ptr = impl;
        return true;
    } catch (const std::exception & e) {
        error_msg = std::string("Failed to create ONNX session: ") + e.what();
        out_ptr = nullptr;
        return false;
    }
}

void destroy_session_impl(void *& ptr) {
    if (!ptr) {
        return;
    }
    auto * impl = as_session(ptr);
    delete impl;
    ptr = nullptr;
}

int64_t gcd64(int64_t a, int64_t b) {
    while (b != 0) {
        int64_t t = a % b;
        a = b;
        b = t;
    }
    return a < 0 ? -a : a;
}

inline float sinc(float x) {
    if (std::fabs(x) < 1e-8f) {
        return 1.0f;
    }
    float px = kPi * x;
    return std::sin(px) / px;
}

inline float bessel_i0(float x) {
    // Numerical approximation used by common Kaiser window implementations.
    float sum = 1.0f;
    float y = x * x / 4.0f;
    float t = y;
    int k = 1;
    while (t > 1e-9f * sum) {
        sum += t;
        ++k;
        t *= y / (float) (k * k);
    }
    return sum;
}

float hz_to_mel_slaney(float hz) {
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp;
    const float logstep = std::log(6.4f) / 27.0f;

    if (hz < min_log_hz) {
        return hz / f_sp;
    }
    return min_log_mel + std::log(hz / min_log_hz) / logstep;
}

float mel_to_hz_slaney(float mel) {
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp;
    const float logstep = std::log(6.4f) / 27.0f;

    if (mel < min_log_mel) {
        return mel * f_sp;
    }
    return min_log_hz * std::exp(logstep * (mel - min_log_mel));
}

void compute_mel_filterbank_slaney(
    const mel_config & cfg,
    std::vector<float> & filterbank) {
    const int n_fft_bins = cfg.n_fft / 2 + 1;
    filterbank.assign((size_t) cfg.n_mels * (size_t) n_fft_bins, 0.0f);

    const float mel_min = hz_to_mel_slaney(cfg.f_min);
    const float mel_max = hz_to_mel_slaney(cfg.f_max);

    std::vector<float> mel_points((size_t) cfg.n_mels + 2);
    for (int i = 0; i < cfg.n_mels + 2; ++i) {
        mel_points[(size_t) i] = mel_min + (mel_max - mel_min) * (float) i / (float) (cfg.n_mels + 1);
    }

    std::vector<float> hz_points((size_t) cfg.n_mels + 2);
    for (int i = 0; i < cfg.n_mels + 2; ++i) {
        hz_points[(size_t) i] = mel_to_hz_slaney(mel_points[(size_t) i]);
    }

    std::vector<float> fft_freqs((size_t) n_fft_bins);
    for (int i = 0; i < n_fft_bins; ++i) {
        fft_freqs[(size_t) i] = (float) i * (float) cfg.sample_rate / (float) cfg.n_fft;
    }

    for (int m = 0; m < cfg.n_mels; ++m) {
        const float left = hz_points[(size_t) m];
        const float center = hz_points[(size_t) m + 1];
        const float right = hz_points[(size_t) m + 2];
        const float enorm = 2.0f / std::max(right - left, 1e-8f);

        for (int k = 0; k < n_fft_bins; ++k) {
            const float f = fft_freqs[(size_t) k];
            float w = 0.0f;
            if (f >= left && f <= center) {
                w = (f - left) / std::max(center - left, 1e-8f);
            } else if (f > center && f <= right) {
                w = (right - f) / std::max(right - center, 1e-8f);
            }
            filterbank[(size_t) m * (size_t) n_fft_bins + (size_t) k] = w * enorm;
        }
    }
}

void compute_centered_periodic_hann(const mel_config & cfg, std::vector<float> & window) {
    window.assign((size_t) cfg.n_fft, 0.0f);
    const int offset = (cfg.n_fft - cfg.win_length) / 2;
    for (int i = 0; i < cfg.win_length; ++i) {
        window[(size_t) (offset + i)] = 0.5f * (1.0f - std::cos(2.0f * kPi * (float) i / (float) cfg.win_length));
    }
}

void compute_dft(const float * input, int n, std::vector<float> & real, std::vector<float> & imag) {
    real.assign((size_t) n, 0.0f);
    imag.assign((size_t) n, 0.0f);
    for (int k = 0; k < n; ++k) {
        float re = 0.0f;
        float im = 0.0f;
        for (int t = 0; t < n; ++t) {
            float angle = -2.0f * kPi * (float) (k * t) / (float) n;
            float x = input[t];
            re += x * std::cos(angle);
            im += x * std::sin(angle);
        }
        real[(size_t) k] = re;
        imag[(size_t) k] = im;
    }
}

bool find_name(const std::vector<std::string> & names, const std::string & target) {
    return std::find(names.begin(), names.end(), target) != names.end();
}

int64_t parse_valid_samples(const Ort::Value & val) {
    auto info = val.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = info.GetElementType();
    size_t n = info.GetElementCount();
    if (n == 0) {
        return 0;
    }
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        const int64_t * p = val.GetTensorData<int64_t>();
        return p[0];
    }
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        const int32_t * p = val.GetTensorData<int32_t>();
        return (int64_t) p[0];
    }
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float * p = val.GetTensorData<float>();
        return (int64_t) std::llround((double) p[0]);
    }
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        const double * p = val.GetTensorData<double>();
        return (int64_t) std::llround(p[0]);
    }
    return 0;
}

} // namespace

bool resample_windowed_sinc(
    const float * input,
    int32_t input_len,
    int32_t input_rate,
    std::vector<float> & output,
    int32_t output_rate) {
    output.clear();
    if (!input || input_len <= 0 || input_rate <= 0 || output_rate <= 0) {
        return false;
    }
    if (input_rate == output_rate) {
        output.assign(input, input + input_len);
        return true;
    }

    const int64_t g = gcd64(input_rate, output_rate);
    const int64_t up = output_rate / g;
    const int64_t down = input_rate / g;
    const double ratio = (double) output_rate / (double) input_rate;
    const int32_t out_len = std::max(1, (int32_t) std::llround((double) input_len * ratio));
    output.assign((size_t) out_len, 0.0f);

    const int taps_per_side = 16;
    const float beta = 5.0f;
    const float i0_beta = bessel_i0(beta);
    const float cutoff = 1.0f / (float) std::max<int64_t>(up, down);

    for (int32_t i = 0; i < out_len; ++i) {
        const double src = (double) i * (double) input_rate / (double) output_rate;
        const int32_t center = (int32_t) std::floor(src);
        float sum = 0.0f;
        float wsum = 0.0f;
        for (int k = -taps_per_side; k <= taps_per_side; ++k) {
            const int32_t idx = center + k;
            if (idx < 0 || idx >= input_len) {
                continue;
            }
            const float x = (float) (src - (double) idx);
            const float z = (float) k / (float) taps_per_side;
            const float win = bessel_i0(beta * std::sqrt(std::max(0.0f, 1.0f - z * z))) / i0_beta;
            const float h = 2.0f * cutoff * sinc(2.0f * cutoff * x) * win;
            sum += input[idx] * h;
            wsum += h;
        }
        if (std::fabs(wsum) > 1e-8f) {
            output[(size_t) i] = sum / wsum;
        } else {
            output[(size_t) i] = 0.0f;
        }
    }
    return true;
}

bool CodecEncoderOnnx::load_model(const std::string & model_path, int32_t intra_threads) {
    unload_model();
    error_msg_.clear();
    return create_session_impl(model_path, intra_threads, error_msg_, session_impl_, input_names_, output_names_) && (loaded_ = true);
}

void CodecEncoderOnnx::unload_model() {
    destroy_session_impl(session_impl_);
    input_names_.clear();
    output_names_.clear();
    loaded_ = false;
}

bool CodecEncoderOnnx::encode(
    const float * samples,
    int32_t n_samples,
    std::vector<int32_t> & codes,
    int32_t & n_frames) {
    codes.clear();
    n_frames = 0;
    if (!loaded_ || !session_impl_) {
        error_msg_ = "Codec encoder is not loaded";
        return false;
    }
    if (!samples || n_samples <= 0) {
        error_msg_ = "Invalid audio buffer for codec encoder";
        return false;
    }
    if (input_names_.empty() || output_names_.empty()) {
        error_msg_ = "Codec encoder session I/O names are missing";
        return false;
    }

    try {
        auto * impl = as_session(session_impl_);
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<float> wav((size_t) n_samples);
        std::memcpy(wav.data(), samples, (size_t) n_samples * sizeof(float));

        std::array<int64_t, 2> in_shape = {1, (int64_t) n_samples};
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            mem, wav.data(), wav.size(), in_shape.data(), in_shape.size());

        const char * in_name = input_names_[0].c_str();
        const char * out_name = output_names_[0].c_str();
        std::array<const char *, 1> in_names = {in_name};
        std::array<const char *, 1> out_names = {out_name};
        std::array<Ort::Value, 1> in_values = {std::move(in_tensor)};

        auto out = impl->session.Run(
            Ort::RunOptions{nullptr},
            in_names.data(),
            in_values.data(),
            in_values.size(),
            out_names.data(),
            out_names.size());

        if (out.empty()) {
            error_msg_ = "Codec encoder returned no outputs";
            return false;
        }

        auto info = out[0].GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        if (shape.size() < 3) {
            error_msg_ = "Codec encoder output shape is invalid";
            return false;
        }

        const int64_t batch = shape[0] < 0 ? 1 : shape[0];
        const int64_t frames = shape[1] < 0 ? 0 : shape[1];
        const int64_t codebooks = shape[2] < 0 ? 16 : shape[2];
        if (batch != 1 || frames <= 0 || codebooks <= 0) {
            error_msg_ = "Codec encoder output dimensions are not supported";
            return false;
        }

        const int64_t total = info.GetElementCount();
        if (total != frames * codebooks) {
            error_msg_ = "Codec encoder output element count mismatch";
            return false;
        }

        const int64_t * raw = out[0].GetTensorData<int64_t>();
        codes.resize((size_t) total);
        for (int64_t i = 0; i < total; ++i) {
            codes[(size_t) i] = (int32_t) raw[i];
        }
        n_frames = (int32_t) frames;
        return true;
    } catch (const std::exception & e) {
        error_msg_ = std::string("Codec encoder inference failed: ") + e.what();
        return false;
    }
}

bool SpeakerEncoderOnnx::load_model(const std::string & model_path, int32_t intra_threads) {
    unload_model();
    error_msg_.clear();
    return create_session_impl(model_path, intra_threads, error_msg_, session_impl_, input_names_, output_names_) && (loaded_ = true);
}

void SpeakerEncoderOnnx::unload_model() {
    destroy_session_impl(session_impl_);
    input_names_.clear();
    output_names_.clear();
    loaded_ = false;
}

bool SpeakerEncoderOnnx::compute_mel_spectrogram(
    const float * samples,
    int32_t n_samples,
    std::vector<float> & mel,
    int32_t & n_frames) {
    mel.clear();
    n_frames = 0;
    if (!samples || n_samples <= 0) {
        error_msg_ = "Invalid audio for mel extraction";
        return false;
    }

    const int padding = (cfg_.n_fft - cfg_.hop_length) / 2;
    const int padded_len = n_samples + 2 * padding;
    if (padded_len < cfg_.n_fft) {
        error_msg_ = "Audio too short for mel extraction";
        return false;
    }

    std::vector<float> wav((size_t) padded_len);
    for (int i = 0; i < padding; ++i) {
        int src = std::min(n_samples - 1, std::max(0, padding - i));
        wav[(size_t) i] = samples[src];
    }
    std::memcpy(wav.data() + padding, samples, (size_t) n_samples * sizeof(float));
    for (int i = 0; i < padding; ++i) {
        int src = std::min(n_samples - 1, std::max(0, n_samples - 2 - i));
        wav[(size_t) (padding + n_samples + i)] = samples[src];
    }

    n_frames = 1 + (padded_len - cfg_.n_fft) / cfg_.hop_length;
    if (n_frames <= 0) {
        error_msg_ = "Failed to compute mel frame count";
        return false;
    }

    const int n_fft_bins = cfg_.n_fft / 2 + 1;
    std::vector<float> filterbank;
    compute_mel_filterbank_slaney(cfg_, filterbank);

    std::vector<float> window;
    compute_centered_periodic_hann(cfg_, window);

    mel.assign((size_t) n_frames * (size_t) cfg_.n_mels, 0.0f);
    std::vector<float> frame((size_t) cfg_.n_fft, 0.0f);
    std::vector<float> real;
    std::vector<float> imag;
    std::vector<float> mag((size_t) n_fft_bins, 0.0f);

    for (int f = 0; f < n_frames; ++f) {
        int start = f * cfg_.hop_length;
        for (int i = 0; i < cfg_.n_fft; ++i) {
            frame[(size_t) i] = wav[(size_t) (start + i)] * window[(size_t) i];
        }

        compute_dft(frame.data(), cfg_.n_fft, real, imag);
        for (int k = 0; k < n_fft_bins; ++k) {
            float re = real[(size_t) k];
            float im = imag[(size_t) k];
            mag[(size_t) k] = std::sqrt(re * re + im * im + kEps);
        }

        for (int m = 0; m < cfg_.n_mels; ++m) {
            float sum = 0.0f;
            const float * fb = &filterbank[(size_t) m * (size_t) n_fft_bins];
            for (int k = 0; k < n_fft_bins; ++k) {
                sum += fb[(size_t) k] * mag[(size_t) k];
            }
            const float log_mel = std::log(std::max(sum, 1e-5f));
            mel[(size_t) f * (size_t) cfg_.n_mels + (size_t) m] = log_mel;
        }
    }
    return true;
}

bool SpeakerEncoderOnnx::encode(const float * samples, int32_t n_samples, std::vector<float> & embedding) {
    embedding.clear();
    if (!loaded_ || !session_impl_) {
        error_msg_ = "Speaker encoder is not loaded";
        return false;
    }
    if (!samples || n_samples <= 0) {
        error_msg_ = "Invalid audio buffer for speaker encoder";
        return false;
    }
    if (input_names_.empty() || output_names_.empty()) {
        error_msg_ = "Speaker encoder session I/O names are missing";
        return false;
    }

    int32_t n_frames = 0;
    std::vector<float> mel;
    if (!compute_mel_spectrogram(samples, n_samples, mel, n_frames)) {
        return false;
    }

    try {
        auto * impl = as_session(session_impl_);
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::array<int64_t, 3> in_shape = {1, (int64_t) n_frames, (int64_t) cfg_.n_mels};
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            mem, mel.data(), mel.size(), in_shape.data(), in_shape.size());

        const char * in_name = input_names_[0].c_str();
        const char * out_name = output_names_[0].c_str();
        std::array<const char *, 1> in_names = {in_name};
        std::array<const char *, 1> out_names = {out_name};
        std::array<Ort::Value, 1> in_values = {std::move(in_tensor)};

        auto out = impl->session.Run(
            Ort::RunOptions{nullptr},
            in_names.data(),
            in_values.data(),
            in_values.size(),
            out_names.data(),
            out_names.size());
        if (out.empty()) {
            error_msg_ = "Speaker encoder returned no outputs";
            return false;
        }

        auto info = out[0].GetTensorTypeAndShapeInfo();
        size_t n = info.GetElementCount();
        if (n == 0) {
            error_msg_ = "Speaker encoder output is empty";
            return false;
        }
        const float * raw = out[0].GetTensorData<float>();
        embedding.assign(raw, raw + n);
        return true;
    } catch (const std::exception & e) {
        error_msg_ = std::string("Speaker encoder inference failed: ") + e.what();
        return false;
    }
}

bool StatefulDecoderOnnx::load_model(const std::string & model_path, int32_t intra_threads) {
    unload_model();
    error_msg_.clear();
    if (!create_session_impl(model_path, intra_threads, error_msg_, session_impl_, input_names_, output_names_)) {
        return false;
    }
    if (input_names_.size() < 5 || output_names_.size() < 2) {
        error_msg_ = "Decoder ONNX I/O count is invalid";
        unload_model();
        return false;
    }
    if (!find_name(input_names_, "audio_codes") || !find_name(input_names_, "is_last")) {
        error_msg_ = "Decoder ONNX missing expected inputs";
        unload_model();
        return false;
    }

    num_layers_ = (int32_t) ((input_names_.size() - 5) / 2);
    if (num_layers_ <= 0) {
        error_msg_ = "Decoder ONNX does not expose KV-cache inputs";
        unload_model();
        return false;
    }

    try {
        auto * impl = as_session(session_impl_);
        size_t key_input_idx = 5; // audio_codes, pre_conv_history, latent_buffer, conv_history, is_last
        auto ti = impl->session.GetInputTypeInfo(key_input_idx).GetTensorTypeAndShapeInfo();
        auto shape = ti.GetShape();
        if (shape.size() >= 4) {
            num_heads_ = (int32_t) (shape[1] > 0 ? shape[1] : 8);
            head_dim_ = (int32_t) (shape[3] > 0 ? shape[3] : 64);
        } else {
            num_heads_ = 8;
            head_dim_ = 64;
        }
    } catch (...) {
        num_heads_ = 8;
        head_dim_ = 64;
    }

    loaded_ = true;
    return true;
}

void StatefulDecoderOnnx::unload_model() {
    destroy_session_impl(session_impl_);
    input_names_.clear();
    output_names_.clear();
    loaded_ = false;
    num_layers_ = 0;
    num_heads_ = 0;
    head_dim_ = 0;
}

bool StatefulDecoderOnnx::decode(const int32_t * codes, int32_t n_frames, std::vector<float> & audio) {
    audio.clear();
    if (!loaded_ || !session_impl_) {
        error_msg_ = "Decoder is not loaded";
        return false;
    }
    if (!codes || n_frames <= 0) {
        error_msg_ = "Invalid codes for decoder";
        return false;
    }
    if (num_layers_ <= 0 || num_heads_ <= 0 || head_dim_ <= 0) {
        error_msg_ = "Decoder state layout is invalid";
        return false;
    }

    try {
        auto * impl = as_session(session_impl_);
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<int64_t> codes_i64((size_t) n_frames * 16);
        for (int64_t i = 0; i < (int64_t) codes_i64.size(); ++i) {
            codes_i64[(size_t) i] = (int64_t) codes[i];
        }

        std::vector<const char *> in_names;
        std::vector<Ort::Value> in_values;
        in_names.reserve((size_t) 5 + (size_t) 2 * (size_t) num_layers_);
        in_values.reserve((size_t) 5 + (size_t) 2 * (size_t) num_layers_);

        std::array<int64_t, 3> codes_shape = {1, (int64_t) n_frames, 16};
        in_names.push_back("audio_codes");
        in_values.emplace_back(Ort::Value::CreateTensor<int64_t>(
            mem, codes_i64.data(), codes_i64.size(), codes_shape.data(), codes_shape.size()));

        static float dummy_f = 0.0f;
        static int64_t dummy_i64 = 0;

        std::array<int64_t, 3> pre_conv_shape = {1, 512, 0};
        std::array<int64_t, 3> latent_shape = {1, 1024, 0};
        std::array<int64_t, 3> conv_shape = {1, 1024, 0};
        in_names.push_back("pre_conv_history");
        in_values.emplace_back(Ort::Value::CreateTensor<float>(mem, &dummy_f, 0, pre_conv_shape.data(), pre_conv_shape.size()));
        in_names.push_back("latent_buffer");
        in_values.emplace_back(Ort::Value::CreateTensor<float>(mem, &dummy_f, 0, latent_shape.data(), latent_shape.size()));
        in_names.push_back("conv_history");
        in_values.emplace_back(Ort::Value::CreateTensor<float>(mem, &dummy_f, 0, conv_shape.data(), conv_shape.size()));

        std::array<int64_t, 1> is_last_shape = {1};
        float is_last_val = 1.0f;
        in_names.push_back("is_last");
        in_values.emplace_back(Ort::Value::CreateTensor<float>(
            mem, &is_last_val, 1, is_last_shape.data(), is_last_shape.size()));

        std::array<int64_t, 4> kv_shape = {1, (int64_t) num_heads_, 0, (int64_t) head_dim_};
        for (int i = 0; i < num_layers_; ++i) {
            std::string name = "past_key_" + std::to_string(i);
            in_names.push_back(input_names_[5 + (size_t) i].c_str());
            (void) name; // keep canonical naming in case exported names are canonical.
            in_values.emplace_back(Ort::Value::CreateTensor<float>(mem, &dummy_f, 0, kv_shape.data(), kv_shape.size()));
        }
        for (int i = 0; i < num_layers_; ++i) {
            std::string name = "past_value_" + std::to_string(i);
            in_names.push_back(input_names_[5 + (size_t) num_layers_ + (size_t) i].c_str());
            (void) name;
            in_values.emplace_back(Ort::Value::CreateTensor<float>(mem, &dummy_f, 0, kv_shape.data(), kv_shape.size()));
        }

        const char * out_names[2] = {
            output_names_[0].c_str(),
            output_names_[1].c_str(),
        };

        auto out = impl->session.Run(
            Ort::RunOptions{nullptr},
            in_names.data(),
            in_values.data(),
            in_values.size(),
            out_names,
            2);

        if (out.size() < 2) {
            error_msg_ = "Decoder returned insufficient outputs";
            return false;
        }

        auto wav_info = out[0].GetTensorTypeAndShapeInfo();
        size_t wav_count = wav_info.GetElementCount();
        const float * wav = out[0].GetTensorData<float>();
        int64_t valid = parse_valid_samples(out[1]);
        if (valid <= 0 || (size_t) valid > wav_count) {
            valid = (int64_t) wav_count;
        }
        audio.assign(wav, wav + valid);
        (void) dummy_i64;
        return true;
    } catch (const std::exception & e) {
        error_msg_ = std::string("Decoder inference failed: ") + e.what();
        return false;
    }
}

} // namespace qwen3_tts
