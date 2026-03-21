#include "onnx_audio_runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <mutex>
#include <memory>
#include <numeric>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "ort_provider_injector.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace qwen3_tts {

namespace {

static constexpr float kPi = 3.14159265358979323846f;
static constexpr float kEps = 1e-9f;

struct ort_env_holder {
    Ort::Env env;
    explicit ort_env_holder(OrtLoggingLevel level) : env(level, "lunavox-ort") {}
};

std::mutex g_ort_env_mu;
OrtLoggingLevel g_requested_log_level = ORT_LOGGING_LEVEL_ERROR;
bool g_ort_env_initialized = false;

Ort::Env & get_ort_env() {
    static std::unique_ptr<ort_env_holder> holder;
    std::lock_guard<std::mutex> lock(g_ort_env_mu);
    if (!holder) {
        holder = std::make_unique<ort_env_holder>(g_requested_log_level);
        g_ort_env_initialized = true;
    }
    return holder->env;
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
    bool apply_providers,
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

        if (apply_providers) {
            apply_decoder_providers(opts);
        }
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

bool is_power_of_two(int v) {
    return v > 0 && (v & (v - 1)) == 0;
}

void fft_inplace(std::vector<std::complex<float>> & a) {
    const size_t n = a.size();
    if (n <= 1) {
        return;
    }
    size_t j = 0;
    for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    for (size_t len = 2; len <= n; len <<= 1) {
        const float angle = -2.0f * kPi / (float) len;
        const std::complex<float> wlen(std::cos(angle), std::sin(angle));
        for (size_t i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            const size_t half = len >> 1;
            for (size_t k = 0; k < half; ++k) {
                const std::complex<float> u = a[i + k];
                const std::complex<float> v = a[i + k + half] * w;
                a[i + k] = u + v;
                a[i + k + half] = u - v;
                w *= wlen;
            }
        }
    }
}

int reflect_index(int idx, int n) {
    if (n <= 1) {
        return 0;
    }
    while (idx < 0 || idx >= n) {
        if (idx < 0) {
            idx = -idx;
        } else {
            idx = 2 * n - 2 - idx;
        }
    }
    return idx;
}

int32_t find_name_index(const std::vector<std::string> & names, const std::string & target) {
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i] == target) {
            return (int32_t) i;
        }
    }
    return -1;
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

bool extract_state_tensor(
    const Ort::Value & val,
    int32_t expected_elem_type,
    std::vector<float> & out_f32,
    std::vector<uint16_t> & out_f16,
    int64_t & seq_dim) {
    out_f32.clear();
    out_f16.clear();
    seq_dim = 0;
    auto info = val.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    size_t total = info.GetElementCount();
    if (shape.size() >= 3) {
        seq_dim = shape[2] > 0 ? shape[2] : 0;
    } else if (shape.size() >= 2) {
        seq_dim = shape[1] > 0 ? shape[1] : 0;
    }
    if (total == 0) {
        return true;
    }

    ONNXTensorElementDataType actual = info.GetElementType();
    if ((int32_t) actual != expected_elem_type) {
        return false;
    }
    if (actual == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float * p = val.GetTensorData<float>();
        out_f32.assign(p, p + total);
        return true;
    }
    if (actual == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const Ort::Float16_t * p = val.GetTensorData<Ort::Float16_t>();
        out_f16.resize(total);
        std::memcpy(out_f16.data(), p, total * sizeof(uint16_t));
        return true;
    }
    if (actual == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
        const Ort::BFloat16_t * p = val.GetTensorData<Ort::BFloat16_t>();
        out_f16.resize(total);
        std::memcpy(out_f16.data(), p, total * sizeof(uint16_t));
        return true;
    }
    return false;
}

} // namespace

void set_ort_debug_log(bool enabled) {
    std::lock_guard<std::mutex> lock(g_ort_env_mu);
    if (g_ort_env_initialized) {
        // ORT env is already created; keep the current level for process safety.
        return;
    }
    g_requested_log_level = enabled ? ORT_LOGGING_LEVEL_WARNING : ORT_LOGGING_LEVEL_ERROR;
}

bool ort_debug_log_enabled() {
    std::lock_guard<std::mutex> lock(g_ort_env_mu);
    return g_requested_log_level <= ORT_LOGGING_LEVEL_WARNING;
}

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
    return create_session_impl(model_path, intra_threads, false, error_msg_, session_impl_, input_names_, output_names_) && (loaded_ = true);
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

    auto * impl = as_session(session_impl_);
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char * in_name = input_names_[0].c_str();
    const char * out_name = output_names_[0].c_str();
    std::array<const char *, 1> in_names = {in_name};
    std::array<const char *, 1> out_names = {out_name};

    auto push_unique = [](std::vector<int32_t> & values, int32_t v) {
        if (v <= 0) {
            return;
        }
        if (std::find(values.begin(), values.end(), v) == values.end()) {
            values.push_back(v);
        }
    };

    std::vector<int32_t> candidate_lengths;
    if (align_grid_enabled_ && align_mod_stride_ > 0) {
        int32_t aligned = std::max(n_samples, align_min_samples_);
        int32_t target_mod = align_mod_base_ % align_mod_stride_;
        if (target_mod < 0) {
            target_mod += align_mod_stride_;
        }
        int32_t rem = aligned % align_mod_stride_;
        if (rem < 0) {
            rem += align_mod_stride_;
        }
        aligned += (target_mod - rem + align_mod_stride_) % align_mod_stride_;

        push_unique(candidate_lengths, aligned);
        // Also try one and two strides forward in case the aligned point is still rejected.
        push_unique(candidate_lengths, aligned + align_mod_stride_);
        push_unique(candidate_lengths, aligned + align_mod_stride_ * 2);
    }

    // Keep fallback probes for non-aligned exports or future model variants.
    const std::array<int32_t, 23> pad_candidates = {
        0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 768, 1024, 1536, 2048, 4096,
    };
    for (int32_t pad : pad_candidates) {
        push_unique(candidate_lengths, n_samples + pad);
    }

    std::string last_runtime_error;
    for (size_t len_idx = 0; len_idx < candidate_lengths.size(); ++len_idx) {
        const int32_t effective_samples = candidate_lengths[len_idx];
        std::vector<float> wav((size_t) effective_samples, 0.0f);
        std::memcpy(wav.data(), samples, (size_t) n_samples * sizeof(float));

        try {
            std::array<int64_t, 2> in_shape = {1, (int64_t) effective_samples};
            Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
                mem, wav.data(), wav.size(), in_shape.data(), in_shape.size());
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
            last_runtime_error = e.what();
            const bool maybe_reshape_mismatch =
                last_runtime_error.find("Reshape") != std::string::npos ||
                last_runtime_error.find("input_shape_size == size was false") != std::string::npos ||
                last_runtime_error.find("BroadcastIterator::Init") != std::string::npos;
            if (maybe_reshape_mismatch && (len_idx + 1) < candidate_lengths.size()) {
                continue;
            }
            error_msg_ = std::string("Codec encoder inference failed: ") + e.what();
            return false;
        }
    }

    error_msg_ =
        "Codec encoder inference failed after retrying candidate lengths: " + last_runtime_error;
    return false;
}

bool SpeakerEncoderOnnx::load_model(const std::string & model_path, int32_t intra_threads) {
    unload_model();
    error_msg_.clear();
    return create_session_impl(model_path, intra_threads, false, error_msg_, session_impl_, input_names_, output_names_) && (loaded_ = true);
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
    if (!ensure_mel_kernels()) {
        return false;
    }
    if (!is_power_of_two(cfg_.n_fft)) {
        error_msg_ = "Speaker mel requires n_fft as power-of-two";
        return false;
    }

    const int padding = (cfg_.n_fft - cfg_.hop_length) / 2;
    const int padded_len = n_samples + 2 * padding;
    if (padded_len < cfg_.n_fft) {
        error_msg_ = "Audio too short for mel extraction";
        return false;
    }

    std::vector<float> wav((size_t) padded_len);
    for (int i = 0; i < padded_len; ++i) {
        const int src_idx = reflect_index(i - padding, n_samples);
        wav[(size_t) i] = samples[src_idx];
    }

    n_frames = 1 + (padded_len - cfg_.n_fft) / cfg_.hop_length;
    if (n_frames <= 0) {
        error_msg_ = "Failed to compute mel frame count";
        return false;
    }

    const int n_fft_bins = cfg_.n_fft / 2 + 1;
    mel.assign((size_t) n_frames * (size_t) cfg_.n_mels, 0.0f);
    std::vector<std::complex<float>> frame_fft((size_t) cfg_.n_fft, std::complex<float>(0.0f, 0.0f));
    std::vector<float> mag((size_t) n_fft_bins, 0.0f);

    for (int f = 0; f < n_frames; ++f) {
        int start = f * cfg_.hop_length;
        for (int i = 0; i < cfg_.n_fft; ++i) {
            frame_fft[(size_t) i] = std::complex<float>(
                wav[(size_t) (start + i)] * window_[(size_t) i],
                0.0f);
        }

        fft_inplace(frame_fft);
        for (int k = 0; k < n_fft_bins; ++k) {
            const float re = frame_fft[(size_t) k].real();
            const float im = frame_fft[(size_t) k].imag();
            mag[(size_t) k] = std::sqrt(re * re + im * im + kEps);
        }

        for (int m = 0; m < cfg_.n_mels; ++m) {
            float sum = 0.0f;
            const float * fb = &mel_filterbank_[(size_t) m * (size_t) n_fft_bins];
            for (int k = 0; k < n_fft_bins; ++k) {
                sum += fb[(size_t) k] * mag[(size_t) k];
            }
            const float log_mel = std::log(std::max(sum, 1e-5f));
            mel[(size_t) f * (size_t) cfg_.n_mels + (size_t) m] = log_mel;
        }
    }
    return true;
}

bool SpeakerEncoderOnnx::ensure_mel_kernels() {
    if (mel_filterbank_.empty()) {
        compute_mel_filterbank_slaney(cfg_, mel_filterbank_);
    }
    if (window_.empty()) {
        compute_centered_periodic_hann(cfg_, window_);
    }
    if (mel_filterbank_.empty() || window_.empty()) {
        error_msg_ = "Failed to initialize speaker mel kernels";
        return false;
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
    if (!create_session_impl(model_path, intra_threads, true, error_msg_, session_impl_, input_names_, output_names_)) {
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

    int32_t n_keys = 0;
    int32_t n_vals = 0;
    for (const auto & n : input_names_) {
        if (n.rfind("past_key_", 0) == 0) {
            ++n_keys;
        } else if (n.rfind("past_value_", 0) == 0) {
            ++n_vals;
        }
    }
    num_layers_ = std::min(n_keys, n_vals);
    if (num_layers_ <= 0) {
        error_msg_ = "Decoder ONNX does not expose KV-cache inputs";
        unload_model();
        return false;
    }

    try {
        auto * impl = as_session(session_impl_);
        int32_t key_name_idx = find_name_index(input_names_, "past_key_0");
        if (key_name_idx < 0) {
            key_name_idx = 5;
        }
        size_t key_input_idx = (size_t) key_name_idx;
        auto ti = impl->session.GetInputTypeInfo(key_input_idx).GetTensorTypeAndShapeInfo();
        auto shape = ti.GetShape();
        kv_elem_type_ = (int32_t) ti.GetElementType();
        if (shape.size() >= 4) {
            num_heads_ = (int32_t) (shape[1] > 0 ? shape[1] : 8);
            head_dim_ = (int32_t) (shape[3] > 0 ? shape[3] : 64);
        } else {
            num_heads_ = 8;
            head_dim_ = 64;
        }

        auto read_channel = [&](const char * name, int32_t fallback) {
            int32_t idx = find_name_index(input_names_, name);
            if (idx < 0) return fallback;
            auto ti2 = impl->session.GetInputTypeInfo((size_t) idx).GetTensorTypeAndShapeInfo();
            auto sh = ti2.GetShape();
            if (sh.size() >= 2 && sh[1] > 0) {
                return (int32_t) sh[1];
            }
            return fallback;
        };
        pre_conv_channels_ = read_channel("pre_conv_history", 512);
        latent_channels_ = read_channel("latent_buffer", 1024);
        conv_channels_ = read_channel("conv_history", 1024);
        int32_t pre_idx = find_name_index(input_names_, "pre_conv_history");
        if (pre_idx >= 0) {
            auto pinfo = impl->session.GetInputTypeInfo((size_t) pre_idx).GetTensorTypeAndShapeInfo();
            state_elem_type_ = (int32_t) pinfo.GetElementType();
        } else {
            state_elem_type_ = (int32_t) ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        }
    } catch (...) {
        num_heads_ = 8;
        head_dim_ = 64;
        pre_conv_channels_ = 512;
        latent_channels_ = 1024;
        conv_channels_ = 1024;
        state_elem_type_ = (int32_t) ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        kv_elem_type_ = (int32_t) ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
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
    state_elem_type_ = (int32_t) ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    kv_elem_type_ = (int32_t) ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
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
        static float dummy_f = 0.0f;
        static Ort::Float16_t dummy_f16 = Ort::Float16_t::FromBits(0);
        static Ort::BFloat16_t dummy_bf16 = Ort::BFloat16_t::FromBits(0);

        auto tensor_type_supported = [](int32_t elem_type) -> bool {
            return elem_type == (int32_t) ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                   elem_type == (int32_t) ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
                   elem_type == (int32_t) ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
        };
        if (!tensor_type_supported(state_elem_type_) || !tensor_type_supported(kv_elem_type_)) {
            error_msg_ = "Decoder state tensor type is unsupported";
            return false;
        }

        state_buffer state;
        state.pre_conv_history.elem_type = state_elem_type_;
        state.latent_buffer.elem_type = state_elem_type_;
        state.conv_history.elem_type = state_elem_type_;
        state.past_keys.resize((size_t) num_layers_);
        state.past_values.resize((size_t) num_layers_);
        for (int i = 0; i < num_layers_; ++i) {
            state.past_keys[(size_t) i].elem_type = kv_elem_type_;
            state.past_values[(size_t) i].elem_type = kv_elem_type_;
        }

        std::vector<int32_t> past_key_input_idx((size_t) num_layers_);
        std::vector<int32_t> past_value_input_idx((size_t) num_layers_);
        for (int i = 0; i < num_layers_; ++i) {
            std::string key_name = "past_key_" + std::to_string(i);
            int32_t key_idx = find_name_index(input_names_, key_name);
            if (key_idx < 0) {
                key_idx = 5 + i;
            }
            past_key_input_idx[(size_t) i] = key_idx;

            std::string value_name = "past_value_" + std::to_string(i);
            int32_t value_idx = find_name_index(input_names_, value_name);
            if (value_idx < 0) {
                value_idx = 5 + num_layers_ + i;
            }
            past_value_input_idx[(size_t) i] = value_idx;
        }

        std::vector<const char *> out_names;
        out_names.reserve(output_names_.size());
        for (const auto & n : output_names_) {
            out_names.push_back(n.c_str());
        }

        std::vector<int64_t> codes_i64;
        codes_i64.reserve((size_t) std::max(1, decode_chunk_frames_) * 16);

        std::vector<const char *> in_names;
        std::vector<Ort::Value> in_values;
        in_names.reserve((size_t) (5 + 2 * num_layers_));
        in_values.reserve((size_t) (5 + 2 * num_layers_));

        audio.clear();
        audio.reserve((size_t) std::max(1, n_frames) * 1920 + 4096);

        auto add_state_tensor = [&](const char * name,
                                    state_buffer::tensor_state & ts,
                                    const int64_t * shape,
                                    size_t rank) -> bool {
            in_names.push_back(name);
            if (ts.elem_type == (int32_t) ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                float * ptr = ts.f32.empty() ? &dummy_f : ts.f32.data();
                const size_t count = ts.f32.empty() ? 0 : ts.f32.size();
                in_values.emplace_back(Ort::Value::CreateTensor<float>(mem, ptr, count, shape, rank));
                return true;
            }
            if (ts.elem_type == (int32_t) ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                Ort::Float16_t * ptr = ts.f16.empty()
                    ? &dummy_f16
                    : reinterpret_cast<Ort::Float16_t *>(ts.f16.data());
                const size_t count = ts.f16.empty() ? 0 : ts.f16.size();
                in_values.emplace_back(Ort::Value::CreateTensor<Ort::Float16_t>(mem, ptr, count, shape, rank));
                return true;
            }
            if (ts.elem_type == (int32_t) ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
                Ort::BFloat16_t * ptr = ts.f16.empty()
                    ? &dummy_bf16
                    : reinterpret_cast<Ort::BFloat16_t *>(ts.f16.data());
                const size_t count = ts.f16.empty() ? 0 : ts.f16.size();
                in_values.emplace_back(Ort::Value::CreateTensor<Ort::BFloat16_t>(mem, ptr, count, shape, rank));
                return true;
            }
            return false;
        };

        for (int32_t frame_offset = 0; frame_offset < n_frames; frame_offset += decode_chunk_frames_) {
            const int32_t chunk_frames = std::min(decode_chunk_frames_, n_frames - frame_offset);
            const bool is_last_chunk = (frame_offset + chunk_frames) >= n_frames;

            const size_t codes_count = (size_t) chunk_frames * 16;
            if (codes_i64.size() != codes_count) {
                codes_i64.resize(codes_count);
            }
            for (int32_t i = 0; i < chunk_frames; ++i) {
                for (int32_t q = 0; q < 16; ++q) {
                    const size_t src_idx = (size_t) (frame_offset + i) * 16 + (size_t) q;
                    const size_t dst_idx = (size_t) i * 16 + (size_t) q;
                    codes_i64[dst_idx] = (int64_t) codes[src_idx];
                }
            }

            in_names.clear();
            in_values.clear();

            auto add_i64_tensor = [&](const char * name, int64_t * ptr, size_t count, const int64_t * shape, size_t rank) {
                in_names.push_back(name);
                in_values.emplace_back(Ort::Value::CreateTensor<int64_t>(
                    mem, ptr, count, shape, rank));
            };

            std::array<int64_t, 3> codes_shape = {1, (int64_t) chunk_frames, 16};
            add_i64_tensor("audio_codes", codes_i64.data(), codes_i64.size(), codes_shape.data(), codes_shape.size());

            std::array<int64_t, 3> pre_shape = {1, (int64_t) pre_conv_channels_, state.pre_conv_history.seq};
            if (!add_state_tensor("pre_conv_history", state.pre_conv_history, pre_shape.data(), pre_shape.size())) {
                error_msg_ = "Decoder pre_conv_history input type is unsupported";
                return false;
            }

            std::array<int64_t, 3> latent_shape = {1, (int64_t) latent_channels_, state.latent_buffer.seq};
            if (!add_state_tensor("latent_buffer", state.latent_buffer, latent_shape.data(), latent_shape.size())) {
                error_msg_ = "Decoder latent_buffer input type is unsupported";
                return false;
            }

            std::array<int64_t, 3> conv_shape = {1, (int64_t) conv_channels_, state.conv_history.seq};
            if (!add_state_tensor("conv_history", state.conv_history, conv_shape.data(), conv_shape.size())) {
                error_msg_ = "Decoder conv_history input type is unsupported";
                return false;
            }

            std::array<int64_t, 1> is_last_shape = {1};
            float is_last_val = is_last_chunk ? 1.0f : 0.0f;
            in_names.push_back("is_last");
            in_values.emplace_back(Ort::Value::CreateTensor<float>(
                mem, &is_last_val, 1, is_last_shape.data(), is_last_shape.size()));

            for (int i = 0; i < num_layers_; ++i) {
                const int32_t idx = past_key_input_idx[(size_t) i];
                std::array<int64_t, 4> kv_shape = {
                    1,
                    (int64_t) num_heads_,
                    state.past_keys[(size_t) i].seq,
                    (int64_t) head_dim_};
                if (!add_state_tensor(input_names_[(size_t) idx].c_str(), state.past_keys[(size_t) i], kv_shape.data(), kv_shape.size())) {
                    error_msg_ = "Decoder past_key input type is unsupported";
                    return false;
                }
            }
            for (int i = 0; i < num_layers_; ++i) {
                const int32_t idx = past_value_input_idx[(size_t) i];
                std::array<int64_t, 4> kv_shape = {
                    1,
                    (int64_t) num_heads_,
                    state.past_values[(size_t) i].seq,
                    (int64_t) head_dim_};
                if (!add_state_tensor(input_names_[(size_t) idx].c_str(), state.past_values[(size_t) i], kv_shape.data(), kv_shape.size())) {
                    error_msg_ = "Decoder past_value input type is unsupported";
                    return false;
                }
            }
            auto out = impl->session.Run(
                Ort::RunOptions{nullptr},
                in_names.data(),
                in_values.data(),
                in_values.size(),
                out_names.data(),
                out_names.size());

            if (out.size() < (size_t) (5 + 2 * num_layers_)) {
                error_msg_ = "Decoder returned insufficient outputs";
                return false;
            }
            auto wav_info = out[0].GetTensorTypeAndShapeInfo();
            if (wav_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                error_msg_ = "Decoder wav output must be float32";
                return false;
            }
            const size_t wav_count = wav_info.GetElementCount();
            const float * wav = out[0].GetTensorData<float>();
            int64_t valid = parse_valid_samples(out[1]);
            if (valid <= 0 || (size_t) valid > wav_count) {
                valid = (int64_t) wav_count;
            }
            const size_t append_count = is_last_chunk ? wav_count : (size_t) valid;
            if (append_count > 0) {
                const size_t old_size = audio.size();
                audio.resize(old_size + append_count);
                std::memcpy(audio.data() + old_size, wav, append_count * sizeof(float));
            }

            if (!extract_state_tensor(
                    out[2],
                    state.pre_conv_history.elem_type,
                    state.pre_conv_history.f32,
                    state.pre_conv_history.f16,
                    state.pre_conv_history.seq)) {
                error_msg_ = "Decoder next_pre_conv_history type is unsupported";
                return false;
            }
            if (!extract_state_tensor(
                    out[3],
                    state.latent_buffer.elem_type,
                    state.latent_buffer.f32,
                    state.latent_buffer.f16,
                    state.latent_buffer.seq)) {
                error_msg_ = "Decoder next_latent_buffer type is unsupported";
                return false;
            }
            if (!extract_state_tensor(
                    out[4],
                    state.conv_history.elem_type,
                    state.conv_history.f32,
                    state.conv_history.f16,
                    state.conv_history.seq)) {
                error_msg_ = "Decoder next_conv_history type is unsupported";
                return false;
            }
            for (int i = 0; i < num_layers_; ++i) {
                if (!extract_state_tensor(
                        out[(size_t) 5 + (size_t) i],
                        state.past_keys[(size_t) i].elem_type,
                        state.past_keys[(size_t) i].f32,
                        state.past_keys[(size_t) i].f16,
                        state.past_keys[(size_t) i].seq)) {
                    error_msg_ = "Decoder next_key type is unsupported";
                    return false;
                }
            }
            for (int i = 0; i < num_layers_; ++i) {
                if (!extract_state_tensor(
                        out[(size_t) 5 + (size_t) num_layers_ + (size_t) i],
                        state.past_values[(size_t) i].elem_type,
                        state.past_values[(size_t) i].f32,
                        state.past_values[(size_t) i].f16,
                        state.past_values[(size_t) i].seq)) {
                    error_msg_ = "Decoder next_value type is unsupported";
                    return false;
                }
            }
        }
        return !audio.empty();
    } catch (const std::exception & e) {
        error_msg_ = std::string("Decoder inference failed: ") + e.what();
        return false;
    }
}

} // namespace qwen3_tts
