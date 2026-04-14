#include "audio_io.h"
#include "logger.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace lunavox {

namespace {

namespace fs = std::filesystem;

constexpr float kPi = 3.14159265358979323846f;

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

} // namespace

bool load_audio_file(const std::string & path, std::vector<float> & samples, int & sample_rate) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        LOG_ERROR("Cannot open WAV file: %s", path.c_str());
        return false;
    }

    char riff[4];
    if (std::fread(riff, 1, 4, f) != 4 || std::strncmp(riff, "RIFF", 4) != 0) {
        std::fclose(f);
        LOG_ERROR("Not a RIFF file: %s", path.c_str());
        return false;
    }
    uint32_t file_size = 0;
    if (std::fread(&file_size, 4, 1, f) != 1) {
        std::fclose(f);
        return false;
    }
    (void) file_size;
    char wave[4];
    if (std::fread(wave, 1, 4, f) != 4 || std::strncmp(wave, "WAVE", 4) != 0) {
        std::fclose(f);
        LOG_ERROR("Not a WAVE file: %s", path.c_str());
        return false;
    }

    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sr = 0;
    while (!std::feof(f)) {
        char chunk_id[4];
        uint32_t chunk_size = 0;
        if (std::fread(chunk_id, 1, 4, f) != 4) break;
        if (std::fread(&chunk_size, 4, 1, f) != 1) break;

        if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
            if (std::fread(&audio_format, 2, 1, f) != 1) break;
            if (std::fread(&num_channels, 2, 1, f) != 1) break;
            if (std::fread(&sr, 4, 1, f) != 1) break;
            std::fseek(f, 6, SEEK_CUR);
            if (std::fread(&bits_per_sample, 2, 1, f) != 1) break;
            if (chunk_size > 16) {
                std::fseek(f, chunk_size - 16, SEEK_CUR);
            }
        } else if (std::strncmp(chunk_id, "data", 4) == 0) {
            sample_rate = (int) sr;
            if (audio_format == 1 && bits_per_sample == 16) {
                int n = (int) (chunk_size / (2 * num_channels));
                std::vector<int16_t> raw((size_t) n * (size_t) num_channels);
                if (std::fread(raw.data(), 2, raw.size(), f) != raw.size()) {
                    std::fclose(f);
                    return false;
                }
                samples.assign((size_t) n, 0.0f);
                for (int i = 0; i < n; ++i) {
                    float sum = 0.0f;
                    for (int c = 0; c < num_channels; ++c) {
                        sum += raw[(size_t) i * (size_t) num_channels + (size_t) c] / 32768.0f;
                    }
                    samples[(size_t) i] = sum / (float) num_channels;
                }
            } else if (audio_format == 3 && bits_per_sample == 32) {
                int n = (int) (chunk_size / (4 * num_channels));
                std::vector<float> raw((size_t) n * (size_t) num_channels);
                if (std::fread(raw.data(), 4, raw.size(), f) != raw.size()) {
                    std::fclose(f);
                    return false;
                }
                samples.assign((size_t) n, 0.0f);
                for (int i = 0; i < n; ++i) {
                    float sum = 0.0f;
                    for (int c = 0; c < num_channels; ++c) {
                        sum += raw[(size_t) i * (size_t) num_channels + (size_t) c];
                    }
                    samples[(size_t) i] = sum / (float) num_channels;
                }
            } else {
                std::fclose(f);
                LOG_ERROR("Unsupported WAV format: audio_format=%u, bits=%u", audio_format, bits_per_sample);
                return false;
            }
            std::fclose(f);
            return true;
        } else {
            std::fseek(f, chunk_size, SEEK_CUR);
        }
    }
    std::fclose(f);
    LOG_ERROR("No data chunk found in WAV file: %s", path.c_str());
    return false;
}

bool save_audio_file(const std::string & path, const std::vector<float> & samples, int sample_rate) {
    try {
        fs::path p(path);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }
    } catch (...) {
        // Continue and let fopen report a concrete error if directory creation fails.
    }
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) {
        LOG_ERROR("Cannot create WAV file: %s", path.c_str());
        return false;
    }
    uint16_t num_channels = 1;
    uint16_t bits_per_sample = 16;
    uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    uint16_t block_align = num_channels * bits_per_sample / 8;
    uint32_t data_size = (uint32_t) samples.size() * block_align;
    uint32_t file_size = 36 + data_size;

    std::fwrite("RIFF", 1, 4, f);
    std::fwrite(&file_size, 4, 1, f);
    std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    std::fwrite(&fmt_size, 4, 1, f);
    uint16_t audio_format = 1;
    std::fwrite(&audio_format, 2, 1, f);
    std::fwrite(&num_channels, 2, 1, f);
    uint32_t sr = (uint32_t) sample_rate;
    std::fwrite(&sr, 4, 1, f);
    std::fwrite(&byte_rate, 4, 1, f);
    std::fwrite(&block_align, 2, 1, f);
    std::fwrite(&bits_per_sample, 2, 1, f);
    std::fwrite("data", 1, 4, f);
    std::fwrite(&data_size, 4, 1, f);

    for (float s : samples) {
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        int16_t pcm = (int16_t) (s * 32767.0f);
        std::fwrite(&pcm, 2, 1, f);
    }
    std::fclose(f);
    return true;
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

} // namespace lunavox
