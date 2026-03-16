#include "audio_tokenizer_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static bool read_wav_file(const std::string & path, std::vector<float> & samples, int & sample_rate) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open WAV file: %s\n", path.c_str());
        return false;
    }

    char riff[4];
    if (fread(riff, 1, 4, f) != 4 || strncmp(riff, "RIFF", 4) != 0) {
        fclose(f);
        return false;
    }

    uint32_t file_size = 0;
    fread(&file_size, 4, 1, f);
    (void) file_size;

    char wave[4];
    if (fread(wave, 1, 4, f) != 4 || strncmp(wave, "WAVE", 4) != 0) {
        fclose(f);
        return false;
    }

    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sr = 0;
    uint16_t bits_per_sample = 0;

    while (!feof(f)) {
        char chunk_id[4];
        uint32_t chunk_size = 0;
        if (fread(chunk_id, 1, 4, f) != 4 || fread(&chunk_size, 4, 1, f) != 1) {
            break;
        }

        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            fread(&audio_format, 2, 1, f);
            fread(&num_channels, 2, 1, f);
            fread(&sr, 4, 1, f);
            fseek(f, 6, SEEK_CUR);
            fread(&bits_per_sample, 2, 1, f);
            if (chunk_size > 16) {
                fseek(f, chunk_size - 16, SEEK_CUR);
            }
        } else if (strncmp(chunk_id, "data", 4) == 0) {
            sample_rate = (int) sr;
            if (audio_format == 1 && bits_per_sample == 16) {
                const int n_samples = (int) (chunk_size / (2 * num_channels));
                samples.resize((size_t) n_samples);
                std::vector<int16_t> raw((size_t) n_samples * num_channels);
                fread(raw.data(), 2, (size_t) n_samples * num_channels, f);
                for (int i = 0; i < n_samples; ++i) {
                    float sum = 0.0f;
                    for (int c = 0; c < num_channels; ++c) {
                        sum += raw[(size_t) i * num_channels + c] / 32768.0f;
                    }
                    samples[(size_t) i] = sum / num_channels;
                }
                fclose(f);
                return true;
            } else if (audio_format == 3 && bits_per_sample == 32) {
                const int n_samples = (int) (chunk_size / (4 * num_channels));
                samples.resize((size_t) n_samples);
                std::vector<float> raw((size_t) n_samples * num_channels);
                fread(raw.data(), 4, (size_t) n_samples * num_channels, f);
                for (int i = 0; i < n_samples; ++i) {
                    float sum = 0.0f;
                    for (int c = 0; c < num_channels; ++c) {
                        sum += raw[(size_t) i * num_channels + c];
                    }
                    samples[(size_t) i] = sum / num_channels;
                }
                fclose(f);
                return true;
            } else {
                fclose(f);
                return false;
            }
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    fclose(f);
    return false;
}

static std::vector<float> linear_resample(
    const std::vector<float> & input,
    int input_rate,
    int output_rate) {
    if (input_rate == output_rate) {
        return input;
    }
    const float ratio = (float) output_rate / (float) input_rate;
    const size_t out_size = (size_t) std::max<int>(1, (int) std::lround((double) input.size() * ratio));
    std::vector<float> out(out_size);
    for (size_t i = 0; i < out_size; ++i) {
        const float src = (float) i / ratio;
        const size_t idx0 = std::min((size_t) src, input.size() - 1);
        const size_t idx1 = std::min(idx0 + 1, input.size() - 1);
        const float frac = src - (float) idx0;
        out[i] = input[idx0] * (1.0f - frac) + input[idx1] * frac;
    }
    return out;
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        result = std::max(result, std::fabs(a[i] - b[i]));
    }
    return result;
}

int main(int argc, char ** argv) {
    std::string model_path = "models/qwen3-tts-0.6B-base.gguf";
    std::string audio_path = "reference/ref-audio.wav";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--audio") == 0 && i + 1 < argc) {
            audio_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [--tokenizer <gguf>] [--audio <wav>]\n", argv[0]);
            return 0;
        }
    }

    printf("=== Audio Tokenizer Encoder Smoke Test ===\n\n");

    qwen3_tts::AudioTokenizerEncoder encoder;
    if (!encoder.load_model(model_path)) {
        fprintf(stderr, "FAIL: %s\n", encoder.get_error().c_str());
        return 1;
    }

    auto config = encoder.get_config();
    std::vector<float> samples;
    int sample_rate = 0;
    if (!read_wav_file(audio_path, samples, sample_rate)) {
        fprintf(stderr, "FAIL: Could not read WAV: %s\n", audio_path.c_str());
        return 1;
    }
    samples = linear_resample(samples, sample_rate, config.sample_rate);

    std::vector<float> embedding_a;
    std::vector<float> embedding_b;
    if (!encoder.encode(samples.data(), (int32_t) samples.size(), embedding_a)) {
        fprintf(stderr, "FAIL: first encode failed: %s\n", encoder.get_error().c_str());
        return 1;
    }
    if (!encoder.encode(samples.data(), (int32_t) samples.size(), embedding_b)) {
        fprintf(stderr, "FAIL: second encode failed: %s\n", encoder.get_error().c_str());
        return 1;
    }

    if ((int32_t) embedding_a.size() != config.embedding_dim || embedding_b.size() != embedding_a.size()) {
        fprintf(stderr, "FAIL: unexpected embedding size\n");
        return 1;
    }

    double sum_sq = 0.0;
    for (float value : embedding_a) {
        if (!std::isfinite(value)) {
            fprintf(stderr, "FAIL: embedding contains non-finite values\n");
            return 1;
        }
        sum_sq += (double) value * (double) value;
    }

    const float norm = (float) std::sqrt(sum_sq);
    const float repeat_diff = max_abs_diff(embedding_a, embedding_b);

    printf("Embedding size: %zu\n", embedding_a.size());
    printf("Embedding L2 norm: %.6f\n", norm);
    printf("Repeat max abs diff: %.9f\n", repeat_diff);

    if (norm <= 1e-4f) {
        fprintf(stderr, "FAIL: embedding norm too small\n");
        return 1;
    }
    if (repeat_diff > 1e-5f) {
        fprintf(stderr, "FAIL: repeated encoding is not stable enough\n");
        return 1;
    }

    printf("PASS: encoder smoke checks succeeded\n");
    return 0;
}
