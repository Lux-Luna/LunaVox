#include "audio_tokenizer_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static bool save_binary_file(const char * path, const void * data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }
    f.write(reinterpret_cast<const char *>(data), size);
    return f.good();
}

static float compute_rms(const std::vector<float> & samples) {
    double sum_sq = 0.0;
    for (float value : samples) {
        sum_sq += (double) value * (double) value;
    }
    return samples.empty() ? 0.0f : (float) std::sqrt(sum_sq / (double) samples.size());
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        result = std::max(result, std::fabs(a[i] - b[i]));
    }
    return result;
}

static const char * default_tokenizer_path() {
    const char * candidates[] = {
        "models/qwen3-tts-tokenizer-f16.gguf",
        "models/qwen3-tts-tokenizer-q8_0.gguf",
    };

    for (const char * candidate : candidates) {
        FILE * f = fopen(candidate, "r");
        if (!f) {
            continue;
        }
        fclose(f);
        return candidate;
    }

    return candidates[0];
}

int main(int argc, char ** argv) {
    const char * tokenizer_path = default_tokenizer_path();
    const char * output_path = nullptr;
    int32_t n_frames = 6;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            n_frames = (int32_t) atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fprintf(stderr, "Usage: %s [--tokenizer <gguf>] [--frames <n>] [--output <bin>]\n", argv[0]);
            return 0;
        }
    }

    printf("=== Audio Tokenizer Decoder Smoke Test ===\n\n");

    qwen3_tts::AudioTokenizerDecoder decoder;
    if (!decoder.load_model(tokenizer_path)) {
        fprintf(stderr, "FAIL: %s\n", decoder.get_error().c_str());
        return 1;
    }

    const auto config = decoder.get_config();
    if (n_frames <= 0) {
        fprintf(stderr, "FAIL: frame count must be positive\n");
        return 1;
    }

    std::vector<int32_t> codes((size_t) n_frames * config.n_codebooks);
    for (int32_t frame = 0; frame < n_frames; ++frame) {
        for (int32_t cb = 0; cb < config.n_codebooks; ++cb) {
            const int32_t index = frame * config.n_codebooks + cb;
            codes[(size_t) index] = (frame * 37 + cb * 13) % config.codebook_size;
        }
    }

    std::vector<float> samples_a;
    std::vector<float> samples_b;
    if (!decoder.decode(codes.data(), n_frames, samples_a)) {
        fprintf(stderr, "FAIL: first decode failed: %s\n", decoder.get_error().c_str());
        return 1;
    }
    if (!decoder.decode(codes.data(), n_frames, samples_b)) {
        fprintf(stderr, "FAIL: second decode failed: %s\n", decoder.get_error().c_str());
        return 1;
    }

    if (samples_a.empty() || samples_a.size() != samples_b.size()) {
        fprintf(stderr, "FAIL: decoder produced invalid output sizes\n");
        return 1;
    }

    float min_value = samples_a[0];
    float max_value = samples_a[0];
    for (float value : samples_a) {
        if (!std::isfinite(value)) {
            fprintf(stderr, "FAIL: decoder produced non-finite values\n");
            return 1;
        }
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }

    const float rms = compute_rms(samples_a);
    const float repeat_diff = max_abs_diff(samples_a, samples_b);

    printf("Frames: %d\n", n_frames);
    printf("Samples: %zu\n", samples_a.size());
    printf("Range: [%.6f, %.6f]\n", min_value, max_value);
    printf("RMS: %.6f\n", rms);
    printf("Repeat max abs diff: %.9f\n", repeat_diff);

    if (samples_a.size() < (size_t) n_frames * 100) {
        fprintf(stderr, "FAIL: decoder produced too few samples\n");
        return 1;
    }
    if (rms <= 1e-5f) {
        fprintf(stderr, "FAIL: decoder output is effectively silent\n");
        return 1;
    }
    if (repeat_diff > 1e-5f) {
        fprintf(stderr, "FAIL: repeated decode is not stable enough\n");
        return 1;
    }

    if (output_path && !save_binary_file(output_path, samples_a.data(), samples_a.size() * sizeof(float))) {
        fprintf(stderr, "FAIL: could not save output file\n");
        return 1;
    }

    printf("PASS: decoder smoke checks succeeded\n");
    return 0;
}
