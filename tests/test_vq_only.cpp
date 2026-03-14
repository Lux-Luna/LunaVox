#include "audio_tokenizer_decoder.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cmath>

int main() {
    // Load model
    qwen3_tts::AudioTokenizerDecoder decoder;
    if (!decoder.load_model("models/qwen3-tts-tokenizer-f16.gguf")) {
        fprintf(stderr, "Failed to load model: %s\n", decoder.get_error().c_str());
        return 1;
    }
    printf("Model loaded\n");
    
    // Load codes
    std::ifstream f("reference/speech_codes.bin", std::ios::binary);
    std::vector<int64_t> codes_i64(63 * 16);
    f.read(reinterpret_cast<char*>(codes_i64.data()), codes_i64.size() * sizeof(int64_t));
    f.close();
    
    std::vector<int32_t> codes(63 * 16);
    for (int i = 0; i < 63 * 16; ++i) {
        codes[i] = static_cast<int32_t>(codes_i64[i]);
    }
    printf("Codes loaded: first code = %d\n", codes[0]);
    
    // Decode
    std::vector<float> samples;
    if (!decoder.decode(codes.data(), 63, samples)) {
        fprintf(stderr, "Failed to decode: %s\n", decoder.get_error().c_str());
        return 1;
    }
    printf("Decoded %zu samples\n", samples.size());
    printf("First 5 samples: %.6f %.6f %.6f %.6f %.6f\n",
           samples[0], samples[1], samples[2], samples[3], samples[4]);
    
    // Load reference
    std::ifstream ref_f("reference/decoded_audio.bin", std::ios::binary | std::ios::ate);
    size_t ref_size = ref_f.tellg();
    ref_f.seekg(0);
    std::vector<float> ref_samples(ref_size / sizeof(float));
    ref_f.read(reinterpret_cast<char*>(ref_samples.data()), ref_size);
    ref_f.close();
    
    printf("Reference: %zu samples\n", ref_samples.size());
    printf("Reference first 5: %.6f %.6f %.6f %.6f %.6f\n",
           ref_samples[0], ref_samples[1], ref_samples[2], ref_samples[3], ref_samples[4]);
    
    // Compute correlation
    size_t n = std::min(samples.size(), ref_samples.size());
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
    for (size_t i = 0; i < n; ++i) {
        sum_x += samples[i];
        sum_y += ref_samples[i];
        sum_xy += samples[i] * ref_samples[i];
        sum_x2 += samples[i] * samples[i];
        sum_y2 += ref_samples[i] * ref_samples[i];
    }
    double mean_x = sum_x / n;
    double mean_y = sum_y / n;
    double var_x = sum_x2 / n - mean_x * mean_x;
    double var_y = sum_y2 / n - mean_y * mean_y;
    double cov = sum_xy / n - mean_x * mean_y;
    double corr = cov / (sqrt(var_x) * sqrt(var_y) + 1e-10);
    
    printf("Correlation: %.6f\n", corr);
    
    return 0;
}
