#include "audio_tokenizer_decoder.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <fstream>
#include <vector>
#include <cmath>

static bool load_binary_file(const char * path, std::vector<uint8_t> & data) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        return false;
    }
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    data.resize(size);
    f.read(reinterpret_cast<char *>(data.data()), size);
    return f.good();
}

static bool save_binary_file(const char * path, const void * data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }
    f.write(reinterpret_cast<const char *>(data), size);
    return f.good();
}

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --tokenizer <path>  Path to tokenizer GGUF file\n");
    fprintf(stderr, "  --codes <path>      Path to speech codes binary file (int64)\n");
    fprintf(stderr, "  --reference <path>  Path to reference audio binary file (float32)\n");
    fprintf(stderr, "  --output <path>     Path to save decoded audio (optional)\n");
    fprintf(stderr, "  --help              Show this help\n");
}

int main(int argc, char ** argv) {
    const char * tokenizer_path = "models/qwen3-tts-tokenizer-f16.gguf";
    const char * codes_path = "reference/speech_codes.bin";
    const char * reference_path = "reference/decoded_audio.bin";
    const char * output_path = nullptr;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "--codes") == 0 && i + 1 < argc) {
            codes_path = argv[++i];
        } else if (strcmp(argv[i], "--reference") == 0 && i + 1 < argc) {
            reference_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    printf("=== Audio Tokenizer Decoder Test ===\n\n");
    
    qwen3_tts::AudioTokenizerDecoder decoder;
    
    printf("Test 1: Load model from %s\n", tokenizer_path);
    if (!decoder.load_model(tokenizer_path)) {
        fprintf(stderr, "  FAIL: %s\n", decoder.get_error().c_str());
        return 1;
    }
    printf("  PASS: Model loaded successfully\n");
    
    auto config = decoder.get_config();
    printf("  Config: sample_rate=%d, n_codebooks=%d, codebook_size=%d\n",
           config.sample_rate, config.n_codebooks, config.codebook_size);
    printf("\n");
    
    printf("Test 2: Load speech codes from %s\n", codes_path);
    std::vector<uint8_t> codes_data;
    if (!load_binary_file(codes_path, codes_data)) {
        fprintf(stderr, "  FAIL: Could not load codes file\n");
        return 1;
    }
    
    int64_t * codes_i64 = reinterpret_cast<int64_t *>(codes_data.data());
    int n_codes = codes_data.size() / sizeof(int64_t);
    int n_frames = n_codes / config.n_codebooks;
    
    printf("  Loaded %d codes (%d frames x %d codebooks)\n", n_codes, n_frames, config.n_codebooks);
    
    std::vector<int32_t> codes_i32(n_codes);
    for (int i = 0; i < n_codes; ++i) {
        codes_i32[i] = static_cast<int32_t>(codes_i64[i]);
    }
    
    printf("  First frame codes: ");
    for (int cb = 0; cb < std::min(8, config.n_codebooks); ++cb) {
        printf("%d ", codes_i32[cb]);
    }
    printf("...\n");
    printf("  PASS: Codes loaded and converted\n\n");
    
    printf("Test 3: Decode speech codes to waveform\n");
    
    printf("  Debug: Testing single frame decode...\n");
    std::vector<float> single_samples;
    if (!decoder.decode(codes_i32.data(), 1, single_samples)) {
        fprintf(stderr, "  FAIL (single frame): %s\n", decoder.get_error().c_str());
    } else {
        printf("  Single frame: %zu samples, first 5: ", single_samples.size());
        for (int i = 0; i < 5 && i < (int)single_samples.size(); ++i) {
            printf("%.6f ", single_samples[i]);
        }
        printf("\n");
    }
    
    std::vector<float> samples;
    if (!decoder.decode(codes_i32.data(), n_frames, samples)) {
        fprintf(stderr, "  FAIL: %s\n", decoder.get_error().c_str());
        return 1;
    }
    printf("  PASS: Decoded %zu samples (%.3f seconds at %d Hz)\n",
           samples.size(), (float)samples.size() / config.sample_rate, config.sample_rate);
    
    float min_val = samples[0], max_val = samples[0], sum = 0;
    for (float s : samples) {
        min_val = std::min(min_val, s);
        max_val = std::max(max_val, s);
        sum += s;
    }
    printf("  Audio stats: min=%.4f, max=%.4f, mean=%.6f\n", min_val, max_val, sum / samples.size());
    printf("\n");
    
    if (output_path) {
        printf("Test 4: Save decoded audio to %s\n", output_path);
        if (save_binary_file(output_path, samples.data(), samples.size() * sizeof(float))) {
            printf("  PASS: Saved %zu samples\n", samples.size());
        } else {
            fprintf(stderr, "  FAIL: Could not save output file\n");
        }
        printf("\n");
    }
    
    printf("Test 5: Compare with reference audio from %s\n", reference_path);
    std::vector<uint8_t> ref_data;
    if (!load_binary_file(reference_path, ref_data)) {
        fprintf(stderr, "  SKIP: Could not load reference file\n");
    } else {
        float * ref_samples = reinterpret_cast<float *>(ref_data.data());
        int ref_n_samples = ref_data.size() / sizeof(float);
        
        printf("  Reference: %d samples\n", ref_n_samples);
        printf("  Generated: %zu samples\n", samples.size());
        
        int compare_len = std::min((int)samples.size(), ref_n_samples);
        
        double l2_sum = 0;
        double ref_sum = 0;
        double gen_sum = 0;
        double ref_sq_sum = 0;
        double gen_sq_sum = 0;
        double cross_sum = 0;
        
        for (int i = 0; i < compare_len; ++i) {
            double diff = samples[i] - ref_samples[i];
            l2_sum += diff * diff;
            ref_sum += ref_samples[i];
            gen_sum += samples[i];
            ref_sq_sum += ref_samples[i] * ref_samples[i];
            gen_sq_sum += samples[i] * samples[i];
            cross_sum += ref_samples[i] * samples[i];
        }
        
        double l2_dist = sqrt(l2_sum / compare_len);
        
        double ref_mean = ref_sum / compare_len;
        double gen_mean = gen_sum / compare_len;
        double ref_var = ref_sq_sum / compare_len - ref_mean * ref_mean;
        double gen_var = gen_sq_sum / compare_len - gen_mean * gen_mean;
        double covar = cross_sum / compare_len - ref_mean * gen_mean;
        double correlation = covar / (sqrt(ref_var) * sqrt(gen_var) + 1e-10);
        
        printf("  L2 distance (RMS): %.6f\n", l2_dist);
        printf("  Correlation: %.6f\n", correlation);
        
        if (l2_dist < 0.001) {
            printf("  PASS: L2 distance < 0.001 (excellent match)\n");
        } else if (l2_dist < 0.01) {
            printf("  PASS: L2 distance < 0.01 (good match)\n");
        } else if (l2_dist < 0.1) {
            printf("  WARN: L2 distance < 0.1 (moderate match)\n");
        } else {
            printf("  FAIL: L2 distance >= 0.1 (poor match)\n");
        }
        
        if (correlation > 0.95) {
            printf("  PASS: Correlation > 0.95 (excellent)\n");
        } else if (correlation > 0.8) {
            printf("  PASS: Correlation > 0.8 (good)\n");
        } else if (correlation > 0.5) {
            printf("  WARN: Correlation > 0.5 (moderate)\n");
        } else {
            printf("  FAIL: Correlation <= 0.5 (poor)\n");
        }
    }
    printf("\n");
    
    printf("=== All tests completed ===\n");
    return 0;
}
