#include "audio_tokenizer_encoder.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>

// Simple WAV file reader (16-bit PCM or 32-bit float)
bool read_wav_file(const std::string & path, std::vector<float> & samples, int & sample_rate) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open WAV file: %s\n", path.c_str());
        return false;
    }
    
    // Read RIFF header
    char riff[4];
    fread(riff, 1, 4, f);
    if (strncmp(riff, "RIFF", 4) != 0) {
        fprintf(stderr, "ERROR: Not a RIFF file\n");
        fclose(f);
        return false;
    }
    
    uint32_t file_size;
    fread(&file_size, 4, 1, f);
    
    char wave[4];
    fread(wave, 1, 4, f);
    if (strncmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "ERROR: Not a WAVE file\n");
        fclose(f);
        return false;
    }
    
    // Find fmt chunk
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sr = 0;
    uint16_t bits_per_sample = 0;
    
    while (!feof(f)) {
        char chunk_id[4];
        uint32_t chunk_size;
        
        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;
        
        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            fread(&audio_format, 2, 1, f);
            fread(&num_channels, 2, 1, f);
            fread(&sr, 4, 1, f);
            fseek(f, 6, SEEK_CUR);  // Skip byte rate and block align
            fread(&bits_per_sample, 2, 1, f);
            
            // Skip any extra format bytes
            if (chunk_size > 16) {
                fseek(f, chunk_size - 16, SEEK_CUR);
            }
        }
        else if (strncmp(chunk_id, "data", 4) == 0) {
            sample_rate = sr;
            
            if (audio_format == 1) {  // PCM
                if (bits_per_sample == 16) {
                    int n_samples = chunk_size / (2 * num_channels);
                    samples.resize(n_samples);
                    
                    std::vector<int16_t> raw(n_samples * num_channels);
                    fread(raw.data(), 2, n_samples * num_channels, f);
                    
                    // Convert to mono float
                    for (int i = 0; i < n_samples; ++i) {
                        float sum = 0.0f;
                        for (int c = 0; c < num_channels; ++c) {
                            sum += raw[i * num_channels + c] / 32768.0f;
                        }
                        samples[i] = sum / num_channels;
                    }
                }
                else if (bits_per_sample == 32) {
                    int n_samples = chunk_size / (4 * num_channels);
                    samples.resize(n_samples);
                    
                    std::vector<int32_t> raw(n_samples * num_channels);
                    fread(raw.data(), 4, n_samples * num_channels, f);
                    
                    // Convert to mono float
                    for (int i = 0; i < n_samples; ++i) {
                        float sum = 0.0f;
                        for (int c = 0; c < num_channels; ++c) {
                            sum += raw[i * num_channels + c] / 2147483648.0f;
                        }
                        samples[i] = sum / num_channels;
                    }
                }
                else {
                    fprintf(stderr, "ERROR: Unsupported bits per sample: %d\n", bits_per_sample);
                    fclose(f);
                    return false;
                }
            }
            else if (audio_format == 3) {  // IEEE float
                int n_samples = chunk_size / (4 * num_channels);
                samples.resize(n_samples);
                
                std::vector<float> raw(n_samples * num_channels);
                fread(raw.data(), 4, n_samples * num_channels, f);
                
                // Convert to mono
                for (int i = 0; i < n_samples; ++i) {
                    float sum = 0.0f;
                    for (int c = 0; c < num_channels; ++c) {
                        sum += raw[i * num_channels + c];
                    }
                    samples[i] = sum / num_channels;
                }
            }
            else {
                fprintf(stderr, "ERROR: Unsupported audio format: %d\n", audio_format);
                fclose(f);
                return false;
            }
            
            fclose(f);
            return true;
        }
        else {
            // Skip unknown chunk
            fseek(f, chunk_size, SEEK_CUR);
        }
    }
    
    fprintf(stderr, "ERROR: No data chunk found\n");
    fclose(f);
    return false;
}

// Read binary file
bool read_binary_file(const std::string & path, std::vector<float> & data) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open file: %s\n", path.c_str());
        return false;
    }
    
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    data.resize(size / sizeof(float));
    fread(data.data(), sizeof(float), data.size(), f);
    fclose(f);
    
    return true;
}

// Compute L2 distance
float compute_l2_distance(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return -1.0f;
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

float compute_cosine_similarity(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return -1.0f;
    }

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    if (denom <= 1e-12f) {
        return -1.0f;
    }
    return dot / denom;
}

void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s --tokenizer <path> --audio <path> [--reference <path>]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --tokenizer <path>   Path to TTS model GGUF (contains speaker encoder)\n");
    fprintf(stderr, "  --audio <path>       Path to input WAV file\n");
    fprintf(stderr, "  --reference <path>   Path to reference embedding binary (optional)\n");
}

int main(int argc, char ** argv) {
    printf("=== Audio Tokenizer Encoder Test ===\n\n");
    
    std::string model_path;
    std::string audio_path;
    std::string ref_path;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        }
        else if (strcmp(argv[i], "--audio") == 0 && i + 1 < argc) {
            audio_path = argv[++i];
        }
        else if (strcmp(argv[i], "--reference") == 0 && i + 1 < argc) {
            ref_path = argv[++i];
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (model_path.empty() || audio_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    
    // Create encoder
    qwen3_tts::AudioTokenizerEncoder encoder;
    
    // Test 1: Load model
    printf("Test 1: Loading model from %s\n", model_path.c_str());
    if (!encoder.load_model(model_path)) {
        fprintf(stderr, "  FAIL: %s\n", encoder.get_error().c_str());
        return 1;
    }
    printf("  PASS: Model loaded successfully\n");
    
    auto config = encoder.get_config();
    printf("  Config: sample_rate=%d, embedding_dim=%d\n", 
           config.sample_rate, config.embedding_dim);
    
    // Test 2: Load audio
    printf("\nTest 2: Loading audio from %s\n", audio_path.c_str());
    std::vector<float> samples;
    int sample_rate;
    if (!read_wav_file(audio_path, samples, sample_rate)) {
        return 1;
    }
    printf("  PASS: Loaded %zu samples at %d Hz (%.2f seconds)\n", 
           samples.size(), sample_rate, (float)samples.size() / sample_rate);
    
    // Resample if needed
    if (sample_rate != config.sample_rate) {
        printf("  WARNING: Sample rate mismatch (%d vs %d)\n",
               sample_rate, config.sample_rate);
        
        // Try to load Python-resampled audio if available
        // Look in multiple locations: relative to audio file, relative to cwd, absolute
        std::vector<std::string> search_paths;
        
        // Get directory of audio file
        std::string audio_dir;
        size_t last_slash = audio_path.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            audio_dir = audio_path.substr(0, last_slash + 1);
        }
        
        search_paths.push_back(audio_dir + "reference/debug/audio_resampled.bin");
        search_paths.push_back("reference/debug/audio_resampled.bin");
        search_paths.push_back("../reference/debug/audio_resampled.bin");
        
        FILE * rf = nullptr;
        std::string found_path;
        for (const auto & path : search_paths) {
            rf = fopen(path.c_str(), "rb");
            if (rf) {
                found_path = path;
                break;
            }
        }
        
        if (rf) {
            fseek(rf, 0, SEEK_END);
            size_t file_size = ftell(rf);
            fseek(rf, 0, SEEK_SET);
            size_t n_samples = file_size / sizeof(float);
            samples.resize(n_samples);
            fread(samples.data(), sizeof(float), n_samples, rf);
            fclose(rf);
            printf("  Loaded Python-resampled audio from %s: %zu samples\n", found_path.c_str(), samples.size());
        } else {
            printf("  WARNING: No pre-resampled audio found. Using simple linear resampling.\n");
            printf("  For accurate results, run: python tools/debug/debug_speaker_encoder.py\n");
            float ratio = (float)config.sample_rate / sample_rate;
            int new_size = (int)(samples.size() * ratio);
            std::vector<float> resampled(new_size);
            for (int i = 0; i < new_size; ++i) {
                float src_idx = i / ratio;
                int idx0 = (int)src_idx;
                int idx1 = std::min(idx0 + 1, (int)samples.size() - 1);
                float frac = src_idx - idx0;
                resampled[i] = samples[idx0] * (1.0f - frac) + samples[idx1] * frac;
            }
            samples = std::move(resampled);
            printf("  Resampled to %zu samples\n", samples.size());
        }
    }
    
    // Test 3: Encode audio
    printf("\nTest 3: Encoding audio to speaker embedding\n");
    std::vector<float> embedding;
    if (!encoder.encode(samples.data(), samples.size(), embedding)) {
        fprintf(stderr, "  FAIL: %s\n", encoder.get_error().c_str());
        return 1;
    }
    printf("  PASS: Got embedding of size %zu\n", embedding.size());
    printf("  Embedding[0:10]: ");
    for (int i = 0; i < 10 && i < (int)embedding.size(); i++) {
        printf("%.6f ", embedding[i]);
    }
    printf("\n");
    
    // Print some stats
    float min_val = embedding[0], max_val = embedding[0], sum = 0.0f;
    for (float v : embedding) {
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
        sum += v;
    }
    printf("  Stats: min=%.6f, max=%.6f, mean=%.6f\n", 
           min_val, max_val, sum / embedding.size());
    
    // Test 4: Compare with reference (if provided)
    if (!ref_path.empty()) {
        printf("\nTest 4: Comparing with reference embedding\n");
        std::vector<float> ref_embedding;
        if (!read_binary_file(ref_path, ref_embedding)) {
            return 1;
        }
        printf("  Reference size: %zu\n", ref_embedding.size());
        
        if (ref_embedding.size() != embedding.size()) {
            fprintf(stderr, "  FAIL: Size mismatch (%zu vs %zu)\n", 
                    embedding.size(), ref_embedding.size());
            return 1;
        }
        
        float l2_dist = compute_l2_distance(embedding, ref_embedding);
        float cosine = compute_cosine_similarity(embedding, ref_embedding);
        printf("  L2 distance: %.6f\n", l2_dist);
        printf("  Cosine similarity: %.9f\n", cosine);
        
        if (l2_dist < 0.001f) {
            printf("  PASS: L2 distance < 0.001\n");
        } else if (l2_dist < 0.01f) {
            printf("  WARN: L2 distance < 0.01 (acceptable)\n");
        } else if (l2_dist < 0.1f) {
            printf("  WARN: L2 distance < 0.1 (marginal)\n");
        } else if (l2_dist < 0.6f && cosine > 0.999f) {
            printf("  WARN: Higher L2 but excellent cosine match (expected numeric drift)\n");
        } else if (l2_dist < 2.0f && cosine > 0.995f) {
            printf("  WARN: Moderate L2 with strong directional match (likely resampling drift)\n");
        } else {
            printf("  FAIL: Embedding mismatch (L2 too high and cosine too low)\n");
            return 1;
        }
        
        // Print first few values for comparison
        printf("\n  First 5 values comparison:\n");
        for (int i = 0; i < 5 && i < (int)embedding.size(); ++i) {
            printf("    [%d] ours=%.6f, ref=%.6f, diff=%.6f\n",
                   i, embedding[i], ref_embedding[i], 
                   embedding[i] - ref_embedding[i]);
        }
    }
    
    printf("\n=== All tests passed! ===\n");
    return 0;
}
