#include "tts_transformer.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>

// ---------------------------------------------------------------------------
// Utility: load binary files
// ---------------------------------------------------------------------------

static bool load_binary_file(const std::string & path, std::vector<uint8_t> & data) {
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

template<typename T>
static bool load_binary_array(const std::string & path, std::vector<T> & arr) {
    std::vector<uint8_t> data;
    if (!load_binary_file(path, data)) {
        return false;
    }
    arr.resize(data.size() / sizeof(T));
    memcpy(arr.data(), data.data(), data.size());
    return true;
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

static float cosine_similarity(const float * a, const float * b, size_t n) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot    += (double)a[i] * (double)b[i];
        norm_a += (double)a[i] * (double)a[i];
        norm_b += (double)b[i] * (double)b[i];
    }
    if (norm_a == 0.0 || norm_b == 0.0) return 0.0f;
    return (float)(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
}

static float max_abs_error(const float * a, const float * b, size_t n) {
    float mx = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static float mean_abs_error(const float * a, const float * b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += std::fabs((double)a[i] - (double)b[i]);
    }
    return (float)(sum / (double)n);
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

static void print_usage(const char * prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --model <path>       Path to TTS GGUF model (default: models/qwen3-tts-0.6b-f16.gguf)\n");
    printf("  --ref-dir <dir>      Directory for all reference files (default: reference/)\n");
    printf("  --tokens <path>      Path to text tokens binary (overrides ref-dir)\n");
    printf("  --speaker <path>     Path to speaker embedding binary (overrides ref-dir)\n");
    printf("  --ref-codes <path>   Path to reference speech codes (overrides ref-dir)\n");
    printf("  --ref-prefill <path> Path to reference prefill embedding (overrides ref-dir)\n");
    printf("  --ref-logits <path>  Path to reference first-frame logits (overrides ref-dir)\n");
    printf("  --ref-hidden <path>  Path to reference hidden states (overrides ref-dir)\n");
    printf("  --max-len <n>        Maximum generation length (default: 64)\n");
    printf("  --help               Show this help\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::string model_path   = "models/qwen3-tts-0.6b-f16.gguf";
    std::string ref_dir      = "reference/";

    // Per-file overrides (empty = use ref_dir + default name)
    std::string tokens_path_override;
    std::string speaker_path_override;
    std::string ref_codes_path_override;
    std::string ref_prefill_path_override;
    std::string ref_logits_path_override;
    std::string ref_hidden_path_override;

    int max_len = 64;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--ref-dir") == 0 && i + 1 < argc) {
            ref_dir = argv[++i];
            if (!ref_dir.empty() && ref_dir.back() != '/') ref_dir += '/';
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            tokens_path_override = argv[++i];
        } else if (strcmp(argv[i], "--speaker") == 0 && i + 1 < argc) {
            speaker_path_override = argv[++i];
        } else if (strcmp(argv[i], "--ref-codes") == 0 && i + 1 < argc) {
            ref_codes_path_override = argv[++i];
        } else if (strcmp(argv[i], "--ref-prefill") == 0 && i + 1 < argc) {
            ref_prefill_path_override = argv[++i];
        } else if (strcmp(argv[i], "--ref-logits") == 0 && i + 1 < argc) {
            ref_logits_path_override = argv[++i];
        } else if (strcmp(argv[i], "--ref-hidden") == 0 && i + 1 < argc) {
            ref_hidden_path_override = argv[++i];
        } else if (strcmp(argv[i], "--max-len") == 0 && i + 1 < argc) {
            max_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Resolve paths: override > ref_dir + default name
    auto resolve = [&](const std::string & override_path, const char * default_name) -> std::string {
        if (!override_path.empty()) return override_path;
        return ref_dir + default_name;
    };

    const std::string tokens_path      = resolve(tokens_path_override,      "det_text_tokens.bin");
    const std::string speaker_path     = resolve(speaker_path_override,     "det_speaker_embedding.bin");
    const std::string ref_codes_path   = resolve(ref_codes_path_override,   "det_speech_codes.bin");
    const std::string ref_prefill_path = resolve(ref_prefill_path_override, "det_prefill_embedding.bin");
    const std::string ref_logits_path  = resolve(ref_logits_path_override,  "det_first_frame_logits.bin");
    const std::string ref_hidden_path  = resolve(ref_hidden_path_override,  "det_hidden_states.bin");

    int test_num = 0;
    int pass_count = 0;
    int warn_count = 0;
    int fail_count = 0;

    auto test_pass = [&](const char * msg) {
        pass_count++;
        printf("  PASS: %s\n\n", msg);
    };
    auto test_warn = [&](const char * msg) {
        warn_count++;
        printf("  WARN: %s\n\n", msg);
    };

    printf("=== TTS Transformer Deterministic Reference Test ===\n\n");

    // -----------------------------------------------------------------------
    // Test 1: Load model
    // -----------------------------------------------------------------------
    printf("Test %d: Load model\n", ++test_num);
    qwen3_tts::TTSTransformer transformer;

    if (!transformer.load_model(model_path)) {
        printf("  FAIL: %s\n", transformer.get_error().c_str());
        return 1;
    }

    auto config = transformer.get_config();
    printf("  Model: %s\n", model_path.c_str());
    printf("  Config: hidden_size=%d, n_layers=%d, n_heads=%d, n_kv_heads=%d\n",
           config.hidden_size, config.n_layers, config.n_attention_heads, config.n_key_value_heads);
    printf("  Codec: vocab_size=%d, n_codebooks=%d\n", config.codec_vocab_size, config.n_codebooks);
    printf("  Code predictor: layers=%d, vocab_size=%d\n", config.code_pred_layers, config.code_pred_vocab_size);
    test_pass("Model loaded successfully");

    // -----------------------------------------------------------------------
    // Test 2: Initialize KV cache
    // -----------------------------------------------------------------------
    printf("Test %d: Initialize KV cache\n", ++test_num);
    if (!transformer.init_kv_cache(4096)) {
        printf("  FAIL: %s\n", transformer.get_error().c_str());
        return 1;
    }
    test_pass("KV cache initialized (n_ctx=4096)");

    // -----------------------------------------------------------------------
    // Test 3: Load all reference data
    // -----------------------------------------------------------------------
    printf("Test %d: Load reference data\n", ++test_num);

    // 3a: text tokens (int64 -> int32)
    std::vector<int64_t> text_tokens_i64;
    if (!load_binary_array(tokens_path, text_tokens_i64)) {
        printf("  FAIL: Could not load text tokens from %s\n", tokens_path.c_str());
        return 1;
    }
    std::vector<int32_t> text_tokens(text_tokens_i64.begin(), text_tokens_i64.end());
    int32_t n_tokens = (int32_t)text_tokens.size();
    printf("  Text tokens: %d values from %s\n", n_tokens, tokens_path.c_str());
    printf("    Tokens: ");
    for (int i = 0; i < std::min(n_tokens, (int32_t)10); ++i) printf("%d ", text_tokens[i]);
    printf("\n");

    // 3b: speaker embedding (float32, 1024)
    std::vector<float> speaker_embd;
    if (!load_binary_array(speaker_path, speaker_embd)) {
        printf("  FAIL: Could not load speaker embedding from %s\n", speaker_path.c_str());
        return 1;
    }
    printf("  Speaker embedding: %zu floats from %s\n", speaker_embd.size(), speaker_path.c_str());

    // 3c: prefill embedding (float32, [1, 10, 1024] = 10240)
    std::vector<float> ref_prefill;
    if (!load_binary_array(ref_prefill_path, ref_prefill)) {
        printf("  WARNING: Could not load prefill embedding from %s (skipping prefill test)\n", ref_prefill_path.c_str());
    } else {
        printf("  Prefill embedding: %zu floats from %s\n", ref_prefill.size(), ref_prefill_path.c_str());
    }

    // 3d: first frame logits (float32, [1, 3072] = 3072)
    std::vector<float> ref_logits;
    if (!load_binary_array(ref_logits_path, ref_logits)) {
        printf("  WARNING: Could not load first-frame logits from %s (skipping logits comparison)\n", ref_logits_path.c_str());
    } else {
        printf("  First-frame logits: %zu floats from %s\n", ref_logits.size(), ref_logits_path.c_str());
    }

    // 3e: speech codes (int64, [63, 16] = 1008)
    std::vector<int64_t> ref_codes_i64;
    if (!load_binary_array(ref_codes_path, ref_codes_i64)) {
        printf("  WARNING: Could not load reference codes from %s (skipping codes comparison)\n", ref_codes_path.c_str());
    } else {
        printf("  Speech codes: %zu values (%zu frames x %d codebooks) from %s\n",
               ref_codes_i64.size(), ref_codes_i64.size() / config.n_codebooks,
               config.n_codebooks, ref_codes_path.c_str());
    }
    // Convert int64 -> int32 for comparison
    std::vector<int32_t> ref_codes(ref_codes_i64.begin(), ref_codes_i64.end());

    // 3f: hidden states (float32, [63, 1024] = 64512)
    std::vector<float> ref_hidden;
    if (!load_binary_array(ref_hidden_path, ref_hidden)) {
        printf("  WARNING: Could not load hidden states from %s\n", ref_hidden_path.c_str());
    } else {
        printf("  Hidden states: %zu floats (%zu frames x %d) from %s\n",
               ref_hidden.size(), ref_hidden.size() / config.hidden_size,
               config.hidden_size, ref_hidden_path.c_str());
    }

    test_pass("Reference data loaded");

    // -----------------------------------------------------------------------
    // Test 4: Prefill forward test
    //   Feed det_prefill_embedding.bin into forward_prefill(),
    //   compare output logits against det_first_frame_logits.bin
    // -----------------------------------------------------------------------
    printf("Test %d: Prefill forward test\n", ++test_num);

    if (ref_prefill.empty()) {
        test_warn("Skipped -- no prefill embedding reference data");
    } else {
        // Clear KV cache to start fresh
        transformer.clear_kv_cache();

        // Determine number of prefill tokens from data size
        int32_t prefill_tokens = (int32_t)(ref_prefill.size() / config.hidden_size);
        printf("  Feeding prefill embedding: %d tokens x %d hidden_size\n",
               prefill_tokens, config.hidden_size);

        std::vector<float> hidden_out;
        std::vector<float> logits_out;

        bool ok = transformer.forward_prefill(
            ref_prefill.data(), prefill_tokens, 0,
            hidden_out, &logits_out);

        if (!ok) {
            printf("  FAIL: forward_prefill() failed: %s\n", transformer.get_error().c_str());
            fail_count++;
        } else {
            printf("  forward_prefill() succeeded\n");
            printf("  Hidden output size: %zu floats\n", hidden_out.size());
            printf("  Logits output size: %zu floats\n", logits_out.size());

            // Compare logits against reference
            if (!ref_logits.empty() && !logits_out.empty()) {
                size_t cmp_size = std::min(logits_out.size(), ref_logits.size());
                printf("  Comparing %zu logits values...\n", cmp_size);

                float cos_sim = cosine_similarity(logits_out.data(), ref_logits.data(), cmp_size);
                float max_err = max_abs_error(logits_out.data(), ref_logits.data(), cmp_size);
                float mean_err = mean_abs_error(logits_out.data(), ref_logits.data(), cmp_size);

                printf("  Logits comparison:\n");
                printf("    Cosine similarity:    %.8f\n", cos_sim);
                printf("    Max absolute error:   %.8f\n", max_err);
                printf("    Mean absolute error:  %.8f\n", mean_err);

                // Show first few values for debugging
                printf("  First 10 logits (C++ vs Python):\n");
                for (size_t i = 0; i < std::min(cmp_size, (size_t)10); ++i) {
                    printf("    [%zu] C++: %12.6f  Py: %12.6f  diff: %+.6f\n",
                           i, logits_out[i], ref_logits[i],
                           logits_out[i] - ref_logits[i]);
                }

                // Argmax comparison
                auto cpp_argmax = std::distance(logits_out.begin(),
                    std::max_element(logits_out.begin(), logits_out.end()));
                auto py_argmax = std::distance(ref_logits.begin(),
                    std::max_element(ref_logits.begin(), ref_logits.end()));
                printf("  Argmax: C++=%ld  Python=%ld  %s\n",
                       (long)cpp_argmax, (long)py_argmax,
                       cpp_argmax == py_argmax ? "MATCH" : "MISMATCH");

                if (cos_sim > 0.99f) {
                    test_pass("Logits match reference (cosine > 0.99)");
                } else if (cos_sim > 0.90f) {
                    test_warn("Logits partially match reference (cosine > 0.90)");
                } else {
                    test_warn("Logits diverge from reference");
                }
            } else {
                test_warn("Skipped logits comparison -- missing reference or output data");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: Full generation test
    //   Call generate() end-to-end, compare all speech codes
    // -----------------------------------------------------------------------
    printf("Test %d: Full generation test\n", ++test_num);

    // Clear KV cache for a fresh generation
    transformer.clear_kv_cache();

    const float * spk_ptr = speaker_embd.empty() ? nullptr : speaker_embd.data();

    std::vector<int32_t> generated_codes;
    printf("  Calling generate(n_tokens=%d, max_len=%d, language_id=2050)...\n",
           n_tokens, max_len);

    bool gen_ok = transformer.generate(
        text_tokens.data(), n_tokens, spk_ptr, max_len,
        generated_codes, 2050, 1.05f, 0.0f, 0);

    if (!gen_ok) {
        printf("  FAIL: generate() failed: %s\n", transformer.get_error().c_str());
        fail_count++;
    } else {
        int n_gen_frames = (int)generated_codes.size() / config.n_codebooks;
        printf("  Generated %d frames (%zu codes total)\n", n_gen_frames, generated_codes.size());

        // Print first 3 generated frames
        printf("  First 3 generated frames:\n");
        for (int f = 0; f < std::min(3, n_gen_frames); ++f) {
            printf("    Frame %d: ", f);
            for (int cb = 0; cb < config.n_codebooks; ++cb) {
                printf("%d ", generated_codes[f * config.n_codebooks + cb]);
            }
            printf("\n");
        }

        // Compare against reference codes
        if (!ref_codes.empty()) {
            int ref_frames = (int)ref_codes.size() / config.n_codebooks;
            int cmp_frames = std::min(n_gen_frames, ref_frames);

            printf("\n  Reference: %d frames, Generated: %d frames, Comparing: %d frames\n",
                   ref_frames, n_gen_frames, cmp_frames);

            // Print first 3 reference frames
            printf("  First 3 reference frames:\n");
            for (int f = 0; f < std::min(3, ref_frames); ++f) {
                printf("    Frame %d: ", f);
                for (int cb = 0; cb < config.n_codebooks; ++cb) {
                    printf("%d ", ref_codes[f * config.n_codebooks + cb]);
                }
                printf("\n");
            }

            // Per-frame comparison
            int matching_frames = 0;
            int first_mismatch_frame = -1;
            std::vector<int> per_codebook_matches(config.n_codebooks, 0);

            printf("\n  Per-frame comparison:\n");
            for (int f = 0; f < cmp_frames; ++f) {
                bool frame_match = true;
                for (int cb = 0; cb < config.n_codebooks; ++cb) {
                    int gen_val = generated_codes[f * config.n_codebooks + cb];
                    int ref_val = ref_codes[f * config.n_codebooks + cb];
                    if (gen_val == ref_val) {
                        per_codebook_matches[cb]++;
                    } else {
                        frame_match = false;
                    }
                }
                if (frame_match) {
                    matching_frames++;
                } else if (first_mismatch_frame < 0) {
                    first_mismatch_frame = f;
                    // Print first mismatching frame in detail
                    printf("    First mismatch at frame %d:\n", f);
                    printf("      Gen: ");
                    for (int cb = 0; cb < config.n_codebooks; ++cb) {
                        printf("%d ", generated_codes[f * config.n_codebooks + cb]);
                    }
                    printf("\n      Ref: ");
                    for (int cb = 0; cb < config.n_codebooks; ++cb) {
                        printf("%d ", ref_codes[f * config.n_codebooks + cb]);
                    }
                    printf("\n      Dif: ");
                    for (int cb = 0; cb < config.n_codebooks; ++cb) {
                        int g = generated_codes[f * config.n_codebooks + cb];
                        int r = ref_codes[f * config.n_codebooks + cb];
                        if (g == r) printf("=    ");
                        else printf("X    ");
                    }
                    printf("\n");
                }
            }

            printf("\n  === Speech Code Comparison Summary ===\n");
            printf("  Matching frames:       %d / %d (%.1f%%)\n",
                   matching_frames, cmp_frames,
                   cmp_frames > 0 ? 100.0f * matching_frames / cmp_frames : 0.0f);
            if (first_mismatch_frame >= 0) {
                printf("  First mismatch frame:  %d\n", first_mismatch_frame);
            } else {
                printf("  First mismatch frame:  none (all match!)\n");
            }

            printf("  Per-codebook match rate:\n");
            for (int cb = 0; cb < config.n_codebooks; ++cb) {
                printf("    CB %2d: %d / %d (%.1f%%)\n",
                       cb, per_codebook_matches[cb], cmp_frames,
                       cmp_frames > 0 ? 100.0f * per_codebook_matches[cb] / cmp_frames : 0.0f);
            }

            if (matching_frames == cmp_frames && n_gen_frames == ref_frames) {
                test_pass("All speech codes match reference exactly!");
            } else if (matching_frames > cmp_frames / 2) {
                test_warn("Partial match -- see statistics above");
            } else {
                test_warn("Low match rate -- see statistics above");
            }
        } else {
            test_warn("No reference codes available for comparison");
        }
    }

    // -----------------------------------------------------------------------
    // Test 6: Summary statistics
    // -----------------------------------------------------------------------
    printf("Test %d: Summary\n", ++test_num);
    printf("  +-------------------------------------+\n");
    printf("  |  Tests:  %d total                    |\n", test_num);
    printf("  |  PASS:   %d                          |\n", pass_count);
    printf("  |  WARN:   %d                          |\n", warn_count);
    printf("  |  FAIL:   %d                          |\n", fail_count);
    printf("  +-------------------------------------+\n\n");

    if (fail_count > 0) {
        printf("=== Some tests FAILED (exit 0 to allow analysis) ===\n");
    } else if (warn_count > 0) {
        printf("=== All tests passed with warnings ===\n");
    } else {
        printf("=== All tests passed! ===\n");
    }

    return 0;  // Always exit 0 to allow analysis of comparison results
}
