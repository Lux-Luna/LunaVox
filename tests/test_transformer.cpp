#include "tts_transformer.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

static const int32_t HELLO_TTS_TOKENS[] = {
    151644, 77091, 198, 9707, 13, 151645, 198, 151644, 77091, 198
};
static const int32_t HELLO_TTS_TOKEN_COUNT = 10;

static bool validate_codes(
    const std::vector<int32_t> & codes,
    const qwen3_tts::tts_transformer_config & config) {
    if (codes.empty() || codes.size() % config.n_codebooks != 0) {
        return false;
    }
    for (size_t i = 0; i < codes.size(); ++i) {
        const int32_t token = codes[i];
        const int32_t codebook = (int32_t) (i % config.n_codebooks);
        const int32_t limit = (codebook == 0) ? config.codec_vocab_size : config.code_pred_vocab_size;
        if (token < 0 || token >= limit) {
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    const char * model_path = "models/qwen3-tts-0.6B-base.gguf";
    int32_t max_len = 4;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--max-len") == 0 && i + 1 < argc) {
            max_len = (int32_t) atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [--model <gguf>] [--max-len <n>]\n", argv[0]);
            return 0;
        }
    }

    printf("=== TTS Transformer Smoke Test ===\n\n");

    qwen3_tts::TTSTransformer transformer_a;
    if (!transformer_a.load_model(model_path)) {
        printf("FAIL: load_model failed: %s\n", transformer_a.get_error().c_str());
        return 1;
    }

    std::vector<int32_t> codes_a;
    if (!transformer_a.generate(
            HELLO_TTS_TOKENS,
            HELLO_TTS_TOKEN_COUNT,
            nullptr,
            max_len,
            codes_a,
            2050,
            1.05f,
            0.0f,
            1)) {
        printf("FAIL: first generate failed: %s\n", transformer_a.get_error().c_str());
        return 1;
    }

    qwen3_tts::TTSTransformer transformer_b;
    if (!transformer_b.load_model(model_path)) {
        printf("FAIL: second load_model failed: %s\n", transformer_b.get_error().c_str());
        return 1;
    }

    std::vector<int32_t> codes_b;
    if (!transformer_b.generate(
            HELLO_TTS_TOKENS,
            HELLO_TTS_TOKEN_COUNT,
            nullptr,
            max_len,
            codes_b,
            2050,
            1.05f,
            0.0f,
            1)) {
        printf("FAIL: second generate failed: %s\n", transformer_b.get_error().c_str());
        return 1;
    }

    const auto config = transformer_a.get_config();
    printf("Generated tokens: %zu\n", codes_a.size());
    printf("Frames: %zu\n", codes_a.size() / config.n_codebooks);

    if (!validate_codes(codes_a, config) || !validate_codes(codes_b, config)) {
        printf("FAIL: generated codes are invalid\n");
        return 1;
    }
    if (codes_a != codes_b) {
        printf("FAIL: repeated generation is not deterministic under greedy decoding\n");
        return 1;
    }
    if ((int32_t) (codes_a.size() / config.n_codebooks) > max_len) {
        printf("FAIL: generated more frames than requested\n");
        return 1;
    }

    printf("PASS: transformer smoke checks succeeded\n");
    return 0;
}
