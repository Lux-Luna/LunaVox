#include "text_tokenizer.h"
#include "gguf_loader.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const int32_t HELLO_TTS_TOKENS[] = {
    151644, 77091, 198, 9707, 13, 151645, 198, 151644, 77091, 198
};
static const size_t HELLO_TTS_TOKEN_COUNT = sizeof(HELLO_TTS_TOKENS) / sizeof(HELLO_TTS_TOKENS[0]);

static void print_usage(const char * prog) {
    printf("Usage: %s [--model <path>] [--text <text>]\n", prog);
}

static void print_tokens(const char * label, const std::vector<int32_t> & tokens) {
    printf("%s [", label);
    for (size_t i = 0; i < tokens.size(); ++i) {
        printf("%d", tokens[i]);
        if (i + 1 < tokens.size()) {
            printf(", ");
        }
    }
    printf("]\n");
}

int main(int argc, char ** argv) {
    const char * model_path = "models/qwen3-tts-0.6B-base.gguf";
    const char * text = "Hello.";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
            text = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    printf("=== Text Tokenizer Smoke Test ===\n\n");

    qwen3_tts::GGUFLoader loader;
    if (!loader.open(model_path)) {
        printf("FAIL: Could not open GGUF file: %s\n", loader.get_error().c_str());
        return 1;
    }

    qwen3_tts::TextTokenizer tokenizer;
    if (!tokenizer.load_from_gguf(loader.get_ctx())) {
        printf("FAIL: Could not load tokenizer: %s\n", tokenizer.get_error().c_str());
        return 1;
    }

    const std::vector<int32_t> empty_tts_tokens = tokenizer.encode_for_tts("");
    if (empty_tts_tokens.size() != 8) {
        printf("FAIL: encode_for_tts(\"\") expected 8 tokens, got %zu\n", empty_tts_tokens.size());
        return 1;
    }

    const std::vector<int32_t> plain_tokens = tokenizer.encode(text);
    const std::vector<int32_t> tts_tokens = tokenizer.encode_for_tts(text);
    const std::string decoded = tokenizer.decode(plain_tokens);

    print_tokens("plain tokens:", plain_tokens);
    print_tokens("tts tokens  :", tts_tokens);

    if (plain_tokens.empty()) {
        printf("FAIL: plain tokenization returned no tokens\n");
        return 1;
    }
    if (decoded != text) {
        printf("FAIL: decoded text mismatch: '%s'\n", decoded.c_str());
        return 1;
    }
    if (tts_tokens.size() != plain_tokens.size() + empty_tts_tokens.size()) {
        printf("FAIL: unexpected TTS token count (%zu)\n", tts_tokens.size());
        return 1;
    }

    for (size_t i = 0; i < 3; ++i) {
        if (tts_tokens[i] != empty_tts_tokens[i]) {
            printf("FAIL: TTS prefix token mismatch at %zu\n", i);
            return 1;
        }
    }

    for (size_t i = 0; i < plain_tokens.size(); ++i) {
        if (tts_tokens[3 + i] != plain_tokens[i]) {
            printf("FAIL: embedded plain token mismatch at %zu\n", i);
            return 1;
        }
    }

    for (size_t i = 3; i < empty_tts_tokens.size(); ++i) {
        const size_t tts_idx = 3 + plain_tokens.size() + (i - 3);
        if (tts_tokens[tts_idx] != empty_tts_tokens[i]) {
            printf("FAIL: TTS suffix token mismatch at %zu\n", tts_idx);
            return 1;
        }
    }

    if (strcmp(text, "Hello.") == 0) {
        if (tts_tokens.size() != HELLO_TTS_TOKEN_COUNT) {
            printf("FAIL: Hello. expected %zu TTS tokens, got %zu\n", HELLO_TTS_TOKEN_COUNT, tts_tokens.size());
            return 1;
        }
        for (size_t i = 0; i < HELLO_TTS_TOKEN_COUNT; ++i) {
            if (tts_tokens[i] != HELLO_TTS_TOKENS[i]) {
                printf("FAIL: Hello. token mismatch at %zu\n", i);
                return 1;
            }
        }
    }

    printf("PASS: tokenizer smoke checks succeeded\n");
    return 0;
}
