#include "text_tokenizer.h"
#include "gguf_loader.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    printf("Usage: %s [model_path]\n", prog);
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

static bool verify_template(
    qwen3_tts::TextTokenizer & tokenizer,
    const std::string & text,
    const char * label) {
    const std::vector<int32_t> empty_tts_tokens = tokenizer.encode_for_tts("");
    const std::vector<int32_t> plain_tokens = tokenizer.encode(text);
    const std::vector<int32_t> tts_tokens = tokenizer.encode_for_tts(text);
    const std::string decoded = tokenizer.decode(plain_tokens);

    printf("\nCase: %s\n", label);
    print_tokens("  plain:", plain_tokens);
    print_tokens("  tts  :", tts_tokens);

    if (plain_tokens.empty()) {
        printf("FAIL: %s plain tokenization returned no tokens\n", label);
        return false;
    }
    if (decoded != text) {
        printf("FAIL: %s decoded mismatch: '%s'\n", label, decoded.c_str());
        return false;
    }
    if (tts_tokens.size() != plain_tokens.size() + empty_tts_tokens.size()) {
        printf("FAIL: %s unexpected TTS token count\n", label);
        return false;
    }

    for (size_t i = 0; i < 3; ++i) {
        if (tts_tokens[i] != empty_tts_tokens[i]) {
            printf("FAIL: %s prefix mismatch at %zu\n", label, i);
            return false;
        }
    }
    for (size_t i = 0; i < plain_tokens.size(); ++i) {
        if (tts_tokens[3 + i] != plain_tokens[i]) {
            printf("FAIL: %s embedded token mismatch at %zu\n", label, i);
            return false;
        }
    }
    for (size_t i = 3; i < empty_tts_tokens.size(); ++i) {
        const size_t idx = 3 + plain_tokens.size() + (i - 3);
        if (tts_tokens[idx] != empty_tts_tokens[i]) {
            printf("FAIL: %s suffix mismatch at %zu\n", label, idx);
            return false;
        }
    }

    return true;
}

int main(int argc, char ** argv) {
    const char * model_path = "models/qwen3-tts-0.6B-base.gguf";
    if (argc > 2 || (argc == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0))) {
        print_usage(argv[0]);
        return argc == 2 ? 0 : 1;
    }
    if (argc == 2) {
        model_path = argv[1];
    }

    qwen3_tts::GGUFLoader loader;
    if (!loader.open(model_path)) {
        printf("FAIL: open model failed: %s\n", loader.get_error().c_str());
        return 1;
    }

    qwen3_tts::TextTokenizer tokenizer;
    if (!tokenizer.load_from_gguf(loader.get_ctx())) {
        printf("FAIL: load tokenizer failed: %s\n", tokenizer.get_error().c_str());
        return 1;
    }

    if (!verify_template(tokenizer, "\xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x8C\xE4\xB8\xAD\xE6\x96\x87", "Chinese")) {
        return 1;
    }
    if (!verify_template(tokenizer, "Mixed 123, hello.", "ASCII")) {
        return 1;
    }

    printf("\nPASS: multilingual tokenizer template checks succeeded\n");
    return 0;
}
