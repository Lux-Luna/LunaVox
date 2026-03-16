#include "text_tokenizer.h"
#include "gguf_loader.h"

#include <cstdio>
#include <string>
#include <vector>

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
    const char * model_path = (argc > 1) ? argv[1] : "models/qwen3-tts-0.6B-base.gguf";
    const std::string text = "\xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x8C\xE4\xB8\xAD\xE6\x96\x87";

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

    const std::vector<int32_t> text_tokens = tokenizer.encode(text);
    const std::vector<int32_t> tts_tokens = tokenizer.encode_for_tts(text);
    const std::vector<int32_t> empty_tts_tokens = tokenizer.encode_for_tts("");

    if (empty_tts_tokens.size() != 8) {
        printf("FAIL: encode_for_tts(\"\") expected 8 tokens, got %zu\n", empty_tts_tokens.size());
        return 1;
    }

    std::vector<int32_t> expected;
    expected.reserve(text_tokens.size() + 8);
    expected.insert(expected.end(), empty_tts_tokens.begin(), empty_tts_tokens.begin() + 3);
    expected.insert(expected.end(), text_tokens.begin(), text_tokens.end());
    expected.insert(expected.end(), empty_tts_tokens.begin() + 3, empty_tts_tokens.end());

    print_tokens("text tokens:", text_tokens);
    print_tokens("tts tokens:", tts_tokens);
    print_tokens("expected  :", expected);

    if (tts_tokens != expected) {
        printf("FAIL: Chinese TTS template tokens mismatch\n");
        return 1;
    }

    printf("PASS: Chinese token template matches <|im_start|>assistant\\n{text}<|im_end|>\\n<|im_start|>assistant\\n\n");
    return 0;
}
