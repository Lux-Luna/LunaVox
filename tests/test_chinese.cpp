#include "text_tokenizer.h"
#include "gguf_loader.h"
#include <cstdio>
#include <vector>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("Usage: %s <model_path> [text]\n", argv[0]);
        return 1;
    }
    
    const char * model_path = argv[1];
    // "你好" in UTF-8: \xE4\xBD\xA0\xE5\xA5\xBD
    std::string text = (argc > 2) ? argv[2] : "\xE4\xBD\xA0\xE5\xA5\xBD";
    
    printf("Input bytes: ");
    for (unsigned char c : text) printf("%02X ", c);
    printf("\n");
    
    qwen3_tts::GGUFLoader loader;
    if (!loader.open(model_path)) {
        printf("Failed to open model: %s\n", loader.get_error().c_str());
        return 1;
    }
    
    qwen3_tts::TextTokenizer tokenizer;
    if (!tokenizer.load_from_gguf(loader.get_ctx())) {
        printf("Failed to load tokenizer: %s\n", tokenizer.get_error().c_str());
        return 1;
    }
    
    printf("Text: '%s'\n", text.c_str());
    auto tokens = tokenizer.encode(text);
    
    printf("Tokens: [");
    for (size_t i = 0; i < tokens.size(); i++) {
        printf("%d", tokens[i]);
        if (i + 1 < tokens.size()) printf(", ");
    }
    printf("]\n");
    
    for (int32_t tid : tokens) {
        printf("  ID %d -> '%s'\n", tid, tokenizer.decode_token(tid).c_str());
    }
    
    return 0;
}
