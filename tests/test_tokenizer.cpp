#include "text_tokenizer.h"
#include "gguf_loader.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <fstream>
#include <vector>

// Expected tokens for "Hello." with TTS format
// Format: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
// Expected: [151644, 77091, 198, 9707, 13, 151645, 198, 151644, 77091, 198]
static const int32_t EXPECTED_TOKENS[] = {151644, 77091, 198, 9707, 13, 151645, 198, 151644, 77091, 198};
static const size_t EXPECTED_TOKEN_COUNT = 10;

void print_usage(const char * prog) {
    printf("Usage: %s --model <path_to_gguf>\n", prog);
    printf("       %s (runs basic tests without model)\n", prog);
}

int main(int argc, char ** argv) {
    printf("=== Text Tokenizer Test ===\n\n");
    
    const char * model_path = nullptr;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    qwen3_tts::TextTokenizer tokenizer;
    
    // Test 1: Check initial state
    printf("Test 1: Initial state\n");
    assert(!tokenizer.is_loaded());
    printf("  PASS: Tokenizer not loaded initially\n\n");
    
    if (!model_path) {
        printf("No model specified. Run with --model <path> for full tests.\n");
        printf("=== Basic tests passed! ===\n");
        return 0;
    }
    
    // Test 2: Load from GGUF
    printf("Test 2: Load tokenizer from GGUF\n");
    printf("  Model: %s\n", model_path);
    
    qwen3_tts::GGUFLoader loader;
    if (!loader.open(model_path)) {
        printf("  FAIL: Could not open GGUF file: %s\n", loader.get_error().c_str());
        return 1;
    }
    
    if (!tokenizer.load_from_gguf(loader.get_ctx())) {
        printf("  FAIL: Could not load tokenizer: %s\n", tokenizer.get_error().c_str());
        return 1;
    }
    
    assert(tokenizer.is_loaded());
    printf("  PASS: Tokenizer loaded successfully\n");
    printf("  Vocab size: %d\n", tokenizer.get_config().vocab_size);
    printf("  BOS token ID: %d\n", tokenizer.bos_token_id());
    printf("  EOS token ID: %d\n", tokenizer.eos_token_id());
    printf("\n");
    
    // Test 3: Encode simple text
    printf("Test 3: Encode 'Hello.'\n");
    auto tokens = tokenizer.encode("Hello.");
    printf("  Tokens: [");
    for (size_t i = 0; i < tokens.size(); i++) {
        printf("%d", tokens[i]);
        if (i + 1 < tokens.size()) printf(", ");
    }
    printf("]\n");
    
    // Check expected tokens for "Hello." (without TTS format)
    // Expected: [9707, 13] for "Hello" and "."
    if (tokens.size() >= 2 && tokens[0] == 9707 && tokens[1] == 13) {
        printf("  PASS: Tokens match expected [9707, 13]\n\n");
    } else {
        printf("  INFO: Tokens differ from expected [9707, 13]\n\n");
    }
    
    // Test 4: Encode with TTS format
    printf("Test 4: Encode 'Hello.' with TTS format\n");
    auto tts_tokens = tokenizer.encode_for_tts("Hello.");
    printf("  Tokens: [");
    for (size_t i = 0; i < tts_tokens.size(); i++) {
        printf("%d", tts_tokens[i]);
        if (i + 1 < tts_tokens.size()) printf(", ");
    }
    printf("]\n");
    printf("  Expected: [");
    for (size_t i = 0; i < EXPECTED_TOKEN_COUNT; i++) {
        printf("%d", EXPECTED_TOKENS[i]);
        if (i + 1 < EXPECTED_TOKEN_COUNT) printf(", ");
    }
    printf("]\n");
    
    // Compare with expected
    bool match = (tts_tokens.size() == EXPECTED_TOKEN_COUNT);
    if (match) {
        for (size_t i = 0; i < EXPECTED_TOKEN_COUNT; i++) {
            if (tts_tokens[i] != EXPECTED_TOKENS[i]) {
                match = false;
                break;
            }
        }
    }
    
    if (match) {
        printf("  PASS: TTS tokens match expected!\n\n");
    } else {
        printf("  FAIL: TTS tokens do not match expected\n\n");
        return 1;
    }
    
    // Test 5: Decode tokens
    printf("Test 5: Decode tokens\n");
    std::string decoded = tokenizer.decode(tokens);
    printf("  Decoded: '%s'\n", decoded.c_str());
    if (decoded == "Hello.") {
        printf("  PASS: Decoded text matches original\n\n");
    } else {
        printf("  INFO: Decoded text differs from original\n\n");
    }
    
    // Test 6: Decode single tokens
    printf("Test 6: Decode individual tokens\n");
    for (size_t i = 0; i < tts_tokens.size(); i++) {
        std::string tok_str = tokenizer.decode_token(tts_tokens[i]);
        printf("  Token %d: '%s'\n", tts_tokens[i], tok_str.c_str());
    }
    printf("\n");
    
    // Test 7: Compare with reference file if available
    printf("Test 7: Compare with reference file\n");
    std::string ref_path = "../reference/text_tokens.bin";
    std::ifstream ref_file(ref_path, std::ios::binary);
    if (ref_file.is_open()) {
        // Read int64 tokens from reference file
        std::vector<int64_t> ref_tokens;
        int64_t val;
        while (ref_file.read(reinterpret_cast<char*>(&val), sizeof(val))) {
            ref_tokens.push_back(val);
        }
        ref_file.close();
        
        printf("  Reference tokens: [");
        for (size_t i = 0; i < ref_tokens.size(); i++) {
            printf("%ld", (long)ref_tokens[i]);
            if (i + 1 < ref_tokens.size()) printf(", ");
        }
        printf("]\n");
        
        // Compare
        bool ref_match = (tts_tokens.size() == ref_tokens.size());
        if (ref_match) {
            for (size_t i = 0; i < ref_tokens.size(); i++) {
                if (tts_tokens[i] != (int32_t)ref_tokens[i]) {
                    ref_match = false;
                    break;
                }
            }
        }
        
        if (ref_match) {
            printf("  PASS: Tokens match reference file!\n\n");
        } else {
            printf("  FAIL: Tokens do not match reference file\n\n");
            return 1;
        }
    } else {
        printf("  SKIP: Reference file not found at %s\n\n", ref_path.c_str());
    }
    
    printf("=== All tests passed! ===\n");
    return 0;
}
