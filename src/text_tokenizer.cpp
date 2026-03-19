#include "text_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>

namespace qwen3_tts {

// GPT-2 byte-to-unicode mapping
// Maps bytes 0-255 to unicode characters to avoid control characters
static const char * BYTE_TO_UNICODE[256] = {
    "膧", "膩", "膫", "膬", "膭", "膮", "膯", "膰", "膱", "膲", "膴", "膵", "膶", "膷", "膸", "膹",
    "膼", "膽", "膾", "膿", "臄", "臅", "臇", "臈", "臉", "臋", "臍", "臎", "臏", "臐", "臑", "臒",
    "臓", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?",
    "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_",
    "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "摹",
    "蘑", "模", "膜", "磨", "摩", "魔", "抹", "末", "莫", "墨", "默", "沫", "漠", "寞", "陌", "谋",
    "牟", "某", "拇", "牡", "亩", "姆", "母", "墓", "暮", "幕", "募", "慕", "木", "目", "艀", "艁",
    "艂", "隆", "垄", "拢", "陇", "楼", "娄", "搂", "篓", "漏", "陋", "芦", "卢", "艃", "庐", "炉",
    "掳", "卤", "虏", "鲁", "麓", "碌", "露", "路", "赂", "鹿", "潞", "禄", "录", "陆", "戮", "驴",
    "脌", "脕", "脗", "脙", "脛", "脜", "脝", "脟", "脠", "脡", "脢", "脣", "脤", "脥", "脦", "脧",
    "脨", "脩", "脪", "脫", "脭", "脮", "脰", "脳", "脴", "脵", "脷", "脹", "脺", "脻", "脼", "脽",
    "脿", "谩", "芒", "茫", "盲", "氓", "忙", "莽", "猫", "茅", "锚", "毛", "矛", "铆", "卯", "茂",
    "冒", "帽", "貌", "贸", "么", "玫", "枚", "梅", "酶", "霉", "煤", "没", "眉", "媒", "镁", "每"
};

// Build reverse mapping at runtime
static std::unordered_map<std::string, uint8_t> build_unicode_to_byte() {
    std::unordered_map<std::string, uint8_t> result;
    for (int i = 0; i < 256; i++) {
        result[BYTE_TO_UNICODE[i]] = (uint8_t)i;
    }
    return result;
}

static const std::unordered_map<std::string, uint8_t> UNICODE_TO_BYTE = build_unicode_to_byte();

TextTokenizer::TextTokenizer() = default;

TextTokenizer::~TextTokenizer() = default;

size_t TextTokenizer::utf8_len(char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1; // Invalid UTF-8, treat as single byte
}

std::string TextTokenizer::bytes_to_unicode(const std::string & text) {
    std::string result;
    for (unsigned char c : text) {
        result += BYTE_TO_UNICODE[c];
    }
    return result;
}

std::string TextTokenizer::unicode_to_bytes(const std::string & text) {
    std::string result;
    size_t i = 0;
    while (i < text.size()) {
        size_t len = utf8_len(text[i]);
        std::string ch = text.substr(i, len);
        auto it = UNICODE_TO_BYTE.find(ch);
        if (it != UNICODE_TO_BYTE.end()) {
            result += (char)it->second;
        } else {
            // Not in mapping, keep as-is (shouldn't happen for valid tokens)
            result += ch;
        }
        i += len;
    }
    return result;
}

bool TextTokenizer::is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||   // CJK Unified Ideographs
           (cp >= 0x3400 && cp <= 0x4DBF) ||   // CJK Unified Ideographs Extension A
           (cp >= 0x20000 && cp <= 0x2A6DF) || // CJK Unified Ideographs Extension B
           (cp >= 0xF900 && cp <= 0xFAFF) ||   // CJK Compatibility Ideographs
           (cp >= 0x2F800 && cp <= 0x2FA1F);   // CJK Compatibility Ideographs Supplement
}

std::vector<std::string> TextTokenizer::split_regex_equivalent(const std::string & text) const {
    std::vector<std::string> words;
    std::string current;
    
    auto flush = [&]() {
        if (!current.empty()) {
            words.push_back(current);
            current.clear();
        }
    };

    size_t i = 0;
    while (i < text.size()) {
        uint32_t cp = 0;
        size_t len = 0;
        unsigned char c = (unsigned char)text[i];
        
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) { 
            cp = ((c & 0x1F) << 6) | (text[i+1] & 0x3F); len = 2; 
        }
        else if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) { 
            cp = ((c & 0x0F) << 12) | ((text[i+1] & 0x3F) << 6) | (text[i+2] & 0x3F); len = 3; 
        }
        else if ((c & 0xF8) == 0xF0 && i + 3 < text.size()) { 
            cp = ((c & 0x07) << 18) | ((text[i+1] & 0x3F) << 12) | ((text[i+2] & 0x3F) << 6) | (text[i+3] & 0x3F); len = 4; 
        }
        else { len = 1; cp = c; } // Invalid

        bool is_alnum_cjk = (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || (cp >= '0' && cp <= '9') || is_cjk(cp);

        if (isspace(c)) {
            flush();
            std::string spaces;
            while (i < text.size() && isspace((unsigned char)text[i])) {
                spaces += text[i];
                i++;
            }
            words.push_back(spaces);
            continue;
        } else if (is_alnum_cjk) {
            current += text.substr(i, len);
        } else {
            // Punctuation: isolate unless it's a known cluster
            flush();
            words.push_back(text.substr(i, len));
        }
        i += len;
    }
    flush();
    
    return words;
}

bool TextTokenizer::load_from_gguf(struct gguf_context * ctx) {
    if (!ctx) {
        error_msg_ = "GGUF context is null";
        return false;
    }
    
    // Get vocabulary
    int64_t tokens_key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (tokens_key < 0) {
        error_msg_ = "tokenizer.ggml.tokens not found in GGUF";
        return false;
    }
    
    size_t n_vocab = gguf_get_arr_n(ctx, tokens_key);
    if (n_vocab == 0) {
        error_msg_ = "Empty vocabulary";
        return false;
    }
    
    config_.vocab_size = (int32_t)n_vocab;
    id_to_token_.resize(n_vocab);
    
    for (size_t i = 0; i < n_vocab; i++) {
        const char * token = gguf_get_arr_str(ctx, tokens_key, i);
        if (token) {
            id_to_token_[i] = token;
            vocab_[token] = (int32_t)i;
            if (i < 10 || (i >= 151640 && i <= 151650)) {
                // printf("    Token %zu: '%s'\n", i, token);
            }
        }
    }
    
    // Get merges
    int64_t merges_key = gguf_find_key(ctx, "tokenizer.ggml.merges");
    if (merges_key >= 0) {
        size_t n_merges = gguf_get_arr_n(ctx, merges_key);
        printf("  Loading %zu merges...\n", n_merges);
        for (size_t i = 0; i < n_merges; i++) {
            const char * merge = gguf_get_arr_str(ctx, merges_key, i);
            if (merge) {
                std::string merge_str(merge);
                size_t space_pos = merge_str.find(' ');
                if (space_pos != std::string::npos) {
                    std::string first = merge_str.substr(0, space_pos);
                    std::string second = merge_str.substr(space_pos + 1);
                    bpe_ranks_[{first, second}] = (int32_t)i;
                }
            }
        }
    }
    
    // Get special token IDs (optional, use defaults if not found)
    int64_t bos_key = gguf_find_key(ctx, "tokenizer.ggml.bos_token_id");
    if (bos_key >= 0) {
        config_.bos_token_id = (int32_t)gguf_get_val_u32(ctx, bos_key);
    }
    
    int64_t eos_key = gguf_find_key(ctx, "tokenizer.ggml.eos_token_id");
    if (eos_key >= 0) {
        config_.eos_token_id = (int32_t)gguf_get_val_u32(ctx, eos_key);
    }
    
    int64_t pad_key = gguf_find_key(ctx, "tokenizer.ggml.padding_token_id");
    if (pad_key >= 0) {
        config_.pad_token_id = (int32_t)gguf_get_val_u32(ctx, pad_key);
    }
    loaded_ = true;
    return true;
}

std::pair<std::string, std::string> TextTokenizer::get_min_pair(
    const std::vector<std::string> & word) const {
    
    std::pair<std::string, std::string> min_pair;
    int32_t min_rank = std::numeric_limits<int32_t>::max();
    
    for (size_t i = 0; i + 1 < word.size(); i++) {
        auto pair = std::make_pair(word[i], word[i + 1]);
        auto it = bpe_ranks_.find(pair);
        if (it != bpe_ranks_.end() && it->second < min_rank) {
            min_rank = it->second;
            min_pair = pair;
        }
    }
    
    return min_pair;
}

std::vector<std::string> TextTokenizer::bpe(const std::string & token) const {
    if (token.empty()) {
        return {};
    }
    
    // Split into unicode characters
    std::vector<std::string> word;
    size_t i = 0;
    while (i < token.size()) {
        size_t len = utf8_len(token[i]);
        word.push_back(token.substr(i, len));
        i += len;
    }
    
    if (word.size() == 1) {
        return word;
    }
    
    // Iteratively merge pairs
    while (true) {
        auto min_pair = get_min_pair(word);
        if (min_pair.first.empty()) {
            break;  // No more merges possible
        }
        
        // Merge all occurrences of the pair
        std::vector<std::string> new_word;
        size_t j = 0;
        while (j < word.size()) {
            if (j + 1 < word.size() && 
                word[j] == min_pair.first && 
                word[j + 1] == min_pair.second) {
                new_word.push_back(min_pair.first + min_pair.second);
                j += 2;
            } else {
                new_word.push_back(word[j]);
                j += 1;
            }
        }
        word = std::move(new_word);
        
        if (word.size() == 1) {
            break;
        }
    }
    
    return word;
}

std::vector<int32_t> TextTokenizer::encode(const std::string & text) const {
    if (!loaded_) {
        return {};
    }
    
    std::vector<int32_t> tokens;
    
    // 1. Pre-tokenize into "words" (regex-equivalent)
    std::vector<std::string> raw_words = split_regex_equivalent(text);
    
    // 2. Process each word
    for (const auto & raw_word : raw_words) {
        // Convert word to GPT-2 unicode representation
        std::string word = bytes_to_unicode(raw_word);
        
        // BPE encode the word
        auto bpe_tokens = bpe(word);
        for (const auto & tok : bpe_tokens) {
            auto it = vocab_.find(tok);
            if (it != vocab_.end()) {
                tokens.push_back(it->second);
            } else {
                // Unknown token - encode as bytes
                for (unsigned char c : tok) {
                    std::string byte_tok = BYTE_TO_UNICODE[c];
                    auto byte_it = vocab_.find(byte_tok);
                    if (byte_it != vocab_.end()) {
                        tokens.push_back(byte_it->second);
                    }
                }
            }
        }
    }
    
    return tokens;
}

std::string TextTokenizer::decode(const std::vector<int32_t> & tokens) const {
    std::string result;
    for (int32_t token : tokens) {
        result += decode_token(token);
    }
    return result;
}

std::string TextTokenizer::decode_token(int32_t token_id) const {
    if (token_id < 0 || token_id >= (int32_t)id_to_token_.size()) {
        return "";
    }

    const std::string & token = id_to_token_[token_id];

    // Convert from GPT-2 unicode back to bytes
    return unicode_to_bytes(token);
}

} // namespace qwen3_tts

