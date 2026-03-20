#include "text_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>

namespace qwen3_tts {

namespace {

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

static std::unordered_map<std::string, uint8_t> build_unicode_to_byte() {
    std::unordered_map<std::string, uint8_t> result;
    for (int i = 0; i < 256; ++i) {
        result[BYTE_TO_UNICODE[i]] = (uint8_t) i;
    }
    return result;
}

static const std::unordered_map<std::string, uint8_t> UNICODE_TO_BYTE = build_unicode_to_byte();

void append_utf8(uint32_t cp, std::string & out) {
    if (cp <= 0x7F) {
        out.push_back((char) cp);
        return;
    }
    if (cp <= 0x7FF) {
        out.push_back((char) (0xC0 | (cp >> 6)));
        out.push_back((char) (0x80 | (cp & 0x3F)));
        return;
    }
    if (cp <= 0xFFFF) {
        out.push_back((char) (0xE0 | (cp >> 12)));
        out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char) (0x80 | (cp & 0x3F)));
        return;
    }
    out.push_back((char) (0xF0 | (cp >> 18)));
    out.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
    out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
    out.push_back((char) (0x80 | (cp & 0x3F)));
}

bool skip_ws(const std::string & s, size_t & i) {
    while (i < s.size() && std::isspace((unsigned char) s[i])) {
        ++i;
    }
    return i < s.size();
}

bool parse_json_string(const std::string & s, size_t & i, std::string & out) {
    out.clear();
    if (i >= s.size() || s[i] != '"') {
        return false;
    }
    ++i;
    while (i < s.size()) {
        char c = s[i++];
        if (c == '"') {
            return true;
        }
        if (c != '\\') {
            out.push_back(c);
            continue;
        }
        if (i >= s.size()) {
            return false;
        }
        char esc = s[i++];
        switch (esc) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                if (i + 4 > s.size()) {
                    return false;
                }
                auto hex = [&](char ch) -> int {
                    if (ch >= '0' && ch <= '9') return ch - '0';
                    if (ch >= 'a' && ch <= 'f') return 10 + (ch - 'a');
                    if (ch >= 'A' && ch <= 'F') return 10 + (ch - 'A');
                    return -1;
                };
                int h0 = hex(s[i]);
                int h1 = hex(s[i + 1]);
                int h2 = hex(s[i + 2]);
                int h3 = hex(s[i + 3]);
                if (h0 < 0 || h1 < 0 || h2 < 0 || h3 < 0) {
                    return false;
                }
                uint32_t cp = (uint32_t) ((h0 << 12) | (h1 << 8) | (h2 << 4) | h3);
                i += 4;

                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    if (i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                        int l0 = hex(s[i + 2]);
                        int l1 = hex(s[i + 3]);
                        int l2 = hex(s[i + 4]);
                        int l3 = hex(s[i + 5]);
                        if (l0 >= 0 && l1 >= 0 && l2 >= 0 && l3 >= 0) {
                            uint32_t lo = (uint32_t) ((l0 << 12) | (l1 << 8) | (l2 << 4) | l3);
                            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + (((cp - 0xD800) << 10) | (lo - 0xDC00));
                                i += 6;
                            }
                        }
                    }
                }
                append_utf8(cp, out);
                break;
            }
            default:
                return false;
        }
    }
    return false;
}

bool parse_int(const std::string & s, size_t & i, int32_t & out) {
    if (!skip_ws(s, i)) {
        return false;
    }
    size_t start = i;
    if (s[i] == '-') {
        ++i;
    }
    while (i < s.size() && std::isdigit((unsigned char) s[i])) {
        ++i;
    }
    if (i == start || (i == start + 1 && s[start] == '-')) {
        return false;
    }
    out = std::atoi(s.substr(start, i - start).c_str());
    return true;
}

bool parse_enclosed_block(const std::string & s, size_t & i, char open_ch, char close_ch, std::string & out) {
    if (!skip_ws(s, i) || i >= s.size() || s[i] != open_ch) {
        return false;
    }
    size_t start = i;
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (; i < s.size(); ++i) {
        char c = s[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == open_ch) {
            ++depth;
        } else if (c == close_ch) {
            --depth;
            if (depth == 0) {
                ++i;
                out = s.substr(start, i - start);
                return true;
            }
        }
    }
    return false;
}

bool find_key(const std::string & s, const std::string & key, size_t & pos_out) {
    std::string needle = "\"" + key + "\"";
    size_t pos = s.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos += needle.size();
    while (pos < s.size() && std::isspace((unsigned char) s[pos])) {
        ++pos;
    }
    if (pos >= s.size() || s[pos] != ':') {
        return false;
    }
    ++pos;
    pos_out = pos;
    return true;
}

bool parse_vocab_object(const std::string & obj, std::unordered_map<std::string, int32_t> & vocab) {
    size_t i = 0;
    if (!skip_ws(obj, i) || obj[i] != '{') {
        return false;
    }
    ++i;
    while (i < obj.size()) {
        skip_ws(obj, i);
        if (i >= obj.size()) break;
        if (obj[i] == '}') {
            return true;
        }

        std::string tok;
        if (!parse_json_string(obj, i, tok)) {
            return false;
        }
        skip_ws(obj, i);
        if (i >= obj.size() || obj[i] != ':') {
            return false;
        }
        ++i;

        int32_t id = -1;
        if (!parse_int(obj, i, id)) {
            return false;
        }
        vocab[tok] = id;

        skip_ws(obj, i);
        if (i < obj.size() && obj[i] == ',') {
            ++i;
        }
    }
    return false;
}

bool parse_merges_array(const std::string & arr, std::vector<std::string> & out) {
    out.clear();
    size_t i = 0;
    if (!skip_ws(arr, i) || arr[i] != '[') {
        return false;
    }
    ++i;
    while (i < arr.size()) {
        skip_ws(arr, i);
        if (i >= arr.size()) break;
        if (arr[i] == ']') {
            return true;
        }
        if (arr[i] == '"') {
            std::string s;
            if (!parse_json_string(arr, i, s)) {
                return false;
            }
            out.push_back(std::move(s));
        } else if (arr[i] == '[') {
            ++i;
            std::string a;
            std::string b;
            if (!skip_ws(arr, i) || !parse_json_string(arr, i, a)) {
                return false;
            }
            if (!skip_ws(arr, i) || i >= arr.size() || arr[i] != ',') {
                return false;
            }
            ++i;
            if (!skip_ws(arr, i) || !parse_json_string(arr, i, b)) {
                return false;
            }
            if (!skip_ws(arr, i) || i >= arr.size() || arr[i] != ']') {
                return false;
            }
            ++i;
            out.push_back(a + " " + b);
        } else {
            return false;
        }
        skip_ws(arr, i);
        if (i < arr.size() && arr[i] == ',') {
            ++i;
        }
    }
    return false;
}

void parse_added_tokens_specials(const std::string & arr, tokenizer_config & cfg) {
    size_t i = 0;
    if (!skip_ws(arr, i) || arr[i] != '[') {
        return;
    }
    ++i;
    while (i < arr.size()) {
        skip_ws(arr, i);
        if (i >= arr.size() || arr[i] == ']') {
            break;
        }
        std::string obj;
        if (!parse_enclosed_block(arr, i, '{', '}', obj)) {
            break;
        }

        size_t p = 0;
        std::string content;
        int32_t id = -1;
        if (find_key(obj, "content", p)) {
            skip_ws(obj, p);
            parse_json_string(obj, p, content);
        }
        if (find_key(obj, "id", p)) {
            parse_int(obj, p, id);
        }
        if (id >= 0) {
            if (content == "<|im_start|>") cfg.bos_token_id = id;
            if (content == "<|im_end|>") cfg.eos_token_id = id;
            if (content == "<|endoftext|>") cfg.pad_token_id = id;
        }
        skip_ws(arr, i);
        if (i < arr.size() && arr[i] == ',') {
            ++i;
        }
    }
}

} // namespace

TextTokenizer::TextTokenizer() = default;
TextTokenizer::~TextTokenizer() = default;

size_t TextTokenizer::utf8_len(char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
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
            result += (char) it->second;
        } else {
            result += ch;
        }
        i += len;
    }
    return result;
}

bool TextTokenizer::is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0x2F800 && cp <= 0x2FA1F);
}

bool TextTokenizer::load_from_json(const std::string & tokenizer_json_path) {
    loaded_ = false;
    error_msg_.clear();
    vocab_.clear();
    id_to_token_.clear();
    bpe_ranks_.clear();

    std::ifstream fin(tokenizer_json_path, std::ios::binary);
    if (!fin) {
        error_msg_ = "Failed to open tokenizer.json: " + tokenizer_json_path;
        return false;
    }
    std::string json((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    if (json.empty()) {
        error_msg_ = "tokenizer.json is empty";
        return false;
    }

    size_t model_pos = 0;
    if (!find_key(json, "model", model_pos)) {
        error_msg_ = "tokenizer.json missing 'model' field";
        return false;
    }
    std::string model_obj;
    if (!parse_enclosed_block(json, model_pos, '{', '}', model_obj)) {
        error_msg_ = "Failed to parse tokenizer model object";
        return false;
    }

    size_t vocab_pos = 0;
    if (!find_key(model_obj, "vocab", vocab_pos)) {
        error_msg_ = "tokenizer model missing 'vocab'";
        return false;
    }
    std::string vocab_obj;
    if (!parse_enclosed_block(model_obj, vocab_pos, '{', '}', vocab_obj)) {
        error_msg_ = "Failed to parse tokenizer vocab object";
        return false;
    }
    if (!parse_vocab_object(vocab_obj, vocab_)) {
        error_msg_ = "Failed to parse tokenizer vocab key/value entries";
        return false;
    }

    size_t merges_pos = 0;
    if (!find_key(model_obj, "merges", merges_pos)) {
        error_msg_ = "tokenizer model missing 'merges'";
        return false;
    }
    std::string merges_arr;
    if (!parse_enclosed_block(model_obj, merges_pos, '[', ']', merges_arr)) {
        error_msg_ = "Failed to parse tokenizer merges array";
        return false;
    }
    std::vector<std::string> merges;
    if (!parse_merges_array(merges_arr, merges)) {
        error_msg_ = "Failed to parse tokenizer merges entries";
        return false;
    }

    for (size_t rank = 0; rank < merges.size(); ++rank) {
        const std::string & m = merges[rank];
        size_t sp = m.find(' ');
        if (sp == std::string::npos) {
            continue;
        }
        std::string a = m.substr(0, sp);
        std::string b = m.substr(sp + 1);
        bpe_ranks_[{a, b}] = (int32_t) rank;
    }

    size_t added_pos = 0;
    if (find_key(json, "added_tokens", added_pos)) {
        std::string added_arr;
        if (parse_enclosed_block(json, added_pos, '[', ']', added_arr)) {
            parse_added_tokens_specials(added_arr, config_);
        }
    }

    int32_t max_id = -1;
    for (const auto & kv : vocab_) {
        max_id = std::max(max_id, kv.second);
    }
    if (max_id < 0) {
        error_msg_ = "Tokenizer vocab is empty";
        return false;
    }
    id_to_token_.assign((size_t) max_id + 1, "");
    for (const auto & kv : vocab_) {
        if (kv.second >= 0 && kv.second < (int32_t) id_to_token_.size()) {
            id_to_token_[(size_t) kv.second] = kv.first;
        }
    }
    config_.vocab_size = (int32_t) id_to_token_.size();
    loaded_ = true;
    return true;
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
        unsigned char c = (unsigned char) text[i];

        if (c < 0x80) {
            cp = c;
            len = 1;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
            cp = ((c & 0x1F) << 6) | (text[i + 1] & 0x3F);
            len = 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
            cp = ((c & 0x0F) << 12) | ((text[i + 1] & 0x3F) << 6) | (text[i + 2] & 0x3F);
            len = 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < text.size()) {
            cp = ((c & 0x07) << 18) | ((text[i + 1] & 0x3F) << 12) | ((text[i + 2] & 0x3F) << 6) | (text[i + 3] & 0x3F);
            len = 4;
        } else {
            cp = c;
            len = 1;
        }

        bool is_alnum_cjk = (cp >= 'a' && cp <= 'z') ||
                             (cp >= 'A' && cp <= 'Z') ||
                             (cp >= '0' && cp <= '9') ||
                             is_cjk(cp);

        if (std::isspace(c)) {
            flush();
            std::string spaces;
            while (i < text.size() && std::isspace((unsigned char) text[i])) {
                spaces += text[i];
                ++i;
            }
            words.push_back(spaces);
            continue;
        }
        if (is_alnum_cjk) {
            current += text.substr(i, len);
        } else {
            flush();
            words.push_back(text.substr(i, len));
        }
        i += len;
    }
    flush();
    return words;
}

std::pair<std::string, std::string> TextTokenizer::get_min_pair(const std::vector<std::string> & word) const {
    std::pair<std::string, std::string> min_pair;
    int32_t min_rank = std::numeric_limits<int32_t>::max();
    for (size_t i = 0; i + 1 < word.size(); ++i) {
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
    std::vector<std::string> word;
    size_t i = 0;
    while (i < token.size()) {
        size_t len = utf8_len(token[i]);
        word.push_back(token.substr(i, len));
        i += len;
    }
    if (word.size() <= 1) {
        return word;
    }

    while (true) {
        auto min_pair = get_min_pair(word);
        if (min_pair.first.empty()) {
            break;
        }
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
                ++j;
            }
        }
        word = std::move(new_word);
        if (word.size() <= 1) {
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
    std::vector<std::string> raw_words = split_regex_equivalent(text);

    for (const auto & raw_word : raw_words) {
        std::string word = bytes_to_unicode(raw_word);
        auto pieces = bpe(word);
        for (const auto & p : pieces) {
            auto it = vocab_.find(p);
            if (it != vocab_.end()) {
                tokens.push_back(it->second);
                continue;
            }
            for (unsigned char c : p) {
                std::string byte_tok = BYTE_TO_UNICODE[c];
                auto bit = vocab_.find(byte_tok);
                if (bit != vocab_.end()) {
                    tokens.push_back(bit->second);
                }
            }
        }
    }
    return tokens;
}

std::string TextTokenizer::decode(const std::vector<int32_t> & tokens) const {
    std::string out;
    for (int32_t id : tokens) {
        out += decode_token(id);
    }
    return out;
}

std::string TextTokenizer::decode_token(int32_t token_id) const {
    if (token_id < 0 || token_id >= (int32_t) id_to_token_.size()) {
        return "";
    }
    return unicode_to_bytes(id_to_token_[(size_t) token_id]);
}

} // namespace qwen3_tts
