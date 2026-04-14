#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace lunavox {

struct tokenizer_config {
    int32_t vocab_size = 151936;
    int32_t pad_token_id = 151643;
    int32_t eos_token_id = 151645; // <|im_end|>
    int32_t bos_token_id = 151644; // <|im_start|>
};

class TextTokenizer {
public:
    TextTokenizer();
    ~TextTokenizer();

    bool load_from_json(const std::string & tokenizer_json_path);

    std::vector<int32_t> encode(const std::string & text) const;
    std::string decode(const std::vector<int32_t> & tokens) const;
    std::string decode_token(int32_t token_id) const;

    const tokenizer_config & get_config() const { return config_; }
    const std::string & get_error() const { return error_msg_; }
    bool is_loaded() const { return loaded_; }

    int32_t bos_token_id() const { return config_.bos_token_id; }
    int32_t eos_token_id() const { return config_.eos_token_id; }
    int32_t pad_token_id() const { return config_.pad_token_id; }

private:
    tokenizer_config config_;
    std::string error_msg_;
    bool loaded_ = false;

    std::unordered_map<std::string, int32_t> vocab_;
    std::vector<std::string> id_to_token_;
    std::map<std::pair<std::string, std::string>, int32_t> bpe_ranks_;

    static std::string bytes_to_unicode(const std::string & text);
    static std::string unicode_to_bytes(const std::string & text);
    static size_t utf8_len(char c);

    std::vector<std::string> bpe(const std::string & token) const;
    std::pair<std::string, std::string> get_min_pair(const std::vector<std::string> & word) const;
    std::vector<std::string> split_regex_equivalent(const std::string & text) const;
    static bool is_cjk(uint32_t cp);
};

} // namespace lunavox

