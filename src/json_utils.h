#ifndef QWEN3_TTS_JSON_UTILS_H
#define QWEN3_TTS_JSON_UTILS_H

#include <string>
#include <vector>
#include <cctype>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <algorithm>

namespace qwen3_tts {

/**
 * Helper: Try to find metadata.json near the binary or in lib/
 */
static inline std::string find_metadata_json() {
    const char* p_list[] = {"metadata.json", "lib/metadata.json", "../lib/metadata.json"};
    for (const char* p : p_list) {
        std::ifstream f(p);
        if (f.good()) return p;
    }
    return "";
}

/**
 * Standard Base64 decoder.
 */
static inline std::vector<uint8_t> base64_decode(const std::string &in) {
    std::vector<uint8_t> out;
    std::vector<int> T(256, -1);
    const char* b64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; i++) T[(unsigned char)b64_chars[i]] = i;

    int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (T[c] == -1) continue;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(uint8_t((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

/**
 * A very lightweight JSON string extractor.
 * Returns true if the key was found and the string extracted into 'out'.
 */
static inline bool json_extract_string(const std::string & json, const std::string & key, std::string & out) {
    std::string qkey = "\"" + key + "\"";
    size_t pos = json.find(qkey);
    if (pos == std::string::npos) return false;
    pos = json.find(":", pos + qkey.length());
    if (pos == std::string::npos) return false;
    pos = json.find("\"", pos);
    if (pos == std::string::npos) return false;
    size_t end = json.find("\"", pos + 1);
    while (end != std::string::npos && json[end - 1] == '\\') {
        end = json.find("\"", end + 1);
    }
    if (end == std::string::npos) return false;
    out = json.substr(pos + 1, end - pos - 1);
    return true;
}

/**
 * Extract a flat list of integers from a JSON array [1, 2, 3].
 */
static inline bool json_extract_flat_int_array(const std::string & json, const std::string & key, std::vector<int32_t> & out) {
    out.clear();
    std::string qkey = "\"" + key + "\"";
    size_t pos = json.find(qkey);
    if (pos == std::string::npos) return false;
    pos = json.find(":", pos + qkey.length());
    if (pos == std::string::npos) return false;
    pos = json.find("[", pos);
    if (pos == std::string::npos) return false;
    
    size_t i = pos + 1;
    std::string current_num;
    while (i < json.size()) {
        char c = json[i];
        if (c == ']') {
            if (!current_num.empty()) {
                out.push_back(std::stoi(current_num));
                current_num.clear();
            }
            break;
        } else if (std::isdigit((unsigned char)c) || c == '-') {
            current_num += c;
        } else if (c == ',' || std::isspace((unsigned char)c)) {
            if (!current_num.empty()) {
                out.push_back(std::stoi(current_num));
                current_num.clear();
            }
        }
        i++;
    }
    return true;
}

} // namespace qwen3_tts

#endif // QWEN3_TTS_JSON_UTILS_H
