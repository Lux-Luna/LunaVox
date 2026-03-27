#ifndef QWEN3_TTS_JSON_UTILS_H
#define QWEN3_TTS_JSON_UTILS_H

#include <string>
#include <vector>
#include <cctype>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cerrno>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace qwen3_tts {

/**
 * Helper: Get the directory of the currently running executable.
 */
static inline std::string get_executable_dir() {
#ifdef _WIN32
    char path[MAX_PATH];
    DWORD size = GetModuleFileNameA(NULL, path, MAX_PATH);
    if (size == 0) return ".";
    std::string s(path);
    size_t last_slash = s.find_last_of("\\/");
    if (last_slash != std::string::npos) return s.substr(0, last_slash);
    return ".";
#else
    char path[1024];
    ssize_t count = readlink("/proc/self/exe", path, sizeof(path));
    if (count != -1) {
        std::string s(path, count);
        size_t last_slash = s.find_last_of("/");
        if (last_slash != std::string::npos) return s.substr(0, last_slash);
    }
    return ".";
#endif
}

/**
 * Helper: Try to find metadata.json near the binary or via environment
 */
static inline std::string find_metadata_json() {
    std::string exe_dir = get_executable_dir();
    const char* lib_dir_env = std::getenv("QWEN3_TTS_LIB_DIR");

    std::vector<std::string> p_list = {
        "metadata.json",                    // CWD
        exe_dir + "/metadata.json",         // Root or Build (portable)
        exe_dir + "/../metadata.json"       // Alternative layout
    };
    
    if (lib_dir_env && lib_dir_env[0]) {
        p_list.push_back(std::string(lib_dir_env) + "/metadata.json");
    }

    for (const auto & p : p_list) {
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

static inline bool json_extract_int(const std::string & json, const std::string & key, int32_t & out) {
    std::string qkey = "\"" + key + "\"";
    size_t pos = json.find(qkey);
    if (pos == std::string::npos) return false;
    pos = json.find(":", pos + qkey.length());
    if (pos == std::string::npos) return false;
    ++pos;
    while (pos < json.size() && std::isspace((unsigned char) json[pos])) ++pos;
    if (pos >= json.size()) return false;

    size_t end = pos;
    if (json[end] == '-') ++end;
    while (end < json.size() && std::isdigit((unsigned char) json[end])) ++end;
    if (end == pos || (end == pos + 1 && json[pos] == '-')) return false;
    try {
        out = (int32_t) std::stoll(json.substr(pos, end - pos));
    } catch (...) {
        return false;
    }
    return true;
}

static inline bool json_extract_bool(const std::string & json, const std::string & key, bool & out) {
    std::string qkey = "\"" + key + "\"";
    size_t pos = json.find(qkey);
    if (pos == std::string::npos) return false;
    pos = json.find(":", pos + qkey.length());
    if (pos == std::string::npos) return false;
    ++pos;
    while (pos < json.size() && std::isspace((unsigned char) json[pos])) ++pos;
    if (pos >= json.size()) return false;

    if (json.compare(pos, 4, "true") == 0) {
        out = true;
        return true;
    }
    if (json.compare(pos, 5, "false") == 0) {
        out = false;
        return true;
    }
    return false;
}

static inline bool json_extract_float(const std::string & json, const std::string & key, float & out) {
    std::string qkey = "\"" + key + "\"";
    size_t pos = json.find(qkey);
    if (pos == std::string::npos) return false;
    pos = json.find(":", pos + qkey.length());
    if (pos == std::string::npos) return false;
    ++pos;
    while (pos < json.size() && std::isspace((unsigned char) json[pos])) ++pos;
    if (pos >= json.size()) return false;

    const char * begin = json.c_str() + pos;
    char * endptr = nullptr;
    errno = 0;
    const double v = std::strtod(begin, &endptr);
    if (begin == endptr || errno == ERANGE) return false;
    out = (float) v;
    return true;
}

static inline bool json_extract_string_array(const std::string & json, const std::string & key, std::vector<std::string> & out) {
    out.clear();
    std::string qkey = "\"" + key + "\"";
    size_t pos = json.find(qkey);
    if (pos == std::string::npos) return false;
    pos = json.find(":", pos + qkey.length());
    if (pos == std::string::npos) return false;
    pos = json.find("[", pos);
    if (pos == std::string::npos) return false;

    int depth = 1;
    size_t i = pos + 1;
    while (i < json.size() && depth > 0) {
        char c = json[i];
        if (c == '[') {
            ++depth;
            ++i;
            continue;
        }
        if (c == ']') {
            --depth;
            ++i;
            continue;
        }
        if (depth == 1 && c == '"') {
            size_t j = i + 1;
            std::string token;
            while (j < json.size()) {
                char cc = json[j];
                if (cc == '\\' && j + 1 < json.size()) {
                    token.push_back(json[j + 1]);
                    j += 2;
                    continue;
                }
                if (cc == '"') {
                    break;
                }
                token.push_back(cc);
                ++j;
            }
            if (j >= json.size() || json[j] != '"') {
                return false;
            }
            out.push_back(token);
            i = j + 1;
            continue;
        }
        ++i;
    }
    return depth == 0;
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
    int depth = 1;
    std::string current_num;
    while (i < json.size() && depth > 0) {
        char c = json[i];
        if (c == '[') {
            ++depth;
        } else if (c == ']') {
            --depth;
            if (!current_num.empty()) {
                out.push_back(std::stoi(current_num));
                current_num.clear();
            }
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
    return depth == 0;
}

} // namespace qwen3_tts

#endif // QWEN3_TTS_JSON_UTILS_H
