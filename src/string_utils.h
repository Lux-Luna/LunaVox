#pragma once

#include <algorithm>
#include <cctype>
#include <string>

namespace lunavox {

inline std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return s;
}

inline bool starts_with(const std::string & s, const std::string & prefix) {
    return s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
}

inline bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() &&
           std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

} // namespace lunavox
