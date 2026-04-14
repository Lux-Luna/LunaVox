#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace lunavox {

inline std::string format_bytes(uint64_t bytes) {
    static const char * units[] = {"B", "KB", "MB", "GB", "TB"};
    double val = (double) bytes;
    int unit = 0;
    while (val >= 1024.0 && unit < 4) {
        val /= 1024.0;
        ++unit;
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f %s", val, units[unit]);
    return std::string(buf);
}

} // namespace lunavox
