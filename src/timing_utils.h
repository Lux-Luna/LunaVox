#pragma once

#include <chrono>
#include <cstdint>

namespace lunavox {

inline int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

} // namespace lunavox
