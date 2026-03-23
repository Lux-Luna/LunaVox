#include "logger.h"
#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <cstring>

namespace qwen3_tts {

Logger & Logger::instance() {
    static Logger logger;
    return logger;
}

bool Logger::init(const char * filename) {
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
    file_stream_.open(filename, std::ios::out | std::ios::trunc);
    return file_stream_.is_open();
}


void Logger::log_backend(int backend_level, const char * text) {
    // Backend logs (llama.cpp, ORT) are treated as DEBUG unless they are warnings/errors
    // For now we just dump them to file as DEBUG
    log(LogLevel::DEBUG_LOG, "[Backend] %s", text);
}

void Logger::log(LogLevel level, const char * fmt, ...) {
    char buf[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    // Strip trailing newlines from buf for consistent formatting
    size_t len = std::strlen(buf);
    while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) {
        buf[--len] = '\0';
    }

    const char * level_str = "";
    switch (level) {
        case LogLevel::DEBUG_LOG: level_str = "DEBUG"; break;
        case LogLevel::INFO_LOG:  level_str = "INFO";  break;
        case LogLevel::WARN_LOG:  level_str = "WARN";  break;
        case LogLevel::ERROR_LOG: level_str = "ERROR"; break;
        case LogLevel::USER_LOG:  level_str = "USER";  break;
    }

    if (file_stream_.is_open()) {
        file_stream_ << "[" << level_str << "] " << buf << std::endl;
        file_stream_.flush();
    }

    if ((int)level >= (int)level_ || level == LogLevel::USER_LOG) {
        const char * color = "";
        const char * reset = "\033[0m";
        if (level == LogLevel::WARN_LOG) color = "\033[33m"; // Yellow
        else if (level == LogLevel::ERROR_LOG) color = "\033[31m"; // Red

        if (level == LogLevel::USER_LOG) {
            std::cout << buf << std::endl;
        } else if (level == LogLevel::ERROR_LOG) {
            std::cerr << color << "[" << level_str << "] " << buf << reset << std::endl;
        } else {
            // INFO/DEBUG/WARN
            std::cout << color << "[" << level_str << "] " << buf << reset << std::endl;
        }
    }
}

} // namespace qwen3_tts
