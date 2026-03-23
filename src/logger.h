#pragma once

#include <string>
#include <fstream>
#include <mutex>

namespace qwen3_tts {

enum class LogLevel {
    DEBUG_LOG,
    INFO_LOG,
    WARN_LOG,
    ERROR_LOG,
    USER_LOG  // For core info always shown to user
};

class Logger {
public:
    static Logger & instance();
    bool init(const char * filename);
    void set_level(LogLevel level) { level_ = level; }

    void log(LogLevel level, const char * fmt, ...);

    // Specifically for backend logs (like llama.cpp)
    void log_backend(int level, const char * text);
    
    std::string get_llama_backend_info() const { return llama_backend_info_; }

private:
    Logger() = default;
    ~Logger() = default;
    Logger(const Logger &) = delete;
    Logger & operator=(const Logger &) = delete;

    LogLevel level_ = LogLevel::INFO_LOG;
    std::ofstream file_stream_;
    std::mutex mutex_;
    std::string llama_backend_info_ = "CPU";
};

// Convenience macros
#define LOG_DEBUG(...) qwen3_tts::Logger::instance().log(qwen3_tts::LogLevel::DEBUG_LOG, __VA_ARGS__)
#define LOG_INFO(...)  qwen3_tts::Logger::instance().log(qwen3_tts::LogLevel::INFO_LOG, __VA_ARGS__)
#define LOG_WARN(...)  qwen3_tts::Logger::instance().log(qwen3_tts::LogLevel::WARN_LOG, __VA_ARGS__)
#define LOG_ERROR(...) qwen3_tts::Logger::instance().log(qwen3_tts::LogLevel::ERROR_LOG, __VA_ARGS__)
#define LOG_USER(...)  qwen3_tts::Logger::instance().log(qwen3_tts::LogLevel::USER_LOG, __VA_ARGS__)

} // namespace qwen3_tts
