#pragma once

#include <string>
#include <fstream>
#include <mutex>

namespace lunavox {

enum class LogLevel {
    DEBUG_LOG,
    INFO_LOG,
    WARN_LOG,
    ERROR_LOG,
    USER_LOG  // For core info always shown to user
};

// External sink signature used by the C API log callback bridge. The int
// argument is the raw lunavox::LogLevel; consumers translate it themselves.
using ExternalLogSink = void (*)(int level, const char * message);

class Logger {
public:
    static Logger & instance();
    bool init(const char * filename);
    void set_level(LogLevel level) { level_ = level; }

    void log(LogLevel level, const char * fmt, ...);

    // Specifically for backend logs (like llama.cpp)
    void log_backend(int level, const char * text);

    // Install (or clear, by passing nullptr) an external sink that receives
    // every log line in addition to the file and console sinks. Used by the
    // C API to bridge C++ logs into Python/GUI consumers without stdout
    // scraping.
    void set_external_sink(ExternalLogSink sink);

    std::string get_llama_backend_info() const { return llama_backend_info_; }

private:
    Logger() = default;
    ~Logger() = default;
    Logger(const Logger &) = delete;
    Logger & operator=(const Logger &) = delete;

    LogLevel level_ = LogLevel::INFO_LOG;
    std::ofstream file_stream_;
    std::mutex mutex_;
    std::string llama_backend_info_ = "cpu";
    ExternalLogSink external_sink_ = nullptr;
};

// Convenience macros
#define LOG_DEBUG(...) lunavox::Logger::instance().log(lunavox::LogLevel::DEBUG_LOG, __VA_ARGS__)
#define LOG_INFO(...)  lunavox::Logger::instance().log(lunavox::LogLevel::INFO_LOG, __VA_ARGS__)
#define LOG_WARN(...)  lunavox::Logger::instance().log(lunavox::LogLevel::WARN_LOG, __VA_ARGS__)
#define LOG_ERROR(...) lunavox::Logger::instance().log(lunavox::LogLevel::ERROR_LOG, __VA_ARGS__)
#define LOG_USER(...)  lunavox::Logger::instance().log(lunavox::LogLevel::USER_LOG, __VA_ARGS__)

} // namespace lunavox
