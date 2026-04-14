#include "logger.h"
#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <cstring>

namespace lunavox {

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

void Logger::set_external_sink(ExternalLogSink sink) {
    std::lock_guard<std::mutex> lock(mutex_);
    external_sink_ = sink;
}


void Logger::log_backend(int backend_level, const char * text) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string t = text;
    
    // Pattern 1: Newer llama.cpp device info
    if (t.find("using device CUDA") != std::string::npos || 
        t.find("ggml_cuda_init: found") != std::string::npos ||
        t.find("CUDA devices found") != std::string::npos) {
        
        size_t start = t.find("CUDA");
        size_t name_start = t.find("(", start);
        size_t name_end = t.find(")", name_start);
        
        if (name_start != std::string::npos && name_end != std::string::npos) {
            std::string device_name = t.substr(name_start + 1, name_end - name_start - 1);
            llama_backend_info_ = "cuda: " + device_name;
        } else if (llama_backend_info_.find("cuda") == std::string::npos) {
            llama_backend_info_ = "cuda";
        }
    } else if (t.find("using device Vulkan") != std::string::npos || 
               t.find("ggml_vulkan: found") != std::string::npos ||
               t.find("Vulkan devices found") != std::string::npos) {
               
        size_t start = t.find("Vulkan");
        size_t name_start = t.find("(", start);
        size_t name_end = t.find(")", name_start);
        
        if (name_start != std::string::npos && name_end != std::string::npos) {
             std::string device_name = t.substr(name_start + 1, name_end - name_start - 1);
             llama_backend_info_ = "vulkan: " + device_name;
        } else if (llama_backend_info_.find("vulkan") == std::string::npos) {
            llama_backend_info_ = "vulkan";
        }
    } else if (t.find("using device Metal") != std::string::npos || 
               t.find("ggml_metal_init") != std::string::npos || 
               t.find("found Metal device") != std::string::npos) {
        llama_backend_info_ = "metal";
    } else if (t.find("using device ROCm") != std::string::npos || 
               t.find("ggml_rocm_init") != std::string::npos) {
        llama_backend_info_ = "rocm";
    } else if (t.find("using device SYCL") != std::string::npos || 
               t.find("ggml_sycl_init") != std::string::npos) {
        llama_backend_info_ = "sycl";
    }
    // Pattern 2: Layer offloading confirmation fallback
    else if (t.find("offloaded ") != std::string::npos && 
            (t.find("layers to GPU") != std::string::npos || 
             t.find("layers to CUDA") != std::string::npos ||
             t.find("layers to Vulkan") != std::string::npos ||
             t.find("layers to Metal") != std::string::npos ||
             t.find("layers to ROCm") != std::string::npos)) {
             
        // Ignore lines like "offloaded 0/28 layers to GPU" which mean no actual acceleration
        if (t.find("offloaded 0/") == std::string::npos) {
            // Only set if we don't have a specific backend name yet or it's still cpu
            if (llama_backend_info_ == "cpu" || llama_backend_info_ == "CPU") {
                if (t.find("CUDA") != std::string::npos) {
                    llama_backend_info_ = "cuda (accelerated)";
                } else if (t.find("Vulkan") != std::string::npos) {
                    llama_backend_info_ = "vulkan (accelerated)";
                } else if (t.find("Metal") != std::string::npos) {
                    llama_backend_info_ = "metal (accelerated)";
                } else if (t.find("ROCm") != std::string::npos) {
                    llama_backend_info_ = "rocm (accelerated)";
                } else {
                    llama_backend_info_ = "cpu";
                }
            }
        }
    }

    // Pattern 3: Legacy ggml logs
    else if (t.find("ggml_vulkan: 0 = ") != std::string::npos) {
        size_t start = t.find("ggml_vulkan: 0 = ") + 17;
        size_t end = t.find(" |", start);
        if (end != std::string::npos) {
            llama_backend_info_ = "vulkan: " + t.substr(start, end - start);
        } else {
            llama_backend_info_ = "vulkan";
        }
    } else if (t.find("ggml_cuda: 0 = ") != std::string::npos) {
        size_t start = t.find("ggml_cuda: 0 = ") + 15;
        size_t end = t.find(" |", start);
        if (end != std::string::npos) {
            llama_backend_info_ = "cuda: " + t.substr(start, end - start);
        } else {
            llama_backend_info_ = "cuda";
        }
    }

    // Log to file/console using internal parts to avoid double-lock
    char buf[2048];
    snprintf(buf, sizeof(buf), "[Backend] %s", text);

    if (file_stream_.is_open()) {
        file_stream_ << "[DEBUG] " << buf << std::endl;
        file_stream_.flush();
    }
    if (external_sink_) {
        external_sink_((int) LogLevel::DEBUG_LOG, buf);
    }
    if ((int)LogLevel::DEBUG_LOG >= (int)level_) {
        std::cout << "[DEBUG] " << buf << std::endl;
    }
}

void Logger::log(LogLevel level, const char * fmt, ...) {
    std::lock_guard<std::mutex> lock(mutex_);
    
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

    if (external_sink_) {
        external_sink_((int) level, buf);
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

} // namespace lunavox
