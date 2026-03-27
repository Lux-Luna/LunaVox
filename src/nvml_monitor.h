#pragma once
#include <cstdint>
#include <string>

namespace qwen3_tts {

/**
 * @brief Simple NVML wrapper to monitor VRAM without compile-time dependency on CUDA/NVML headers.
 * Uses dynamic loading (LoadLibrary/dlopen) for safety on non-NVIDIA systems.
 */
class NVMLMonitor {
public:
    static NVMLMonitor& instance();
    ~NVMLMonitor();

    bool init();
    void shutdown();

    bool is_available() const { return initialized_; }
    
    // Returns used VRAM in bytes for the first device found.
    uint64_t get_used_vram();
    
    // Returns device name
    std::string get_device_name();

private:
    NVMLMonitor() = default;
    bool initialized_ = false;
    void* library_handle_ = nullptr;
    void* device_handle_ = nullptr;

    // NVML function pointers
    typedef int (*nvml_init_t)();
    typedef int (*nvml_shutdown_t)();
    typedef int (*nvml_device_get_count_t)(unsigned int*);
    typedef int (*nvml_device_get_handle_by_index_t)(unsigned int, void**);
    typedef int (*nvml_device_get_memory_info_t)(void*, struct nvml_memory_t*);
    typedef int (*nvml_device_get_name_t)(void*, char*, unsigned int);

    nvml_init_t nvmlInit = nullptr;
    nvml_shutdown_t nvmlShutdown = nullptr;
    nvml_device_get_count_t nvmlDeviceGetCount = nullptr;
    nvml_device_get_handle_by_index_t nvmlDeviceGetHandleByIndex = nullptr;
    nvml_device_get_memory_info_t nvmlDeviceGetMemoryInfo = nullptr;
    nvml_device_get_name_t nvmlDeviceGetName = nullptr;
};

// Internal NVML struct
struct nvml_memory_t {
    uint64_t total;
    uint64_t free;
    uint64_t used;
};

} // namespace qwen3_tts
