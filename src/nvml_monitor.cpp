#include "nvml_monitor.h"

#ifdef _WIN32
#include <windows.h>
#define LIB_HANDLE HMODULE
#define LOAD_LIB(name) LoadLibraryA(name)
#define GET_PROC(h, name) GetProcAddress(h, name)
#define FREE_LIB(h) FreeLibrary(h)
#else
#include <dlfcn.h>
#define LIB_HANDLE void*
#define LOAD_LIB(name) dlopen(name, RTLD_LAZY)
#define GET_PROC(h, name) dlsym(h, name)
#define FREE_LIB(h) dlclose(h)
#endif

namespace qwen3_tts {

NVMLMonitor & NVMLMonitor::instance() {
    static NVMLMonitor mon;
    return mon;
}

NVMLMonitor::~NVMLMonitor() {
    shutdown();
}

bool NVMLMonitor::init() {
    if (initialized_) return true;

#ifdef _WIN32
    library_handle_ = LOAD_LIB("nvml.dll");
#else
    library_handle_ = LOAD_LIB("libnvidia-ml.so.1");
#endif

    if (!library_handle_) return false;

    // Load function pointers
    nvmlInit = (nvml_init_t)GET_PROC((LIB_HANDLE)library_handle_, "nvmlInit");
    nvmlShutdown = (nvml_shutdown_t)GET_PROC((LIB_HANDLE)library_handle_, "nvmlShutdown");
    nvmlDeviceGetCount = (nvml_device_get_count_t)GET_PROC((LIB_HANDLE)library_handle_, "nvmlDeviceGetCount");
    nvmlDeviceGetHandleByIndex = (nvml_device_get_handle_by_index_t)GET_PROC((LIB_HANDLE)library_handle_, "nvmlDeviceGetHandleByIndex");
    nvmlDeviceGetMemoryInfo = (nvml_device_get_memory_info_t)GET_PROC((LIB_HANDLE)library_handle_, "nvmlDeviceGetMemoryInfo");
    nvmlDeviceGetName = (nvml_device_get_name_t)GET_PROC((LIB_HANDLE)library_handle_, "nvmlDeviceGetName");

    if (!nvmlInit || !nvmlShutdown || !nvmlDeviceGetCount || !nvmlDeviceGetHandleByIndex || !nvmlDeviceGetMemoryInfo) {
        FREE_LIB((LIB_HANDLE)library_handle_);
        library_handle_ = nullptr;
        return false;
    }

    // Initialize NVML
    if (nvmlInit() != 0) {
        FREE_LIB((LIB_HANDLE)library_handle_);
        library_handle_ = nullptr;
        return false;
    }

    // Get handle for device 0
    unsigned int deviceCount = 0;
    if (nvmlDeviceGetCount(&deviceCount) != 0 || deviceCount == 0) {
        nvmlShutdown();
        FREE_LIB((LIB_HANDLE)library_handle_);
        library_handle_ = nullptr;
        return false;
    }

    if (nvmlDeviceGetHandleByIndex(0, &device_handle_) != 0) {
        nvmlShutdown();
        FREE_LIB((LIB_HANDLE)library_handle_);
        library_handle_ = nullptr;
        return false;
    }

    initialized_ = true;
    return true;
}

void NVMLMonitor::shutdown() {
    if (initialized_ && nvmlShutdown) {
        nvmlShutdown();
    }
    if (library_handle_) {
        FREE_LIB((LIB_HANDLE)library_handle_);
    }
    initialized_ = false;
    library_handle_ = nullptr;
}

uint64_t NVMLMonitor::get_used_vram() {
    if (!initialized_ || !device_handle_) return 0;
    
    struct nvml_memory_t mem{};
    if (nvmlDeviceGetMemoryInfo(device_handle_, &mem) == 0) {
        return mem.used;
    }
    return 0;
}

std::string NVMLMonitor::get_device_name() {
    if (!initialized_ || !device_handle_) return "NVIDIA GPU";
    
    char buf[128];
    if (nvmlDeviceGetName && nvmlDeviceGetName(device_handle_, buf, sizeof(buf)) == 0) {
        return std::string(buf);
    }
    return "NVIDIA GPU";
}

} // namespace qwen3_tts
