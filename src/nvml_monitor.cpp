#include "nvml_monitor.h"
#include "platform_utils.h"

namespace lunavox {

// NVIDIA ships the management library under a platform-specific name
// (nvml.dll on Windows, libnvidia-ml.so.1 on Linux). Probe a short list and
// let the loader tell us which one exists — avoids hard-coding a branch
// outside platform_utils.
namespace {
static const char * kNvmlCandidates[] = {
    "nvml.dll",            // Windows
    "libnvidia-ml.so.1",   // Linux
};
}

NVMLMonitor & NVMLMonitor::instance() {
    static NVMLMonitor mon;
    return mon;
}

NVMLMonitor::~NVMLMonitor() {
    shutdown();
}

bool NVMLMonitor::init() {
    if (initialized_) return true;

    for (const char * name : kNvmlCandidates) {
        library_handle_ = platform::dynlib_open(name);
        if (library_handle_) break;
    }
    if (!library_handle_) return false;

    nvmlInit = (nvml_init_t) platform::dynlib_symbol(library_handle_, "nvmlInit");
    nvmlShutdown = (nvml_shutdown_t) platform::dynlib_symbol(library_handle_, "nvmlShutdown");
    nvmlDeviceGetCount = (nvml_device_get_count_t) platform::dynlib_symbol(library_handle_, "nvmlDeviceGetCount");
    nvmlDeviceGetHandleByIndex = (nvml_device_get_handle_by_index_t) platform::dynlib_symbol(library_handle_, "nvmlDeviceGetHandleByIndex");
    nvmlDeviceGetMemoryInfo = (nvml_device_get_memory_info_t) platform::dynlib_symbol(library_handle_, "nvmlDeviceGetMemoryInfo");
    nvmlDeviceGetName = (nvml_device_get_name_t) platform::dynlib_symbol(library_handle_, "nvmlDeviceGetName");

    if (!nvmlInit || !nvmlShutdown || !nvmlDeviceGetCount || !nvmlDeviceGetHandleByIndex || !nvmlDeviceGetMemoryInfo) {
        platform::dynlib_close(library_handle_);
        library_handle_ = nullptr;
        return false;
    }

    if (nvmlInit() != 0) {
        platform::dynlib_close(library_handle_);
        library_handle_ = nullptr;
        return false;
    }

    unsigned int deviceCount = 0;
    if (nvmlDeviceGetCount(&deviceCount) != 0 || deviceCount == 0) {
        nvmlShutdown();
        platform::dynlib_close(library_handle_);
        library_handle_ = nullptr;
        return false;
    }

    if (nvmlDeviceGetHandleByIndex(0, &device_handle_) != 0) {
        nvmlShutdown();
        platform::dynlib_close(library_handle_);
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
        platform::dynlib_close(library_handle_);
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

} // namespace lunavox
