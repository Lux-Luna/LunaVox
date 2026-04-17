#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace lunavox {

/**
 * @brief NVML wrapper with per-PID VRAM attribution.
 *
 * Dynamically loads libnvidia-ml so non-NVIDIA systems fail softly. The
 * per-call entry point is `sample_pid_vram(pid)` which sums only the given
 * process's allocations across all visible devices — fixes the old
 * `get_used_vram()` behaviour of returning whole-device usage (contaminated
 * by every other process on the GPU).
 */
class NVMLMonitor {
public:
    struct VramSample {
        uint64_t bytes = 0;   // sum of usedGpuMemory for `pid` across visible devices
        bool attributed = false; // true only when every device returned a meaningful reading
    };

    static NVMLMonitor & instance();
    ~NVMLMonitor();

    bool init();
    void shutdown();

    bool is_available() const { return initialized_; }

    // Sum of `pid`'s VRAM usage across every visible NVIDIA device. Returns
    // {0, false} when NVML is unavailable or when any device reported
    // NVML_VALUE_NOT_AVAILABLE for memory on a matching process entry —
    // callers treat `attributed=false` as "not measured" instead of
    // manufacturing a fake zero.
    VramSample sample_pid_vram(uint32_t pid);

    // Primary device name (index 0). Used for diagnostic log lines only.
    std::string get_device_name();

private:
    NVMLMonitor() = default;

    bool initialized_ = false;
    void * library_handle_ = nullptr;
    std::vector<void *> devices_;
    size_t proc_struct_size_ = 0;  // picked at init: 16 (v1 layout) or 24 (v2 layout)

    // NVML function pointers
    typedef int (*nvml_init_t)();
    typedef int (*nvml_shutdown_t)();
    typedef int (*nvml_device_get_count_t)(unsigned int *);
    typedef int (*nvml_device_get_handle_by_index_t)(unsigned int, void **);
    typedef int (*nvml_device_get_name_t)(void *, char *, unsigned int);
    // nvmlDeviceGet{Compute,Graphics}RunningProcesses* all share this shape:
    //   (device, in/out count, out buffer of proc-info structs)
    typedef int (*nvml_device_get_procs_t)(void *, unsigned int *, void *);

    nvml_init_t nvmlInit = nullptr;
    nvml_shutdown_t nvmlShutdown = nullptr;
    nvml_device_get_count_t nvmlDeviceGetCount = nullptr;
    nvml_device_get_handle_by_index_t nvmlDeviceGetHandleByIndex = nullptr;
    nvml_device_get_name_t nvmlDeviceGetName = nullptr;

    nvml_device_get_procs_t compute_procs_ = nullptr;
    nvml_device_get_procs_t graphics_procs_ = nullptr;
};

} // namespace lunavox
