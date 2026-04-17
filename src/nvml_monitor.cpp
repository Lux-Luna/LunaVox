#include "nvml_monitor.h"
#include "platform_utils.h"

#include <cstdint>
#include <cstring>
#include <vector>

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

// NVML error constants we care about — kept local so we don't depend on
// shipping the NVML headers.
constexpr int NVML_SUCCESS_ = 0;
constexpr int NVML_ERROR_INSUFFICIENT_SIZE_ = 7;

// Sentinel returned in `usedGpuMemory` when the driver cannot attribute a
// process entry to a byte count (e.g. MIG partitions without per-process
// telemetry). Matches NVML_VALUE_NOT_AVAILABLE.
constexpr uint64_t NVML_NOT_AVAILABLE_ = (uint64_t) -1;

// Both nvmlProcessInfo_t layouts start with `unsigned int pid;` followed by
// 4 bytes of padding (so `unsigned long long usedGpuMemory` lands at offset
// 8). v2 adds two tail `unsigned int` fields. Reading pid/usedGpuMemory via
// fixed offsets works regardless of which layout the driver fills.
constexpr size_t kProcStructV1Size_ = 16; // {pid, pad, usedGpuMemory}
constexpr size_t kProcStructV2Size_ = 24; // v1 + {gpuInstanceId, computeInstanceId}
constexpr size_t kProcPidOffset_ = 0;
constexpr size_t kProcUsedMemOffset_ = 8;

// Same shape as NVMLMonitor::nvml_device_get_procs_t — redeclared here so
// the file-scope helpers below don't need access to the private typedef.
using nvml_device_get_procs_fn = int (*)(void *, unsigned int *, void *);
} // namespace

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
    nvmlDeviceGetName = (nvml_device_get_name_t) platform::dynlib_symbol(library_handle_, "nvmlDeviceGetName");

    if (!nvmlInit || !nvmlShutdown || !nvmlDeviceGetCount || !nvmlDeviceGetHandleByIndex) {
        platform::dynlib_close(library_handle_);
        library_handle_ = nullptr;
        return false;
    }

    // Prefer the modern v3 symbol (R460+, 2021+) which writes the 24-byte
    // nvmlProcessInfo_v2_t layout. Fall back to _v2 / unsuffixed which write
    // the 16-byte layout. Either way the pid/usedGpuMemory offsets match,
    // so reading works uniformly once proc_struct_size_ is known.
    compute_procs_ = (nvml_device_get_procs_t) platform::dynlib_symbol(library_handle_, "nvmlDeviceGetComputeRunningProcesses_v3");
    graphics_procs_ = (nvml_device_get_procs_t) platform::dynlib_symbol(library_handle_, "nvmlDeviceGetGraphicsRunningProcesses_v3");
    if (compute_procs_ || graphics_procs_) {
        proc_struct_size_ = kProcStructV2Size_;
    } else {
        compute_procs_ = (nvml_device_get_procs_t) platform::dynlib_symbol(library_handle_, "nvmlDeviceGetComputeRunningProcesses_v2");
        graphics_procs_ = (nvml_device_get_procs_t) platform::dynlib_symbol(library_handle_, "nvmlDeviceGetGraphicsRunningProcesses_v2");
        if (!compute_procs_) {
            compute_procs_ = (nvml_device_get_procs_t) platform::dynlib_symbol(library_handle_, "nvmlDeviceGetComputeRunningProcesses");
        }
        if (!graphics_procs_) {
            graphics_procs_ = (nvml_device_get_procs_t) platform::dynlib_symbol(library_handle_, "nvmlDeviceGetGraphicsRunningProcesses");
        }
        proc_struct_size_ = kProcStructV1Size_;
    }

    if (nvmlInit() != NVML_SUCCESS_) {
        platform::dynlib_close(library_handle_);
        library_handle_ = nullptr;
        return false;
    }

    unsigned int device_count = 0;
    if (nvmlDeviceGetCount(&device_count) != NVML_SUCCESS_ || device_count == 0) {
        nvmlShutdown();
        platform::dynlib_close(library_handle_);
        library_handle_ = nullptr;
        return false;
    }

    devices_.reserve(device_count);
    for (unsigned int i = 0; i < device_count; ++i) {
        void * handle = nullptr;
        if (nvmlDeviceGetHandleByIndex(i, &handle) == NVML_SUCCESS_ && handle) {
            devices_.push_back(handle);
        }
    }
    if (devices_.empty()) {
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
    devices_.clear();
    proc_struct_size_ = 0;
    compute_procs_ = nullptr;
    graphics_procs_ = nullptr;
}

namespace {

// Walks the output buffer of a running-processes call and, for every entry
// whose pid matches `wanted_pid`, adds its `usedGpuMemory` to `*sum`. Sets
// `*saw_not_available = true` if any matching entry reports
// NVML_VALUE_NOT_AVAILABLE — the caller uses that to mark the whole sample
// unattributed instead of silently under-counting.
void collect_pid_usage(const uint8_t * buf,
                       unsigned int count,
                       size_t stride,
                       uint32_t wanted_pid,
                       uint64_t & sum,
                       bool & saw_not_available) {
    for (unsigned int i = 0; i < count; ++i) {
        const uint8_t * slot = buf + (size_t) i * stride;
        unsigned int pid = 0;
        uint64_t used = 0;
        std::memcpy(&pid, slot + kProcPidOffset_, sizeof(unsigned int));
        std::memcpy(&used, slot + kProcUsedMemOffset_, sizeof(uint64_t));
        if ((uint32_t) pid != wanted_pid) continue;
        if (used == NVML_NOT_AVAILABLE_) {
            saw_not_available = true;
            continue;
        }
        sum += used;
    }
}

// Calls `fn(device, &count, buf)` with a buffer sized to the current count
// estimate, retrying once if the driver asks for more slots. Writes the
// accumulated per-PID byte count into `*sum`. Returns false only on a
// driver error that should poison the whole sample; an empty process list
// for this device is a valid success (the GPU just isn't being used by any
// process yet).
bool gather_one_device(nvml_device_get_procs_fn fn,
                       void * device,
                       size_t stride,
                       uint32_t wanted_pid,
                       uint64_t & sum,
                       bool & saw_not_available) {
    if (!fn) return true;
    unsigned int count = 0;
    int rc = fn(device, &count, nullptr);
    if (rc == NVML_SUCCESS_) {
        // Zero processes on this device — nothing to add.
        return true;
    }
    if (rc != NVML_ERROR_INSUFFICIENT_SIZE_) {
        // Anything other than "need a bigger buffer" is a real error.
        return false;
    }
    // Grow a byte buffer to count * stride and try again. Leave a little
    // headroom so a new process starting between probes doesn't force a
    // third retry.
    count += 4;
    std::vector<uint8_t> buf((size_t) count * stride, 0);
    rc = fn(device, &count, buf.data());
    if (rc != NVML_SUCCESS_) {
        return false;
    }
    collect_pid_usage(buf.data(), count, stride, wanted_pid, sum, saw_not_available);
    return true;
}

} // namespace

NVMLMonitor::VramSample NVMLMonitor::sample_pid_vram(uint32_t pid) {
    VramSample out;
    if (!initialized_ || devices_.empty() || proc_struct_size_ == 0) {
        return out;
    }
    uint64_t sum = 0;
    bool saw_not_available = false;
    auto compute_fn = (nvml_device_get_procs_fn) compute_procs_;
    auto graphics_fn = (nvml_device_get_procs_fn) graphics_procs_;
    for (void * device : devices_) {
        if (!gather_one_device(compute_fn, device, proc_struct_size_, pid, sum, saw_not_available)) {
            return VramSample{};  // hard failure — don't pretend we measured
        }
        if (!gather_one_device(graphics_fn, device, proc_struct_size_, pid, sum, saw_not_available)) {
            return VramSample{};
        }
    }
    if (saw_not_available) {
        // Driver couldn't attribute a per-process byte count — surface as
        // "not measured" so the UI shows "—" rather than misleading partial
        // number.
        return VramSample{};
    }
    out.bytes = sum;
    out.attributed = true;
    return out;
}

std::string NVMLMonitor::get_device_name() {
    if (!initialized_ || devices_.empty()) return "NVIDIA GPU";
    char buf[128];
    if (nvmlDeviceGetName && nvmlDeviceGetName(devices_[0], buf, sizeof(buf)) == NVML_SUCCESS_) {
        return std::string(buf);
    }
    return "NVIDIA GPU";
}

} // namespace lunavox
