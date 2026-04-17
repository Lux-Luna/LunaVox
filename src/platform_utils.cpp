#include "platform_utils.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <shellapi.h>
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <objc/objc.h>
#include <objc/message.h>
#else
#ifndef _WIN32
#include <sys/resource.h>
#endif
#endif

namespace lunavox {
namespace platform {

namespace fs = std::filesystem;

// --- Dynamic library loading --------------------------------------------------

void * dynlib_open(const char * path) {
#ifdef _WIN32
    // LOAD_WITH_ALTERED_SEARCH_PATH makes Windows resolve dependent DLLs
    // against the loaded library's own directory first. Without it,
    // loading `build/llama.dll` from a non-build host process would fail
    // to find `build/ggml.dll` because the default search path skips the
    // DLL's own directory. The CLI binary happens to work unmodified
    // because it already runs from build/, so its exe-dir and the
    // dependency-dir coincide — but the Python ctypes binding doesn't.
    return (void *) LoadLibraryExA(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

void dynlib_close(void * handle) {
    if (!handle) return;
#ifdef _WIN32
    FreeLibrary((HMODULE) handle);
#else
    dlclose(handle);
#endif
}

void * dynlib_symbol(void * handle, const char * name) {
    if (!handle) return nullptr;
#ifdef _WIN32
    return (void *) GetProcAddress((HMODULE) handle, name);
#else
    return dlsym(handle, name);
#endif
}

void * dynlib_find_loaded(const char * base_name) {
#ifdef _WIN32
    return (void *) GetModuleHandleA(base_name);
#else
    (void) base_name;
    return nullptr;
#endif
}

void * dynlib_try_open_in(const std::string & dir,
                          std::initializer_list<const char *> candidates,
                          std::string * out_path_tried) {
    const fs::path base = fs::u8path(dir);
    for (const char * name : candidates) {
        const std::string s = (base / name).string();
        if (out_path_tried) *out_path_tried = s;
        void * h = dynlib_open(s.c_str());
        if (h) return h;
    }
    return nullptr;
}

void * dynlib_try_open_in(const std::string & dir,
                          const std::vector<std::string> & candidates,
                          std::string * out_path_tried) {
    const fs::path base = fs::u8path(dir);
    for (const std::string & name : candidates) {
        const std::string s = (base / name).string();
        if (out_path_tried) *out_path_tried = s;
        void * h = dynlib_open(s.c_str());
        if (h) return h;
    }
    return nullptr;
}

std::vector<std::string> dynlib_candidate_names(const char * stem) {
    if (!stem) return {};
#ifdef _WIN32
    return { std::string(stem) + ".dll" };
#elif defined(__APPLE__)
    return {
        std::string("lib") + stem + ".dylib",
        std::string("lib") + stem + ".so",
    };
#else
    return { std::string("lib") + stem + ".so" };
#endif
}

bool dynlib_supports_loaded_lookup() {
#ifdef _WIN32
    return true;
#else
    return false;
#endif
}

// --- Environment --------------------------------------------------------------

bool set_env(const char * name, const char * value) {
    if (!name || !value) return false;
#ifdef _WIN32
    return _putenv_s(name, value) == 0;
#else
    return setenv(name, value, 1) == 0;
#endif
}

// --- Executable / argv --------------------------------------------------------

std::string executable_dir() {
#ifdef _WIN32
    char path[MAX_PATH];
    DWORD size = GetModuleFileNameA(nullptr, path, MAX_PATH);
    if (size == 0) return ".";
    std::string s(path, size);
    size_t last = s.find_last_of("\\/");
    return last == std::string::npos ? std::string(".") : s.substr(0, last);
#elif defined(__APPLE__)
    char buf[4096];
    uint32_t bufsize = sizeof(buf);
    if (_NSGetExecutablePath(buf, &bufsize) != 0) return ".";
    std::string s(buf);
    size_t last = s.find_last_of('/');
    return last == std::string::npos ? std::string(".") : s.substr(0, last);
#else
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf));
    if (n <= 0) return ".";
    std::string s(buf, (size_t) n);
    size_t last = s.find_last_of('/');
    return last == std::string::npos ? std::string(".") : s.substr(0, last);
#endif
}

std::vector<std::string> utf8_args(int argc, char ** argv) {
    std::vector<std::string> args;
#ifdef _WIN32
    int wide_argc = 0;
    LPWSTR * wide_argv = CommandLineToArgvW(GetCommandLineW(), &wide_argc);
    if (wide_argv && wide_argc > 0) {
        args.reserve((size_t) wide_argc);
        for (int i = 0; i < wide_argc; ++i) {
            const std::wstring ws = wide_argv[i] ? wide_argv[i] : L"";
            if (ws.empty()) {
                args.emplace_back();
                continue;
            }
            int size = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int) ws.size(),
                                           nullptr, 0, nullptr, nullptr);
            if (size <= 0) {
                args.emplace_back();
                continue;
            }
            std::string utf8((size_t) size, '\0');
            WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int) ws.size(),
                                &utf8[0], size, nullptr, nullptr);
            args.emplace_back(std::move(utf8));
        }
        LocalFree(wide_argv);
        return args;
    }
#endif
    args.reserve((size_t) argc);
    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i] ? argv[i] : "");
    }
    return args;
}

// --- Process memory ----------------------------------------------------------

bool process_memory_snapshot(ProcessMemorySnapshot & out) {
#ifdef __APPLE__
    mach_task_basic_info_data_t basic_info = {};
    mach_msg_type_number_t basic_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&basic_info), &basic_count) != KERN_SUCCESS) {
        return false;
    }
    out.rss_bytes = (uint64_t) basic_info.resident_size;

    task_vm_info_data_t vm_info = {};
    mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&vm_info), &vm_count) == KERN_SUCCESS) {
        out.phys_footprint_bytes = (uint64_t) vm_info.phys_footprint;
    } else {
        out.phys_footprint_bytes = out.rss_bytes;
    }
    return true;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        out.rss_bytes = (uint64_t) counters.WorkingSetSize;
        out.phys_footprint_bytes = out.rss_bytes;
        return true;
    }
    return false;
#else
    struct rusage usage = {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return false;
    }
    out.rss_bytes = (uint64_t) usage.ru_maxrss * 1024ULL;
    out.phys_footprint_bytes = out.rss_bytes;
    return true;
#endif
}

uint32_t current_pid() {
#ifdef _WIN32
    return (uint32_t) GetCurrentProcessId();
#else
    return (uint32_t) getpid();
#endif
}

// --- Memory-mapped read-only files -------------------------------------------

#ifndef _WIN32
static std::string posix_errno_string(int err_no) {
    char buf[128] = {};
#if ((_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600) && !_GNU_SOURCE) || defined(__APPLE__)
    if (strerror_r(err_no, buf, sizeof(buf)) == 0) {
        return std::string(buf);
    }
    return "errno=" + std::to_string(err_no);
#else
    return std::string(strerror_r(err_no, buf, sizeof(buf)));
#endif
}
#endif

bool mmap_file_open(const std::string & path, MmapFile & out, std::string & err) {
    out = MmapFile{};
#ifdef _WIN32
    HANDLE file = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        err = "CreateFile failed";
        return false;
    }

    LARGE_INTEGER size_li = {};
    if (!GetFileSizeEx(file, &size_li) || size_li.QuadPart <= 0) {
        CloseHandle(file);
        err = "GetFileSizeEx failed";
        return false;
    }
    if ((uint64_t) size_li.QuadPart > (uint64_t) std::numeric_limits<size_t>::max()) {
        CloseHandle(file);
        err = "File too large for mmap on this platform";
        return false;
    }
    const size_t file_size = (size_t) size_li.QuadPart;

    HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping) {
        CloseHandle(file);
        err = "CreateFileMapping failed";
        return false;
    }

    void * view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        CloseHandle(mapping);
        CloseHandle(file);
        err = "MapViewOfFile failed";
        return false;
    }

    out.file_handle = reinterpret_cast<intptr_t>(file);
    out.mapping_handle = reinterpret_cast<intptr_t>(mapping);
    out.data = reinterpret_cast<const uint8_t *>(view);
    out.size = file_size;
    return true;
#else
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        err = "open failed: " + posix_errno_string(errno);
        return false;
    }

    struct stat st = {};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        err = "fstat failed: " + posix_errno_string(errno);
        return false;
    }
    if ((uint64_t) st.st_size > (uint64_t) std::numeric_limits<size_t>::max()) {
        ::close(fd);
        err = "File too large for mmap on this platform";
        return false;
    }
    const size_t file_size = (size_t) st.st_size;

    void * view = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (view == MAP_FAILED) {
        ::close(fd);
        err = "mmap failed: " + posix_errno_string(errno);
        return false;
    }

    out.file_handle = (intptr_t) fd;
    out.mapping_handle = -1;
    out.data = reinterpret_cast<const uint8_t *>(view);
    out.size = file_size;
    return true;
#endif
}

void mmap_file_close(MmapFile & m) {
#ifdef _WIN32
    if (m.data) {
        UnmapViewOfFile(m.data);
    }
    if (m.mapping_handle != -1) {
        CloseHandle(reinterpret_cast<HANDLE>(m.mapping_handle));
    }
    if (m.file_handle != -1 &&
        m.file_handle != reinterpret_cast<intptr_t>(INVALID_HANDLE_VALUE)) {
        CloseHandle(reinterpret_cast<HANDLE>(m.file_handle));
    }
#else
    if (m.data && m.size > 0) {
        munmap(const_cast<uint8_t *>(m.data), m.size);
    }
    if (m.file_handle >= 0) {
        ::close((int) m.file_handle);
    }
#endif
    m.data = nullptr;
    m.size = 0;
    m.file_handle = -1;
    m.mapping_handle = -1;
}

// --- Apple autorelease pool --------------------------------------------------

#ifdef __APPLE__
AutoreleasePoolScope::AutoreleasePoolScope() {
    id pool = ((id(*)(id, SEL))objc_msgSend)(
        (id) objc_getClass("NSAutoreleasePool"),
        sel_registerName("new"));
    pool_ = (void *) pool;
}
AutoreleasePoolScope::~AutoreleasePoolScope() {
    if (!pool_) return;
    ((void(*)(id, SEL))objc_msgSend)((id) pool_, sel_registerName("drain"));
    pool_ = nullptr;
}
#else
AutoreleasePoolScope::AutoreleasePoolScope() = default;
AutoreleasePoolScope::~AutoreleasePoolScope() = default;
#endif

} // namespace platform
} // namespace lunavox
