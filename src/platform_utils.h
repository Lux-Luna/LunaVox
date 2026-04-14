#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

// platform_utils is the single home for every OS branch in the engine.
// All LoadLibrary/dlopen, mmap/CreateFileMapping, mach/GetProcessMemoryInfo,
// CommandLineToArgvW, GetModuleFileName, readlink("/proc/self/exe"), etc.
// live behind this facade. No other translation unit is allowed to include
// <windows.h>, <dlfcn.h>, <mach/mach.h>, <sys/mman.h>, <psapi.h> or test
// _WIN32 / __APPLE__ / __linux__ macros. Break that rule and the grep guard
// in docs/ one-shot refactor plan will fail.

namespace lunavox {
namespace platform {

// --- Dynamic library loading --------------------------------------------------

void * dynlib_open(const char * path);
void   dynlib_close(void * handle);
void * dynlib_symbol(void * handle, const char * name);

// On Windows this probes GetModuleHandle for a library that the process has
// already mapped (typically pulled in as a transitive dependency). Returns
// nullptr on POSIX where there is no equivalent cheap lookup.
void * dynlib_find_loaded(const char * base_name);

// Try a list of candidate filenames inside `dir`, returning the first that
// loads. `out_path_tried` (optional) is overwritten with the last path the
// function attempted — useful for crafting error messages.
void * dynlib_try_open_in(const std::string & dir,
                          std::initializer_list<const char *> candidates,
                          std::string * out_path_tried = nullptr);

void * dynlib_try_open_in(const std::string & dir,
                          const std::vector<std::string> & candidates,
                          std::string * out_path_tried = nullptr);

// Expand a library stem ("llama", "ggml-base") into plausible file names for
// the current platform. Windows: {"<stem>.dll"}; macOS: {"lib<stem>.dylib",
// "lib<stem>.so"}; Linux / other POSIX: {"lib<stem>.so"}.
std::vector<std::string> dynlib_candidate_names(const char * stem);

// True on platforms where dynlib_find_loaded can cheaply look up an
// already-mapped module (Windows). Callers use this to decide whether to
// probe for a pre-loaded dependent library before pulling a second copy.
bool dynlib_supports_loaded_lookup();

// --- Environment --------------------------------------------------------------

// Set a process-wide environment variable. Wraps _putenv_s on Windows and
// setenv on POSIX so callers never test _WIN32. Returns true on success.
bool set_env(const char * name, const char * value);

// --- Executable / argv --------------------------------------------------------

// Directory containing the current executable (no trailing separator).
// Returns "." if the platform query fails.
std::string executable_dir();

// Collect argv as UTF-8, regardless of the host encoding. On Windows this
// re-fetches CommandLineToArgvW and converts via WideCharToMultiByte; on
// POSIX argv is returned as-is.
std::vector<std::string> utf8_args(int argc, char ** argv);

// --- Process memory ----------------------------------------------------------

struct ProcessMemorySnapshot {
    uint64_t rss_bytes = 0;             // "working set" / RSS
    uint64_t phys_footprint_bytes = 0;  // macOS phys_footprint; equals rss elsewhere
};

bool process_memory_snapshot(ProcessMemorySnapshot & out);

// --- Memory-mapped read-only files -------------------------------------------

// Opaque mapping handles. On Windows `file_handle` is a HANDLE and
// `mapping_handle` is the CreateFileMapping HANDLE. On POSIX `file_handle`
// is the fd and `mapping_handle` is unused (-1). Consumers treat these as
// opaque and always pass them back to `mmap_file_close`.
struct MmapFile {
    const uint8_t * data = nullptr;
    size_t          size = 0;
    intptr_t        file_handle = -1;
    intptr_t        mapping_handle = -1;
};

bool mmap_file_open(const std::string & path, MmapFile & out, std::string & err);
void mmap_file_close(MmapFile & m);

// --- Apple autorelease pool ---------------------------------------------------

// RAII guard that creates an NSAutoreleasePool on macOS and drains it on
// destruction. On non-Apple platforms it's an empty no-op — safe to drop on
// every hot synthesis path so we don't need platform branches at call sites.
// Used by the C API synth path to reclaim Objective-C temporaries when
// invoked from non-main threads (e.g. Python, Unity).
class AutoreleasePoolScope {
public:
    AutoreleasePoolScope();
    ~AutoreleasePoolScope();
    AutoreleasePoolScope(const AutoreleasePoolScope &) = delete;
    AutoreleasePoolScope & operator=(const AutoreleasePoolScope &) = delete;
private:
    void * pool_ = nullptr;
};

} // namespace platform
} // namespace lunavox
