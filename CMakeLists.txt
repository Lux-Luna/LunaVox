cmake_minimum_required(VERSION 3.16)

# On macOS pin a minimum deployment target before project() so clang settles
# ABI decisions early. CMAKE_OSX_ARCHITECTURES is intentionally NOT forced
# here: the Python build driver passes it explicitly (arm64 on Apple Silicon,
# x86_64 on Intel). Both variables can also be set from the command line.
if(APPLE)
    if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND NOT DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS deployment target" FORCE)
    endif()
endif()

project(lunavox VERSION 2.1.6 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LUNAVOX_TIMING "Enable detailed timing instrumentation" OFF)
option(LUNAVOX_USE_OPENMP "Enable OpenMP when available" ON)
option(LUNAVOX_NATIVE "Enable host-specific native optimizations for this project" OFF)
option(LUNAVOX_COPY_RUNTIME "Copy runtime DLLs near built targets on Windows" ON)

set(LUNAVOX_PREBUILT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/llama" CACHE PATH
    "Directory containing prebuilt llama runtime binaries")
set(LUNAVOX_ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/lib/onnx" CACHE PATH
    "Directory containing ONNX Runtime C/C++ SDK (include/ + lib/)")

if(NOT EXISTS "${LUNAVOX_PREBUILT_LIB_DIR}")
    message(FATAL_ERROR "Prebuilt runtime directory not found: ${LUNAVOX_PREBUILT_LIB_DIR}")
endif()
if(NOT EXISTS "${LUNAVOX_ORT_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime headers not found under ${LUNAVOX_ORT_ROOT}.\n"
        "Expected: include/onnxruntime_cxx_api.h")
endif()

if(WIN32)
    add_library(Threads::Threads INTERFACE IMPORTED)
else()
    find_package(Threads REQUIRED)
endif()

set(LUNAVOX_HAVE_OPENMP OFF)
if(LUNAVOX_USE_OPENMP)
    if(WIN32)
        message(STATUS "lunavox OpenMP: disabled on Windows for stable configure")
    else()
        find_package(OpenMP QUIET)
        if(OpenMP_CXX_FOUND)
            set(LUNAVOX_HAVE_OPENMP ON)
            message(STATUS "lunavox OpenMP: enabled")
        else()
            message(STATUS "lunavox OpenMP: not found, building without OpenMP")
        endif()
    endif()
else()
    message(STATUS "lunavox OpenMP: disabled by user")
endif()

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime libonnxruntime
    HINTS "${LUNAVOX_ORT_ROOT}/lib"
    NO_DEFAULT_PATH)

if(NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "Failed to locate ONNX Runtime library under ${LUNAVOX_ORT_ROOT}/lib")
endif()

add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${LUNAVOX_ORT_ROOT}/include")

set(LUNAVOX_RUNTIME_DIRS "${LUNAVOX_PREBUILT_LIB_DIR};${LUNAVOX_ORT_ROOT}/lib")
set(LUNAVOX_RUNTIME_BINS "")
if(WIN32)
    file(GLOB _lunavox_runtime_bins
        "${LUNAVOX_PREBUILT_LIB_DIR}/ggml*.dll"
        "${LUNAVOX_PREBUILT_LIB_DIR}/llama*.dll"
        "${LUNAVOX_PREBUILT_LIB_DIR}/libomp*.dll"
        "${LUNAVOX_PREBUILT_LIB_DIR}/mtmd*.dll"
        "${LUNAVOX_ORT_ROOT}/lib/onnxruntime*.dll")
elseif(APPLE)
    file(GLOB _lunavox_runtime_bins
        "${LUNAVOX_PREBUILT_LIB_DIR}/libggml*.dylib"
        "${LUNAVOX_PREBUILT_LIB_DIR}/libllama*.dylib"
        "${LUNAVOX_PREBUILT_LIB_DIR}/libmtmd*.dylib"
        "${LUNAVOX_ORT_ROOT}/lib/libonnxruntime*.dylib")
else()
    file(GLOB _lunavox_runtime_bins
        "${LUNAVOX_PREBUILT_LIB_DIR}/libggml*.so*"
        "${LUNAVOX_PREBUILT_LIB_DIR}/libllama*.so*"
        "${LUNAVOX_PREBUILT_LIB_DIR}/libmtmd*.so*"
        "${LUNAVOX_ORT_ROOT}/lib/libonnxruntime*.so*")
endif()
if(_lunavox_runtime_bins)
    list(APPEND LUNAVOX_RUNTIME_BINS ${_lunavox_runtime_bins})
    list(REMOVE_DUPLICATES LUNAVOX_RUNTIME_BINS)
endif()

function(lunavox_finalize_binary target_name)
    if(NOT TARGET "${target_name}")
        message(FATAL_ERROR "lunavox_finalize_binary called for missing target: ${target_name}")
    endif()

    if(LUNAVOX_RUNTIME_DIRS)
        if(APPLE)
            # @loader_path resolves relative to the binary itself; adding a
            # second entry one level up keeps install trees bin/+lib/ working.
            set_target_properties(${target_name} PROPERTIES
                BUILD_RPATH "${LUNAVOX_RUNTIME_DIRS};@loader_path;@loader_path/../lib"
                INSTALL_RPATH "@loader_path;@loader_path/../lib"
                MACOSX_RPATH ON)
        elseif(UNIX)
            set_target_properties(${target_name} PROPERTIES
                BUILD_RPATH "${LUNAVOX_RUNTIME_DIRS};$ORIGIN;$ORIGIN/../lib"
                INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib")
        endif()
    endif()

    if(LUNAVOX_COPY_RUNTIME AND LUNAVOX_RUNTIME_BINS)
        foreach(_bin IN LISTS LUNAVOX_RUNTIME_BINS)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_bin}" "$<TARGET_FILE_DIR:${target_name}>"
                VERBATIM)
        endforeach()
    endif()
endfunction()

add_library(lunavox_build_options INTERFACE)
target_compile_features(lunavox_build_options INTERFACE cxx_std_17)
target_link_libraries(lunavox_build_options INTERFACE Threads::Threads)

if(WIN32)
    target_compile_definitions(lunavox_build_options INTERFACE
        NOMINMAX
        _USE_MATH_DEFINES
        _CRT_SECURE_NO_WARNINGS)
endif()

if(LUNAVOX_HAVE_OPENMP)
    target_link_libraries(lunavox_build_options INTERFACE OpenMP::OpenMP_CXX)
endif()
if(LUNAVOX_TIMING)
    target_compile_definitions(lunavox_build_options INTERFACE LUNAVOX_TIMING)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(lunavox_build_options INTERFACE -Wall -Wextra -Wpedantic)
    if(LUNAVOX_NATIVE)
        target_compile_options(lunavox_build_options INTERFACE
            "$<$<COMPILE_LANGUAGE:CXX>:-march=native>")
    endif()
elseif(MSVC)
    target_compile_options(lunavox_build_options INTERFACE /W4 /utf-8)
endif()

# platform_utils: the single home for every OS branch. Must not depend on
# anything except the standard library and OS headers — all other layers
# depend on *it*.
add_library(lunavox_platform_utils STATIC src/platform_utils.cpp)
target_include_directories(lunavox_platform_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(lunavox_platform_utils PUBLIC lunavox_build_options)
if(WIN32)
    target_link_libraries(lunavox_platform_utils PUBLIC psapi)
elseif(APPLE)
    target_link_libraries(lunavox_platform_utils PUBLIC "-framework Foundation")
else()
    target_link_libraries(lunavox_platform_utils PUBLIC dl)
endif()

add_library(lunavox_logger STATIC src/logger.cpp)
target_include_directories(lunavox_logger PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(lunavox_logger PUBLIC lunavox_build_options)

add_library(lunavox_audio_io STATIC src/audio_io.cpp)
target_include_directories(lunavox_audio_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(lunavox_audio_io PUBLIC lunavox_build_options lunavox_logger)

add_library(lunavox_tokenizer STATIC src/text_tokenizer.cpp)
target_include_directories(lunavox_tokenizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(lunavox_tokenizer PUBLIC lunavox_build_options lunavox_logger)

add_library(lunavox_audio_decoder STATIC
    src/audio_decoder.cpp
    src/provider_policy.cpp)
target_include_directories(lunavox_audio_decoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${LUNAVOX_ORT_ROOT}/include)
target_link_libraries(lunavox_audio_decoder PUBLIC
    lunavox_build_options
    onnxruntime::onnxruntime
    lunavox_logger
    lunavox_platform_utils)

add_library(lunavox_predictor STATIC
    src/llama_wrapper.cpp
    src/assets_manager.cpp
    src/talker_predictor_llama.cpp)
target_include_directories(lunavox_predictor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(lunavox_predictor PUBLIC
    lunavox_build_options
    lunavox_logger
    lunavox_platform_utils)

add_library(lunavox_core STATIC src/lunavox_engine.cpp src/nvml_monitor.cpp)
target_include_directories(lunavox_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(lunavox_core PUBLIC
    lunavox_build_options
    lunavox_tokenizer
    lunavox_predictor
    lunavox_audio_decoder
    lunavox_audio_io
    lunavox_logger
    lunavox_platform_utils)

add_library(lunavox_shared SHARED src/lunavox_c_api.cpp)
set_target_properties(lunavox_shared PROPERTIES
    OUTPUT_NAME "lunavox"
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    # Auto-export every extern "C" symbol from the C API translation
    # unit without maintaining a .def file. Safe here because
    # lunavox_c_api.cpp is the only source file in this target, so only
    # the public C ABI surfaces.
    WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_include_directories(lunavox_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(lunavox_shared PRIVATE lunavox_core)
lunavox_finalize_binary(lunavox_shared)

add_executable(lunavox-cli src/main.cpp)
target_link_libraries(lunavox-cli PRIVATE lunavox_core)
lunavox_finalize_binary(lunavox-cli)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME cli_help_smoke COMMAND lunavox-cli --help)
endif()

install(TARGETS
    lunavox_platform_utils
    lunavox_logger
    lunavox_audio_io
    lunavox_tokenizer
    lunavox_audio_decoder
    lunavox_predictor
    lunavox_core
    lunavox_shared
    lunavox-cli
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
