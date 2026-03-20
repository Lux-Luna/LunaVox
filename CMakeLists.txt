cmake_minimum_required(VERSION 3.16)

project(qwen3-tts VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QWEN3_TTS_TIMING "Enable detailed timing instrumentation" OFF)
option(QWEN3_TTS_USE_OPENMP "Enable OpenMP when available" ON)
option(QWEN3_TTS_NATIVE "Enable host-specific native optimizations for this project" OFF)
option(QWEN3_TTS_COPY_RUNTIME "Copy runtime DLLs near built targets on Windows" ON)

set(QWEN3_TTS_PREBUILT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib" CACHE PATH
    "Directory containing prebuilt llama runtime binaries")
set(QWEN3_TTS_ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime" CACHE PATH
    "Directory containing ONNX Runtime C/C++ SDK (include/ + lib/)")

if(NOT EXISTS "${QWEN3_TTS_PREBUILT_LIB_DIR}")
    message(FATAL_ERROR "Prebuilt runtime directory not found: ${QWEN3_TTS_PREBUILT_LIB_DIR}")
endif()
if(NOT EXISTS "${QWEN3_TTS_ORT_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime headers not found under ${QWEN3_TTS_ORT_ROOT}.\n"
        "Expected: include/onnxruntime_cxx_api.h")
endif()

find_package(Threads REQUIRED)

set(QWEN3_TTS_HAVE_OPENMP OFF)
if(QWEN3_TTS_USE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        set(QWEN3_TTS_HAVE_OPENMP ON)
        message(STATUS "qwen3-tts OpenMP: enabled")
    else()
        message(STATUS "qwen3-tts OpenMP: not found, building without OpenMP")
    endif()
else()
    message(STATUS "qwen3-tts OpenMP: disabled by user")
endif()

if(WIN32)
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        HINTS "${QWEN3_TTS_ORT_ROOT}/lib"
        NO_DEFAULT_PATH)
else()
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        HINTS "${QWEN3_TTS_ORT_ROOT}/lib"
        NO_DEFAULT_PATH)
endif()

if(NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "Failed to locate ONNX Runtime library under ${QWEN3_TTS_ORT_ROOT}/lib")
endif()

add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${QWEN3_TTS_ORT_ROOT}/include")

set(QWEN3_TTS_RUNTIME_DIRS "${QWEN3_TTS_PREBUILT_LIB_DIR};${QWEN3_TTS_ORT_ROOT}/lib")
set(QWEN3_TTS_RUNTIME_DLLS "")
if(WIN32)
    file(GLOB _qwen3_tts_runtime_dlls
        "${QWEN3_TTS_PREBUILT_LIB_DIR}/ggml*.dll"
        "${QWEN3_TTS_PREBUILT_LIB_DIR}/llama*.dll"
        "${QWEN3_TTS_PREBUILT_LIB_DIR}/libomp*.dll"
        "${QWEN3_TTS_ORT_ROOT}/lib/onnxruntime*.dll")
    if(_qwen3_tts_runtime_dlls)
        list(APPEND QWEN3_TTS_RUNTIME_DLLS ${_qwen3_tts_runtime_dlls})
        list(REMOVE_DUPLICATES QWEN3_TTS_RUNTIME_DLLS)
    endif()
endif()

function(qwen3_tts_finalize_binary target_name)
    if(NOT TARGET "${target_name}")
        message(FATAL_ERROR "qwen3_tts_finalize_binary called for missing target: ${target_name}")
    endif()

    if(QWEN3_TTS_RUNTIME_DIRS)
        if(APPLE)
            set(_origin "@loader_path")
            set_target_properties(${target_name} PROPERTIES
                BUILD_RPATH "${QWEN3_TTS_RUNTIME_DIRS}"
                INSTALL_RPATH "${_origin};${_origin}/../lib")
        elseif(UNIX)
            set(_origin "$ORIGIN")
            set_target_properties(${target_name} PROPERTIES
                BUILD_RPATH "${QWEN3_TTS_RUNTIME_DIRS}"
                INSTALL_RPATH "${_origin};${_origin}/../lib")
        endif()
    endif()

    if(WIN32 AND QWEN3_TTS_COPY_RUNTIME AND QWEN3_TTS_RUNTIME_DLLS)
        foreach(_dll IN LISTS QWEN3_TTS_RUNTIME_DLLS)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_dll}" "$<TARGET_FILE_DIR:${target_name}>"
                VERBATIM)
        endforeach()
    endif()
endfunction()

add_library(qwen3_tts_build_options INTERFACE)
target_compile_features(qwen3_tts_build_options INTERFACE cxx_std_17)
target_link_libraries(qwen3_tts_build_options INTERFACE Threads::Threads)

if(WIN32)
    target_compile_definitions(qwen3_tts_build_options INTERFACE
        NOMINMAX
        _USE_MATH_DEFINES
        _CRT_SECURE_NO_WARNINGS)
endif()

if(QWEN3_TTS_HAVE_OPENMP)
    target_link_libraries(qwen3_tts_build_options INTERFACE OpenMP::OpenMP_CXX)
endif()
if(QWEN3_TTS_TIMING)
    target_compile_definitions(qwen3_tts_build_options INTERFACE QWEN3_TTS_TIMING)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(qwen3_tts_build_options INTERFACE -Wall -Wextra -Wpedantic)
    if(QWEN3_TTS_NATIVE)
        target_compile_options(qwen3_tts_build_options INTERFACE
            "$<$<COMPILE_LANGUAGE:CXX>:-march=native>")
    endif()
elseif(MSVC)
    target_compile_options(qwen3_tts_build_options INTERFACE /W4 /utf-8)
endif()

add_library(text_tokenizer STATIC src/text_tokenizer.cpp)
target_include_directories(text_tokenizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(text_tokenizer PUBLIC qwen3_tts_build_options)

add_library(onnx_audio_runtime STATIC src/onnx_audio_runtime.cpp)
target_include_directories(onnx_audio_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${QWEN3_TTS_ORT_ROOT}/include)
target_link_libraries(onnx_audio_runtime PUBLIC
    qwen3_tts_build_options
    onnxruntime::onnxruntime)

add_library(talker_predictor_llama STATIC
    src/llama_wrapper.cpp
    src/assets_manager.cpp
    src/talker_predictor_llama.cpp)
target_include_directories(talker_predictor_llama PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(talker_predictor_llama PUBLIC qwen3_tts_build_options)
if(UNIX AND NOT APPLE)
    target_link_libraries(talker_predictor_llama PUBLIC dl)
endif()

add_library(qwen3_tts STATIC src/qwen3_tts.cpp)
target_include_directories(qwen3_tts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(qwen3_tts PUBLIC
    qwen3_tts_build_options
    text_tokenizer
    talker_predictor_llama
    onnx_audio_runtime)
if(WIN32)
    target_link_libraries(qwen3_tts PUBLIC psapi)
endif()

add_library(qwen3tts_shared SHARED src/qwen3tts_c_api.cpp)
set_target_properties(qwen3tts_shared PROPERTIES
    OUTPUT_NAME "qwen3tts"
    VERSION ${PROJECT_VERSION}
    SOVERSION 0)
target_include_directories(qwen3tts_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(qwen3tts_shared PRIVATE qwen3_tts)
qwen3_tts_finalize_binary(qwen3tts_shared)

add_executable(qwen3-tts-cli src/main.cpp)
target_link_libraries(qwen3-tts-cli PRIVATE qwen3_tts)
qwen3_tts_finalize_binary(qwen3-tts-cli)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME cli_help_smoke COMMAND qwen3-tts-cli --help)
endif()

install(TARGETS
    text_tokenizer
    onnx_audio_runtime
    talker_predictor_llama
    qwen3_tts
    qwen3tts_shared
    qwen3-tts-cli
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
