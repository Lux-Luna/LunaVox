cmake_minimum_required(VERSION 3.14)
project(qwen3-tts-ggml VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QWEN3_TTS_TIMING "Enable detailed timing instrumentation" OFF)
option(QWEN3_TTS_COREML "Enable CoreML code predictor bridge on macOS" ON)
option(QWEN3_TTS_USE_OPENMP "Enable OpenMP when it is available" ON)
option(QWEN3_TTS_NATIVE "Enable host-specific native optimizations for this top-level project" OFF)
option(QWEN3_TTS_COPY_GGML_RUNTIME
    "Copy discovered GGML runtime DLLs next to built qwen3-tts targets on Windows"
    ON)

set(GGML_BUILD_DIR "" CACHE PATH
    "Path to a configured GGML build tree")

if(APPLE AND QWEN3_TTS_COREML)
    enable_language(OBJCXX)
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

function(qwen3_tts_append_if_exists out_var path)
    if(EXISTS "${path}")
        set(${out_var} "${${out_var}};${path}" PARENT_SCOPE)
    endif()
endfunction()

function(qwen3_tts_read_cache_value cache_file entry out_var)
    if(NOT EXISTS "${cache_file}")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    file(STRINGS "${cache_file}" _cache_line REGEX "^${entry}(:[^=]+)?=")
    if(_cache_line)
        list(GET _cache_line 0 _cache_line_first)
        string(REGEX REPLACE "^[^=]+=(.*)$" "\\1" _cache_value "${_cache_line_first}")
        set(${out_var} "${_cache_value}" PARENT_SCOPE)
    else()
        set(${out_var} "" PARENT_SCOPE)
    endif()
endfunction()

function(qwen3_tts_import_ggml_build_tree build_dir)
    set(_cache_file "${build_dir}/CMakeCache.txt")
    qwen3_tts_read_cache_value("${_cache_file}" "ggml_SOURCE_DIR" _ggml_source_dir)
    qwen3_tts_read_cache_value("${_cache_file}" "GGML_AVAILABLE_BACKENDS" _ggml_available_backends)

    if(NOT _ggml_source_dir)
        message(FATAL_ERROR
            "GGML build tree at ${build_dir} does not expose ggml_SOURCE_DIR in CMakeCache.txt")
    endif()

    set(_ggml_include_dir "${_ggml_source_dir}/include")
    set(_ggml_lib_dir "${build_dir}/src")
    if(NOT EXISTS "${_ggml_include_dir}")
        message(FATAL_ERROR
            "GGML build tree include directory not found: ${_ggml_include_dir}")
    endif()
    if(NOT EXISTS "${_ggml_lib_dir}")
        message(FATAL_ERROR
            "GGML build tree library directory not found: ${_ggml_lib_dir}")
    endif()

    find_library(_ggml_library ggml REQUIRED HINTS "${_ggml_lib_dir}" NO_DEFAULT_PATH)
    find_library(_ggml_base_library ggml-base REQUIRED HINTS "${_ggml_lib_dir}" NO_DEFAULT_PATH)

    add_library(ggml::ggml UNKNOWN IMPORTED)
    set_target_properties(ggml::ggml PROPERTIES
        IMPORTED_LOCATION "${_ggml_library}"
        INTERFACE_INCLUDE_DIRECTORIES "${_ggml_include_dir}")

    add_library(ggml::ggml-base UNKNOWN IMPORTED)
    set_target_properties(ggml::ggml-base PROPERTIES
        IMPORTED_LOCATION "${_ggml_base_library}"
        INTERFACE_INCLUDE_DIRECTORIES "${_ggml_include_dir}")

    set(_ggml_backend_targets "")
    if(_ggml_available_backends)
        string(REPLACE ";" ";" _ggml_available_backends "${_ggml_available_backends}")
        foreach(_ggml_backend IN LISTS _ggml_available_backends)
            string(REPLACE "-" "_" _ggml_backend_var "${_ggml_backend}")
            string(TOUPPER "${_ggml_backend_var}" _ggml_backend_var)
            set(_ggml_backend_search_dirs
                "${_ggml_lib_dir}"
                "${_ggml_lib_dir}/${_ggml_backend}")
            find_library(${_ggml_backend_var}_LIBRARY
                ${_ggml_backend}
                REQUIRED
                HINTS ${_ggml_backend_search_dirs}
                NO_DEFAULT_PATH)
            add_library(ggml::${_ggml_backend} UNKNOWN IMPORTED)
            set_target_properties(ggml::${_ggml_backend} PROPERTIES
                IMPORTED_LOCATION "${${_ggml_backend_var}_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${_ggml_include_dir}"
                INTERFACE_LINK_LIBRARIES "ggml::ggml-base")
            list(APPEND _ggml_backend_targets "ggml::${_ggml_backend}")
        endforeach()
    endif()

    set_target_properties(ggml::ggml PROPERTIES
        INTERFACE_LINK_LIBRARIES "ggml::ggml-base;${_ggml_backend_targets}")

    add_library(ggml::all INTERFACE IMPORTED)
    set_target_properties(ggml::all PROPERTIES
        INTERFACE_LINK_LIBRARIES "${_ggml_backend_targets}")

    set(GGML_INCLUDE_DIR "${_ggml_include_dir}" PARENT_SCOPE)
    set(GGML_LIB_DIR "${_ggml_lib_dir}" PARENT_SCOPE)
    set(GGML_AVAILABLE_BACKENDS "${_ggml_available_backends}" PARENT_SCOPE)
endfunction()

function(qwen3_tts_finalize_binary target_name)
    if(NOT TARGET "${target_name}")
        message(FATAL_ERROR "qwen3_tts_finalize_binary called for missing target: ${target_name}")
    endif()

    if(QWEN3_TTS_GGML_RUNTIME_DIRS)
        if(APPLE)
            set(_origin "@loader_path")
            set_target_properties(${target_name} PROPERTIES
                BUILD_RPATH "${QWEN3_TTS_GGML_RUNTIME_DIRS}"
                INSTALL_RPATH "${_origin};${_origin}/../lib")
        elseif(UNIX)
            set(_origin "$ORIGIN")
            set_target_properties(${target_name} PROPERTIES
                BUILD_RPATH "${QWEN3_TTS_GGML_RUNTIME_DIRS}"
                INSTALL_RPATH "${_origin};${_origin}/../lib")
        endif()
    endif()

    if(WIN32 AND QWEN3_TTS_COPY_GGML_RUNTIME AND QWEN3_TTS_GGML_DLLS)
        foreach(_dll IN LISTS QWEN3_TTS_GGML_DLLS)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_dll}" "$<TARGET_FILE_DIR:${target_name}>"
                VERBATIM)
        endforeach()
    endif()
endfunction()

if(NOT GGML_BUILD_DIR)
    foreach(_ggml_build_hint
            "${CMAKE_CURRENT_SOURCE_DIR}/ggml/build"
            "${CMAKE_CURRENT_SOURCE_DIR}/ggml1/build")
        if(EXISTS "${_ggml_build_hint}/CMakeCache.txt")
            set(GGML_BUILD_DIR "${_ggml_build_hint}")
            break()
        endif()
    endforeach()
endif()

set(_qwen3_tts_ggml_provider "")
set(_qwen3_tts_ggml_package_hints "")

if(GGML_BUILD_DIR)
    foreach(_ggml_package_hint
            "${GGML_BUILD_DIR}/lib/cmake/ggml"
            "${GGML_BUILD_DIR}/share/cmake/ggml")
        if(EXISTS "${_ggml_package_hint}/ggml-config.cmake")
            list(APPEND _qwen3_tts_ggml_package_hints "${_ggml_package_hint}")
        endif()
    endforeach()

    if(_qwen3_tts_ggml_package_hints)
        find_package(ggml CONFIG QUIET PATHS ${_qwen3_tts_ggml_package_hints} NO_DEFAULT_PATH)
        if(ggml_FOUND)
            set(_qwen3_tts_ggml_provider "package")
        endif()
    endif()

    if(NOT ggml_FOUND AND EXISTS "${GGML_BUILD_DIR}/CMakeCache.txt")
        qwen3_tts_import_ggml_build_tree("${GGML_BUILD_DIR}")
        set(ggml_FOUND TRUE)
        set(_qwen3_tts_ggml_provider "build-tree")
    endif()
endif()

if(NOT ggml_FOUND)
    find_package(ggml CONFIG QUIET)
    if(ggml_FOUND)
        set(_qwen3_tts_ggml_provider "package")
    endif()
endif()

if(NOT ggml_FOUND)
    string(REPLACE ";" "\n  " _qwen3_tts_hints_display "${_qwen3_tts_ggml_package_hints}")
    message(FATAL_ERROR
        "Failed to find GGML.\n"
        "Provide one of:\n"
        "  -DGGML_BUILD_DIR=<path-to-configured-ggml-build>\n"
        "  -Dggml_DIR=<path-to-ggml-config.cmake>\n"
        "  -DCMAKE_PREFIX_PATH=<ggml-install-prefix>\n"
        "Package hints searched:\n  ${_qwen3_tts_hints_display}")
endif()

add_library(qwen3_tts_ggml INTERFACE)
target_link_libraries(qwen3_tts_ggml INTERFACE ggml::ggml)
if(DEFINED GGML_INCLUDE_DIR AND EXISTS "${GGML_INCLUDE_DIR}")
    target_include_directories(qwen3_tts_ggml INTERFACE "${GGML_INCLUDE_DIR}")
endif()

set(QWEN3_TTS_GGML_BACKENDS "CPU")
if(DEFINED GGML_AVAILABLE_BACKENDS AND GGML_AVAILABLE_BACKENDS)
    foreach(_backend_target IN LISTS GGML_AVAILABLE_BACKENDS)
        string(REPLACE "ggml-" "" _backend_label "${_backend_target}")
        string(TOUPPER "${_backend_label}" _backend_label)
        if(NOT _backend_label STREQUAL "CPU")
            list(APPEND QWEN3_TTS_GGML_BACKENDS "${_backend_label}")
        endif()
    endforeach()
endif()
foreach(_backend_name BLAS CUDA HIP METAL VULKAN WEBGPU OPENCL SYCL OPENVINO RPC)
    if(DEFINED GGML_${_backend_name} AND GGML_${_backend_name})
        list(APPEND QWEN3_TTS_GGML_BACKENDS "${_backend_name}")
    endif()
endforeach()
list(REMOVE_DUPLICATES QWEN3_TTS_GGML_BACKENDS)

set(QWEN3_TTS_GGML_RUNTIME_DIRS "")
if(DEFINED GGML_LIB_DIR AND EXISTS "${GGML_LIB_DIR}")
    list(APPEND QWEN3_TTS_GGML_RUNTIME_DIRS "${GGML_LIB_DIR}")
    get_filename_component(_qwen3_tts_ggml_prefix "${GGML_LIB_DIR}" DIRECTORY)
    qwen3_tts_append_if_exists(QWEN3_TTS_GGML_RUNTIME_DIRS "${_qwen3_tts_ggml_prefix}/bin")
endif()
if(GGML_BUILD_DIR)
    foreach(_dir
            "${GGML_BUILD_DIR}/bin"
            "${GGML_BUILD_DIR}/src"
            "${GGML_BUILD_DIR}/src/Release"
            "${GGML_BUILD_DIR}/src/Debug"
            "${GGML_BUILD_DIR}/src/RelWithDebInfo"
            "${GGML_BUILD_DIR}/src/MinSizeRel")
        qwen3_tts_append_if_exists(QWEN3_TTS_GGML_RUNTIME_DIRS "${_dir}")
    endforeach()
endif()
list(REMOVE_DUPLICATES QWEN3_TTS_GGML_RUNTIME_DIRS)

set(QWEN3_TTS_GGML_DLLS "")
if(WIN32 AND QWEN3_TTS_GGML_RUNTIME_DIRS)
    foreach(_runtime_dir IN LISTS QWEN3_TTS_GGML_RUNTIME_DIRS)
        if(EXISTS "${_runtime_dir}")
            file(GLOB _runtime_dlls "${_runtime_dir}/ggml*.dll")
            if(_runtime_dlls)
                list(APPEND QWEN3_TTS_GGML_DLLS ${_runtime_dlls})
            endif()
        endif()
    endforeach()
    list(REMOVE_DUPLICATES QWEN3_TTS_GGML_DLLS)
endif()

string(REPLACE ";" ", " _qwen3_tts_backends_display "${QWEN3_TTS_GGML_BACKENDS}")
message(STATUS "qwen3-tts GGML provider: ${_qwen3_tts_ggml_provider}")
if(GGML_BUILD_DIR)
    message(STATUS "qwen3-tts GGML build hint: ${GGML_BUILD_DIR}")
endif()
message(STATUS "qwen3-tts GGML backends: ${_qwen3_tts_backends_display}")

add_library(qwen3_tts_build_options INTERFACE)
target_compile_features(qwen3_tts_build_options INTERFACE cxx_std_17)
target_link_libraries(qwen3_tts_build_options INTERFACE Threads::Threads)

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
    target_compile_definitions(qwen3_tts_build_options INTERFACE _USE_MATH_DEFINES)
endif()

add_library(qwen3_tts_component_deps INTERFACE)
target_include_directories(qwen3_tts_component_deps INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(qwen3_tts_component_deps INTERFACE
    qwen3_tts_build_options
    qwen3_tts_ggml)

add_library(text_tokenizer STATIC
    src/text_tokenizer.cpp
    src/gguf_loader.cpp
)
target_link_libraries(text_tokenizer PUBLIC qwen3_tts_component_deps)

set(TTS_TRANSFORMER_SOURCES
    src/gguf_loader.cpp
    src/tts_transformer.cpp
)
if(APPLE AND QWEN3_TTS_COREML)
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor.mm)
    set_source_files_properties(src/coreml_code_predictor.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor_stub.cpp)
endif()

add_library(tts_transformer STATIC
    ${TTS_TRANSFORMER_SOURCES}
)
target_link_libraries(tts_transformer PUBLIC qwen3_tts_component_deps)
if(APPLE AND QWEN3_TTS_COREML)
    target_link_libraries(tts_transformer PUBLIC
        "-framework Foundation"
        "-framework CoreML"
    )
endif()

add_library(audio_tokenizer_encoder STATIC
    src/audio_tokenizer_encoder.cpp
    src/gguf_loader.cpp
)
target_link_libraries(audio_tokenizer_encoder PUBLIC qwen3_tts_component_deps)

add_library(audio_tokenizer_decoder STATIC
    src/audio_tokenizer_decoder.cpp
    src/gguf_loader.cpp
)
target_link_libraries(audio_tokenizer_decoder PUBLIC qwen3_tts_component_deps)

add_library(qwen3_tts STATIC
    src/qwen3_tts.cpp
)
target_include_directories(qwen3_tts PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(qwen3_tts PUBLIC
    qwen3_tts_build_options
    text_tokenizer
    tts_transformer
    audio_tokenizer_encoder
    audio_tokenizer_decoder
)
if(WIN32)
    target_link_libraries(qwen3_tts PUBLIC psapi)
endif()

add_library(qwen3tts_shared SHARED
    src/qwen3tts_c_api.cpp
)
set_target_properties(qwen3tts_shared PROPERTIES
    OUTPUT_NAME "qwen3tts"
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)
target_include_directories(qwen3tts_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(qwen3tts_shared PRIVATE
    qwen3_tts
)
qwen3_tts_finalize_binary(qwen3tts_shared)

add_executable(qwen3-tts-cli
    src/main.cpp
)
target_link_libraries(qwen3-tts-cli PRIVATE
    qwen3_tts
)
qwen3_tts_finalize_binary(qwen3-tts-cli)

add_executable(test_tokenizer
    tests/test_tokenizer.cpp
)
target_link_libraries(test_tokenizer PRIVATE
    text_tokenizer
)
qwen3_tts_finalize_binary(test_tokenizer)

add_executable(test_encoder
    tests/test_encoder.cpp
)
target_link_libraries(test_encoder PRIVATE
    audio_tokenizer_encoder
)
qwen3_tts_finalize_binary(test_encoder)

add_executable(test_transformer
    tests/test_transformer.cpp
)
target_link_libraries(test_transformer PRIVATE
    tts_transformer
)
qwen3_tts_finalize_binary(test_transformer)

add_executable(test_decoder
    tests/test_decoder.cpp
)
target_link_libraries(test_decoder PRIVATE
    audio_tokenizer_decoder
)
qwen3_tts_finalize_binary(test_decoder)

add_executable(test_chinese
    tests/test_chinese.cpp
)
target_link_libraries(test_chinese PRIVATE
    text_tokenizer
)
qwen3_tts_finalize_binary(test_chinese)

add_executable(test_tts_template_chinese
    tests/test_tts_template_chinese.cpp
)
target_link_libraries(test_tts_template_chinese PRIVATE
    text_tokenizer
)
qwen3_tts_finalize_binary(test_tts_template_chinese)

install(TARGETS text_tokenizer tts_transformer audio_tokenizer_encoder audio_tokenizer_decoder qwen3_tts qwen3tts_shared qwen3-tts-cli
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES
    src/gguf_loader.h
    src/text_tokenizer.h
    src/tts_transformer.h
    src/coreml_code_predictor.h
    src/audio_tokenizer_encoder.h
    src/audio_tokenizer_decoder.h
    src/qwen3_tts.h
    src/qwen3tts_c_api.h
    DESTINATION include
)

enable_testing()
add_test(NAME tokenizer_test
    COMMAND test_tokenizer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME encoder_test
    COMMAND test_encoder --tokenizer models/qwen3-tts-0.6B-base.gguf --audio clone.wav --reference reference/ref_audio_embedding.bin
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME transformer_test
    COMMAND test_transformer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME decoder_test
    COMMAND test_decoder
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME tts_template_chinese_test
    COMMAND test_tts_template_chinese models/qwen3-tts-0.6B-base.gguf
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
