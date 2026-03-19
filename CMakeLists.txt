cmake_minimum_required(VERSION 3.14)
project(qwen3-tts VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QWEN3_TTS_TIMING "Enable detailed timing instrumentation" OFF)
option(QWEN3_TTS_USE_OPENMP "Enable OpenMP when it is available" ON)
option(QWEN3_TTS_NATIVE "Enable host-specific native optimizations for this top-level project" OFF)
option(QWEN3_TTS_COPY_RUNTIME
    "Copy runtime DLLs from prebuilt lib directory next to built qwen3-tts targets on Windows"
    ON)

set(QWEN3_TTS_PREBUILT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib" CACHE PATH
    "Directory containing prebuilt llama/ggml runtime binaries and import libraries")
set(QWEN3_TTS_GGML_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/include" CACHE PATH
    "Directory containing ggml headers (ggml.h/ggml-backend.h/gguf.h)")

if(NOT EXISTS "${QWEN3_TTS_PREBUILT_LIB_DIR}")
    message(FATAL_ERROR "Prebuilt runtime directory not found: ${QWEN3_TTS_PREBUILT_LIB_DIR}")
endif()
if(NOT EXISTS "${QWEN3_TTS_GGML_INCLUDE_DIR}/ggml.h")
    message(FATAL_ERROR "ggml headers not found in: ${QWEN3_TTS_GGML_INCLUDE_DIR}")
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
    find_library(GGML_BASE_LIBRARY
        NAMES ggml-base libggml-base
        HINTS "${QWEN3_TTS_PREBUILT_LIB_DIR}"
        NO_DEFAULT_PATH)
    find_library(GGML_LIBRARY
        NAMES ggml libggml
        HINTS "${QWEN3_TTS_PREBUILT_LIB_DIR}"
        NO_DEFAULT_PATH)
else()
    find_library(GGML_BASE_LIBRARY
        NAMES ggml-base
        HINTS "${QWEN3_TTS_PREBUILT_LIB_DIR}"
        NO_DEFAULT_PATH)
    find_library(GGML_LIBRARY
        NAMES ggml
        HINTS "${QWEN3_TTS_PREBUILT_LIB_DIR}"
        NO_DEFAULT_PATH)
endif()

if(NOT GGML_BASE_LIBRARY OR NOT GGML_LIBRARY)
    message(FATAL_ERROR
        "Failed to locate GGML import libraries in ${QWEN3_TTS_PREBUILT_LIB_DIR}.\n"
        "Expected libggml-base and libggml import libs for your toolchain.\n"
        "Use `python manage.py build` (or tools/build_manager.py) to generate missing import libs from DLLs.")
endif()

add_library(ggml::ggml-base UNKNOWN IMPORTED)
set_target_properties(ggml::ggml-base PROPERTIES
    IMPORTED_LOCATION "${GGML_BASE_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${QWEN3_TTS_GGML_INCLUDE_DIR}")

add_library(ggml::ggml UNKNOWN IMPORTED)
set_target_properties(ggml::ggml PROPERTIES
    IMPORTED_LOCATION "${GGML_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${QWEN3_TTS_GGML_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "ggml::ggml-base")

set(QWEN3_TTS_RUNTIME_DIRS "${QWEN3_TTS_PREBUILT_LIB_DIR}")
set(QWEN3_TTS_RUNTIME_DLLS "")
if(WIN32)
    file(GLOB _qwen3_tts_runtime_dlls
        "${QWEN3_TTS_PREBUILT_LIB_DIR}/ggml*.dll"
        "${QWEN3_TTS_PREBUILT_LIB_DIR}/llama*.dll"
        "${QWEN3_TTS_PREBUILT_LIB_DIR}/libomp*.dll")
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
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${QWEN3_TTS_GGML_INCLUDE_DIR})
target_link_libraries(qwen3_tts_component_deps INTERFACE
    qwen3_tts_build_options
    ggml::ggml)

add_library(text_tokenizer STATIC
    src/text_tokenizer.cpp
    src/gguf_loader.cpp
)
target_link_libraries(text_tokenizer PUBLIC qwen3_tts_component_deps)

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

add_library(talker_predictor_llama STATIC
    src/llama_wrapper.cpp
    src/assets_manager.cpp
    src/talker_predictor_llama.cpp
)
target_link_libraries(talker_predictor_llama PUBLIC qwen3_tts_build_options)

add_library(qwen3_tts STATIC
    src/qwen3_tts.cpp
)
target_include_directories(qwen3_tts PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(qwen3_tts PUBLIC
    qwen3_tts_build_options
    text_tokenizer
    talker_predictor_llama
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

add_executable(test_decoder
    tests/test_decoder.cpp
)
target_link_libraries(test_decoder PRIVATE
    audio_tokenizer_decoder
)
qwen3_tts_finalize_binary(test_decoder)

add_executable(test_tts_template_chinese
    tests/test_tts_template_chinese.cpp
)
target_link_libraries(test_tts_template_chinese PRIVATE
    text_tokenizer
)
qwen3_tts_finalize_binary(test_tts_template_chinese)

install(TARGETS
    text_tokenizer
    audio_tokenizer_encoder
    audio_tokenizer_decoder
    qwen3_tts
    qwen3tts_shared
    qwen3-tts-cli
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES
    src/gguf_loader.h
    src/text_tokenizer.h
    src/llama_wrapper.h
    src/assets_manager.h
    src/talker_predictor_llama.h
    src/audio_tokenizer_encoder.h
    src/audio_tokenizer_decoder.h
    src/qwen3_tts.h
    src/qwen3tts_c_api.h
    DESTINATION include
)

enable_testing()
add_test(NAME tokenizer_test
    COMMAND test_tokenizer --model models/qwen3_tts_speaker_encoder.gguf
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME encoder_test
    COMMAND test_encoder --tokenizer models/qwen3_tts_speaker_encoder.gguf --audio ${CMAKE_CURRENT_BINARY_DIR}/ctest_basic.wav
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME decoder_test
    COMMAND test_decoder --tokenizer models/qwen3_tts_codec_decoder.gguf --frames 4
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME tts_template_chinese_test
    COMMAND test_tts_template_chinese models/qwen3_tts_speaker_encoder.gguf
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME cli_basic_smoke_test
    COMMAND qwen3-tts-cli -m models -t "Hello from ctest." -o ${CMAKE_CURRENT_BINARY_DIR}/ctest_basic.wav --temperature 0 --max-tokens 24
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME cli_clone_smoke_test
    COMMAND qwen3-tts-cli -m models -t "Hello from the reference voice." -r ${CMAKE_CURRENT_BINARY_DIR}/ctest_basic.wav -o ${CMAKE_CURRENT_BINARY_DIR}/ctest_clone.wav --temperature 0 --max-tokens 24
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(encoder_test PROPERTIES DEPENDS cli_basic_smoke_test)
set_tests_properties(cli_clone_smoke_test PROPERTIES DEPENDS cli_basic_smoke_test)

set_tests_properties(
    tokenizer_test
    encoder_test
    decoder_test
    tts_template_chinese_test
    PROPERTIES TIMEOUT 300
)
set_tests_properties(
    cli_basic_smoke_test
    cli_clone_smoke_test
    PROPERTIES TIMEOUT 900
)
