# CMake generated Testfile for 
# Source directory: D:/TTS/lunavox
# Build directory: D:/TTS/lunavox/build-cuda-timing
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(tokenizer_test "D:/TTS/lunavox/build-cuda-timing/test_tokenizer.exe" "--model" "models/qwen3-tts-0.6B-base.gguf")
set_tests_properties(tokenizer_test PROPERTIES  TIMEOUT "300" WORKING_DIRECTORY "D:/TTS/lunavox" _BACKTRACE_TRIPLES "D:/TTS/lunavox/CMakeLists.txt;487;add_test;D:/TTS/lunavox/CMakeLists.txt;0;")
add_test(encoder_test "D:/TTS/lunavox/build-cuda-timing/test_encoder.exe" "--tokenizer" "models/qwen3-tts-0.6B-base.gguf" "--audio" "reference/ref-audio.wav")
set_tests_properties(encoder_test PROPERTIES  TIMEOUT "300" WORKING_DIRECTORY "D:/TTS/lunavox" _BACKTRACE_TRIPLES "D:/TTS/lunavox/CMakeLists.txt;491;add_test;D:/TTS/lunavox/CMakeLists.txt;0;")
add_test(transformer_test "D:/TTS/lunavox/build-cuda-timing/test_transformer.exe" "--model" "models/qwen3-tts-0.6B-base.gguf" "--max-len" "2")
set_tests_properties(transformer_test PROPERTIES  TIMEOUT "300" WORKING_DIRECTORY "D:/TTS/lunavox" _BACKTRACE_TRIPLES "D:/TTS/lunavox/CMakeLists.txt;495;add_test;D:/TTS/lunavox/CMakeLists.txt;0;")
add_test(decoder_test "D:/TTS/lunavox/build-cuda-timing/test_decoder.exe" "--frames" "4")
set_tests_properties(decoder_test PROPERTIES  TIMEOUT "300" WORKING_DIRECTORY "D:/TTS/lunavox" _BACKTRACE_TRIPLES "D:/TTS/lunavox/CMakeLists.txt;499;add_test;D:/TTS/lunavox/CMakeLists.txt;0;")
add_test(tts_template_chinese_test "D:/TTS/lunavox/build-cuda-timing/test_tts_template_chinese.exe" "models/qwen3-tts-0.6B-base.gguf")
set_tests_properties(tts_template_chinese_test PROPERTIES  TIMEOUT "300" WORKING_DIRECTORY "D:/TTS/lunavox" _BACKTRACE_TRIPLES "D:/TTS/lunavox/CMakeLists.txt;503;add_test;D:/TTS/lunavox/CMakeLists.txt;0;")
add_test(cli_basic_smoke_test "D:/TTS/lunavox/build-cuda-timing/qwen3-tts-cli.exe" "-m" "models" "-t" "Hello from ctest." "-o" "D:/TTS/lunavox/build-cuda-timing/ctest_basic.wav" "--temperature" "0" "--max-tokens" "24")
set_tests_properties(cli_basic_smoke_test PROPERTIES  TIMEOUT "900" WORKING_DIRECTORY "D:/TTS/lunavox" _BACKTRACE_TRIPLES "D:/TTS/lunavox/CMakeLists.txt;507;add_test;D:/TTS/lunavox/CMakeLists.txt;0;")
add_test(cli_clone_smoke_test "D:/TTS/lunavox/build-cuda-timing/qwen3-tts-cli.exe" "-m" "models" "-t" "Hello from the reference voice." "-r" "reference/ref-audio.wav" "-o" "D:/TTS/lunavox/build-cuda-timing/ctest_clone.wav" "--temperature" "0" "--max-tokens" "24")
set_tests_properties(cli_clone_smoke_test PROPERTIES  TIMEOUT "900" WORKING_DIRECTORY "D:/TTS/lunavox" _BACKTRACE_TRIPLES "D:/TTS/lunavox/CMakeLists.txt;511;add_test;D:/TTS/lunavox/CMakeLists.txt;0;")
