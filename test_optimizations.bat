@echo off
mkdir output 2>nul

echo Test 1: Explicit --language none
build\qwen3-tts-cli.exe -m models\base_small -t "This is a test with no language explicitly set." -l none -o output\test_lang_none.wav --stats-json output\test_lang_none.json

echo Test 2: Voice Clone using ref_text from JSON
build\qwen3-tts-cli.exe -m models\base_small -t "Testing the voice cloning with reference text extracted from JSON." --mode clone -r ref\ref_1.7B.json -o output\test_clone_json.wav --stats-json output\test_clone_json.json

echo Test 3: Voice Clone using explicit --ref-text
build\qwen3-tts-cli.exe -m models\base_small -t "Testing voice cloning with explicitly provided reference text via CLI." --mode clone -r ref\ref.wav --ref-text "This is the reference text provided via CLI." -o output\test_clone_cli.wav --stats-json output\test_clone_cli.json

echo Test 4: Custom Voice without instruct
build\qwen3-tts-cli.exe -m models\custom_small -t "Testing custom voice generation without any instructions." --mode custom --speaker Vivian -o output\test_custom_no_instruct.wav --stats-json output\test_custom_no_instruct.json

echo All tests completed.
