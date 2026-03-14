#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

TEST_OUTPUT_DIR="$PROJECT_ROOT/test_output"
mkdir -p "$TEST_OUTPUT_DIR"

RESULTS_FILE="$PROJECT_ROOT/test_results.txt"

log() { echo -e "$1" | tee -a "$RESULTS_FILE"; }
pass() { log "${GREEN}[PASS]${NC} $1"; ((PASS_COUNT++)) || true; }
fail() { log "${RED}[FAIL]${NC} $1"; ((FAIL_COUNT++)) || true; }
skip() { log "${YELLOW}[SKIP]${NC} $1"; ((SKIP_COUNT++)) || true; }

file_size_bytes() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo ""
        return 0
    fi
    if stat -f%z "$path" >/dev/null 2>&1; then
        stat -f%z "$path"
    else
        stat -c%s "$path" 2>/dev/null
    fi
}

log_output_tail() {
    local output="$1"
    local lines="${2:-20}"
    if [[ -z "$output" ]]; then
        return 0
    fi
    log "  Last command output lines:"
    while IFS= read -r line; do
        log "    $line"
    done < <(echo "$output" | tail -n "$lines")
}

check_wav() {
    [[ -f "$1" ]] && [[ "$(head -c 4 "$1" 2>/dev/null)" == "RIFF" ]]
}

echo "========================================" > "$RESULTS_FILE"
echo "Qwen3-TTS-GGML Test Results" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

log "Starting comprehensive test suite..."
log ""

log "============================================"
log "SECTION 1: Component Tests"
log "============================================"
log ""

log "--- Test 1.1: Tokenizer ---"
if [[ -x "./build/test_tokenizer" ]]; then
    output=$(timeout 60 ./build/test_tokenizer --model models/qwen3-tts-0.6b-f16.gguf 2>&1)
    rc=$?
    if [[ $rc -eq 0 ]] && echo "$output" | grep -q "All tests passed"; then
        pass "Tokenizer test"
    else
        fail "Tokenizer test (exit code: $rc)"
        log_output_tail "$output"
    fi
else
    skip "Tokenizer test (binary not found)"
fi
log ""

log "--- Test 1.2: Encoder ---"
if [[ -x "./build/test_encoder" ]]; then
    if [[ ! -f "reference/debug/audio_resampled.bin" ]] && command -v python3 >/dev/null 2>&1; then
        mkdir -p reference/debug
        prep_output=$(python3 - <<'PY' 2>&1
import numpy as np
import soundfile as sf
from scipy import signal

audio, sr = sf.read("clone.wav")
if audio.ndim > 1:
    audio = audio.mean(axis=1)
if sr != 24000:
    audio = signal.resample(audio, int(len(audio) * 24000 / sr))
audio = np.asarray(audio, dtype=np.float32)
audio.tofile("reference/debug/audio_resampled.bin")
print(f"generated reference/debug/audio_resampled.bin ({audio.shape[0]} samples)")
PY
)
        prep_rc=$?
        if [[ $prep_rc -eq 0 ]]; then
            log "  Prepared Python-resampled audio for encoder comparison."
            log_output_tail "$prep_output" 2
        else
            log "  WARN: Could not prepare Python-resampled audio; encoder test will use fallback resampling."
            log_output_tail "$prep_output" 4
        fi
    fi

    output=$(timeout 120 ./build/test_encoder --tokenizer models/qwen3-tts-0.6b-f16.gguf --audio clone.wav --reference reference/ref_audio_embedding.bin 2>&1)
    rc=$?
    l2=$(echo "$output" | grep "L2 distance:" | head -1 | awk '{print $3}')
    if [[ $rc -eq 0 ]] && echo "$output" | grep -q "All tests passed"; then
        pass "Encoder test (L2 distance: $l2)"
    else
        fail "Encoder test (L2 distance: ${l2:-n/a}, exit code: $rc)"
        log_output_tail "$output"
    fi
else
    skip "Encoder test (binary not found)"
fi
log ""

log "--- Test 1.3: Transformer ---"
if [[ -x "./build/test_transformer" ]]; then
    output=$(timeout 180 ./build/test_transformer --model models/qwen3-tts-0.6b-f16.gguf --ref-dir reference/ 2>&1)
    rc=$?
    if [[ $rc -eq 0 ]] && echo "$output" | grep -q "All tests passed"; then
        pass "Transformer test (generates speech codes)"
    else
        fail "Transformer test (exit code: $rc)"
        log_output_tail "$output"
    fi
else
    skip "Transformer test (binary not found)"
fi
log ""

log "--- Test 1.4: Decoder ---"
if [[ -x "./build/test_decoder" ]]; then
    output=$(timeout 180 ./build/test_decoder --tokenizer models/qwen3-tts-tokenizer-f16.gguf --codes reference/speech_codes.bin --reference reference/decoded_audio.bin 2>&1)
    rc=$?
    if [[ $rc -eq 0 ]] && echo "$output" | grep -q "Decoded.*samples"; then
        samples=$(echo "$output" | grep "PASS: Decoded" | sed 's/.*Decoded \([0-9]*\) samples.*/\1/')
        pass "Decoder test (produces $samples samples)"
    else
        fail "Decoder test (exit code: $rc)"
        log_output_tail "$output"
    fi
else
    skip "Decoder test (binary not found)"
fi
log ""

log "============================================"
log "SECTION 2: CLI Tests with F16 Model"
log "============================================"
log ""

run_cli_test() {
    local name="$1"
    local output_file="$2"
    local output
    local rc
    shift 2
    
    log "--- $name ---"
    rm -f "$output_file"
    
    output=$(timeout 300 ./build/qwen3-tts-cli "$@" -o "$output_file" 2>&1)
    rc=$?
    if [[ $rc -eq 0 ]]; then
        if check_wav "$output_file"; then
            local size
            size=$(file_size_bytes "$output_file")
            pass "$name - WAV produced (${size} bytes)"
            return 0
        fi
    fi
    fail "$name (exit code: $rc)"
    log_output_tail "$output"
    return 1
}

if [[ -x "./build/qwen3-tts-cli" ]] && [[ -f "models/qwen3-tts-0.6b-f16.gguf" ]]; then
    run_cli_test "F16 basic synthesis" "$TEST_OUTPUT_DIR/test_f16_basic.wav" \
        -m models -t "Hello world" --max-tokens 100 || true
    log ""
    
    run_cli_test "F16 voice cloning" "$TEST_OUTPUT_DIR/test_f16_clone.wav" \
        -m models -t "Hello world" -r clone.wav --max-tokens 100 || true
    log ""
    
    run_cli_test "F16 longer text" "$TEST_OUTPUT_DIR/test_f16_long.wav" \
        -m models -t "This is a longer sentence to test synthesis." -r clone.wav --max-tokens 200 || true
    log ""
    
    run_cli_test "F16 temperature 0.5" "$TEST_OUTPUT_DIR/test_f16_temp.wav" \
        -m models -t "Testing temperature" -r clone.wav --temperature 0.5 --max-tokens 100 || true
    log ""
else
    skip "F16 CLI tests (CLI or model not found)"
fi

log "============================================"
log "SECTION 3: Q8_0 Model Verification"
log "============================================"
log ""

if [[ -f "models/qwen3-tts-0.6b-q8_0.gguf" ]]; then
    size=$(file_size_bytes "models/qwen3-tts-0.6b-q8_0.gguf")
    log "Q8_0 model file size: $size bytes"
    pass "Q8_0 model file exists"
else
    skip "Q8_0 model not found"
fi
log ""

log "============================================"
log "SECTION 4: Input Text Variations"
log "============================================"
log ""

if [[ -x "./build/qwen3-tts-cli" ]]; then
    run_cli_test "Short text (Hi)" "$TEST_OUTPUT_DIR/test_short.wav" \
        -m models -t "Hi" -r clone.wav --max-tokens 50 || true
    log ""
    
    run_cli_test "Punctuation text" "$TEST_OUTPUT_DIR/test_punct.wav" \
        -m models -t "Hello! How are you? I am fine, thank you." -r clone.wav --max-tokens 150 || true
    log ""
    
    run_cli_test "Numbers text" "$TEST_OUTPUT_DIR/test_numbers.wav" \
        -m models -t "The year is 2024 and the temperature is 72 degrees." -r clone.wav --max-tokens 150 || true
    log ""
else
    skip "Text variation tests (CLI not found)"
fi

log "============================================"
log "SECTION 5: Output File Validation"
log "============================================"
log ""

log "Generated WAV files:"
shopt -s nullglob
for wav in "$TEST_OUTPUT_DIR"/*.wav; do
    if [[ -f "$wav" ]]; then
        size=$(file_size_bytes "$wav")
        log "  $(basename "$wav"): $size bytes"
    fi
done
shopt -u nullglob
log ""

log "============================================"
log "SECTION 6: E2E Python vs C++ Comparison"
log "============================================"
log ""

log "--- Test 6.1: E2E Comparison ---"
if command -v uv &>/dev/null && [[ -f "scripts/compare_e2e.py" ]]; then
    dep_check=$(timeout 60 uv run python -c "import qwen_tts" 2>&1)
    dep_rc=$?
    if [[ $dep_rc -ne 0 ]]; then
        skip "E2E comparison (python dependency 'qwen_tts' not available in uv environment)"
        log_output_tail "$dep_check" 10
    else
        output=$(timeout 600 uv run python scripts/compare_e2e.py 2>&1)
        rc=$?
        if [[ $rc -eq 0 ]] && echo "$output" | grep -q "PASS"; then
            pass "E2E comparison test"
        else
            fail "E2E comparison test (exit code: $rc)"
            log_output_tail "$output"
        fi
    fi
else
    skip "E2E comparison (uv or script not found)"
fi
log ""

log "============================================"
log "TEST SUMMARY"
log "============================================"
log ""
log "Total PASS: $PASS_COUNT"
log "Total FAIL: $FAIL_COUNT"
log "Total SKIP: $SKIP_COUNT"
log ""

TOTAL=$((PASS_COUNT + FAIL_COUNT))
if [[ $TOTAL -gt 0 ]]; then
    PASS_RATE=$((PASS_COUNT * 100 / TOTAL))
    log "Pass Rate: ${PASS_RATE}% ($PASS_COUNT/$TOTAL)"
fi

log ""
log "Test artifacts saved to: $TEST_OUTPUT_DIR"
log "Full results saved to: $RESULTS_FILE"
log ""

if [[ $FAIL_COUNT -eq 0 ]]; then
    log "${GREEN}All tests passed!${NC}"
    exit 0
else
    log "${YELLOW}Some tests failed${NC}"
    exit 1
fi
