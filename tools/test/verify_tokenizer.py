#!/usr/bin/env python3
"""
Verify tokenizer output against Python reference implementation.

Usage:
    python tools/test/verify_tokenizer.py --model models/qwen3-tts-0.6B-base.gguf
"""

import argparse
import subprocess
import sys
import os

def get_python_tokens(text: str, model_path: str) -> list[int]:
    """Get tokens using Python transformers tokenizer."""
    from transformers import AutoTokenizer
    
    # Find the HuggingFace model directory
    model_dir = os.path.dirname(model_path)
    hf_model_dir = os.path.join(model_dir, "Qwen3-TTS-12Hz-0.6B-Base")
    
    if not os.path.exists(hf_model_dir):
        print(f"Warning: HuggingFace model not found at {hf_model_dir}")
        print("Using default Qwen tokenizer...")
        hf_model_dir = model_path
    
    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, trust_remote_code=True)
    
    # Format for TTS: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    formatted = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    tokens = tokenizer.encode(formatted, add_special_tokens=False)
    
    return tokens

def get_cpp_tokens(text: str, model_path: str, test_binary: str) -> list[int]:
    """Get tokens using C++ tokenizer (via test binary output)."""
    # Run the test binary and parse output
    result = subprocess.run(
        [test_binary, "--model", model_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"C++ test failed:\n{result.stderr}")
        return []
    
    # Parse tokens from output
    # Look for line like "  Tokens: [151644, 77091, ...]"
    for line in result.stdout.split('\n'):
        if 'TTS tokens' in line or 'Tokens:' in line:
            # Extract tokens from brackets
            start = line.find('[')
            end = line.find(']')
            if start >= 0 and end > start:
                tokens_str = line[start+1:end]
                tokens = [int(t.strip()) for t in tokens_str.split(',') if t.strip()]
                return tokens
    
    return []

def main():
    parser = argparse.ArgumentParser(description="Verify tokenizer output")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--text", default="Hello.", help="Text to tokenize")
    parser.add_argument("--test-binary", default="build/test_tokenizer", 
                        help="Path to C++ test binary")
    args = parser.parse_args()
    
    print(f"Text: '{args.text}'")
    print(f"Model: {args.model}")
    print()
    
    # Get Python reference tokens
    print("Getting Python reference tokens...")
    try:
        py_tokens = get_python_tokens(args.text, args.model)
        print(f"Python tokens: {py_tokens}")
    except Exception as e:
        print(f"Error getting Python tokens: {e}")
        py_tokens = []
    
    # Get C++ tokens
    print("\nGetting C++ tokens...")
    if os.path.exists(args.test_binary):
        cpp_tokens = get_cpp_tokens(args.text, args.model, args.test_binary)
        print(f"C++ tokens: {cpp_tokens}")
    else:
        print(f"Test binary not found: {args.test_binary}")
        cpp_tokens = []
    
    # Compare
    print("\n=== Comparison ===")
    if py_tokens and cpp_tokens:
        if py_tokens == cpp_tokens:
            print("PASS: Tokens match!")
            return 0
        else:
            print("FAIL: Tokens differ!")
            print(f"  Python: {py_tokens}")
            print(f"  C++:    {cpp_tokens}")
            
            # Show differences
            for i, (p, c) in enumerate(zip(py_tokens, cpp_tokens)):
                if p != c:
                    print(f"  Difference at index {i}: Python={p}, C++={c}")
            
            if len(py_tokens) != len(cpp_tokens):
                print(f"  Length difference: Python={len(py_tokens)}, C++={len(cpp_tokens)}")
            
            return 1
    else:
        print("Could not compare (missing tokens)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
