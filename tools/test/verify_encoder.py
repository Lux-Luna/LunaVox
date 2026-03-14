#!/usr/bin/env python3
"""
Verify speaker encoder output against Python reference.

Usage:
    python tools/test/verify_encoder.py --audio clone.wav --reference reference/ref_audio_embedding.bin
"""

import argparse
import numpy as np
import torch
import soundfile as sf
from pathlib import Path


def load_audio(path: str, target_sr: int = 24000) -> np.ndarray:
    """Load audio file and resample if needed."""
    audio, sr = sf.read(path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if sr != target_sr:
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * target_sr / sr))
    
    return audio.astype(np.float32)


def extract_speaker_embedding_python(audio_path: str, model_path: str) -> np.ndarray:
    """Extract speaker embedding using Python model."""
    from qwen_tts import Qwen3TTSModel
    
    # Load model
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )
    
    # Create voice clone prompt to extract speaker embedding
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=audio_path,
        ref_text="",  # Not needed for speaker embedding
        x_vector_only_mode=True,
    )
    
    # Get speaker embedding
    spk_emb = prompt_items[0].ref_spk_embedding
    return spk_emb.float().numpy()


def compute_l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute L2 distance between two vectors."""
    return np.sqrt(np.sum((a - b) ** 2))


def main():
    parser = argparse.ArgumentParser(description="Verify speaker encoder output")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--reference", type=str, help="Path to reference embedding binary")
    parser.add_argument("--model", type=str, default="models/Qwen3-TTS-12Hz-0.6B-Base",
                        help="Path to TTS model")
    parser.add_argument("--output", type=str, help="Path to save extracted embedding")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Speaker Encoder Verification")
    print("=" * 60)
    
    # Extract embedding using Python
    print(f"\nExtracting speaker embedding from: {args.audio}")
    print(f"Using model: {args.model}")
    
    try:
        embedding = extract_speaker_embedding_python(args.audio, args.model)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding stats: min={embedding.min():.6f}, max={embedding.max():.6f}, mean={embedding.mean():.6f}")
    except Exception as e:
        print(f"ERROR: Failed to extract embedding: {e}")
        return 1
    
    # Save embedding if requested
    if args.output:
        embedding.tofile(args.output)
        print(f"Saved embedding to: {args.output}")
    
    # Compare with reference if provided
    if args.reference:
        print(f"\nComparing with reference: {args.reference}")
        ref_embedding = np.fromfile(args.reference, dtype=np.float32)
        print(f"Reference shape: {ref_embedding.shape}")
        
        if ref_embedding.shape != embedding.shape:
            print(f"ERROR: Shape mismatch!")
            return 1
        
        l2_dist = compute_l2_distance(embedding, ref_embedding)
        print(f"L2 distance: {l2_dist:.6f}")
        
        if l2_dist < 0.001:
            print("PASS: L2 distance < 0.001")
        elif l2_dist < 0.01:
            print("WARN: L2 distance < 0.01 (acceptable)")
        else:
            print(f"FAIL: L2 distance >= 0.01")
        
        # Print first few values
        print("\nFirst 5 values comparison:")
        for i in range(5):
            print(f"  [{i}] extracted={embedding[i]:.6f}, ref={ref_embedding[i]:.6f}, diff={embedding[i]-ref_embedding[i]:.6f}")
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
