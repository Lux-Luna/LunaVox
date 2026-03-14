#!/usr/bin/env python3
"""
Generate reference outputs for Qwen3-TTS GGML conversion.

This script runs the full TTS pipeline and saves intermediate outputs at each stage:
1. Text tokenization
2. Reference audio embedding (speaker embedding)
3. Speech codes from transformer
4. Decoded audio from vocoder

Usage:
    /root/.local/bin/uv run python scripts/generate_reference_outputs.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def save_tensor_as_bin(tensor: torch.Tensor, path: str, name: str) -> dict:
    """Save tensor as raw binary and return metadata."""
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    
    # Save raw bytes
    arr.tofile(path)
    
    return {
        "name": name,
        "path": os.path.basename(path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size_bytes": arr.nbytes,
    }


def main():
    print("=" * 60)
    print("Qwen3-TTS Reference Output Generator")
    print("=" * 60)
    
    # Configuration
    model_path = PROJECT_ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-Base"
    ref_audio_path = PROJECT_ROOT / "clone.wav"
    ref_text_path = PROJECT_ROOT / "reference_text.txt"
    output_dir = PROJECT_ROOT / "reference"
    
    # Text to synthesize (keep very short for faster generation on CPU)
    synth_text = "Hello."
    
    # Validate inputs
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Trying HuggingFace model ID instead...")
        model_path = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    
    if not ref_audio_path.exists():
        print(f"ERROR: Reference audio not found at {ref_audio_path}")
        sys.exit(1)
    
    if not ref_text_path.exists():
        print(f"ERROR: Reference text not found at {ref_text_path}")
        sys.exit(1)
    
    # Read reference text
    with open(ref_text_path, "r") as f:
        ref_text = f.read().strip()
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print(f"  Reference audio: {ref_audio_path}")
    print(f"  Reference text: {ref_text[:50]}...")
    print(f"  Synthesis text: {synth_text}")
    print(f"  Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Metadata to collect
    metadata = {
        "model_path": str(model_path),
        "ref_audio_path": str(ref_audio_path),
        "ref_text": ref_text,
        "synth_text": synth_text,
        "outputs": [],
    }
    
    # Import qwen_tts
    print("\n[1/6] Loading Qwen3-TTS model...")
    from qwen_tts import Qwen3TTSModel
    
    # Load model with bfloat16 for memory efficiency
    model = Qwen3TTSModel.from_pretrained(
        str(model_path),
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )
    
    print(f"  Model loaded on device: {model.device}")
    print(f"  Model type: {model.model.tts_model_type}")
    print(f"  Tokenizer type: {model.model.tokenizer_type}")
    
    # Store model info
    metadata["model_info"] = {
        "tts_model_type": model.model.tts_model_type,
        "tokenizer_type": model.model.tokenizer_type,
        "device": str(model.device),
        "dtype": "bfloat16",
    }
    
    # Step 2: Tokenize text
    print("\n[2/6] Tokenizing text...")
    
    # Build the text in the format the model expects
    assistant_text = model._build_assistant_text(synth_text)
    ref_text_formatted = model._build_ref_text(ref_text)
    
    # Tokenize
    input_ids = model._tokenize_texts([assistant_text])[0]
    ref_ids = model._tokenize_texts([ref_text_formatted])[0]
    
    print(f"  Input text tokens shape: {input_ids.shape}")
    print(f"  Reference text tokens shape: {ref_ids.shape}")
    
    # Save text tokens
    text_tokens_path = output_dir / "text_tokens.bin"
    meta = save_tensor_as_bin(input_ids, str(text_tokens_path), "text_tokens")
    metadata["outputs"].append(meta)
    print(f"  Saved: {text_tokens_path}")
    
    # Also save ref tokens
    ref_tokens_path = output_dir / "ref_text_tokens.bin"
    meta = save_tensor_as_bin(ref_ids, str(ref_tokens_path), "ref_text_tokens")
    metadata["outputs"].append(meta)
    print(f"  Saved: {ref_tokens_path}")
    
    # Step 3: Create voice clone prompt (extracts speaker embedding and ref codes)
    print("\n[3/6] Extracting reference audio embedding...")
    
    # Use x_vector_only_mode=True for faster generation (no ICL)
    # This uses only speaker embedding, not reference codes
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=str(ref_audio_path),
        ref_text=ref_text,
        x_vector_only_mode=True,  # Faster mode - speaker embedding only
    )
    
    prompt_item = prompt_items[0]
    
    # Save speaker embedding
    spk_emb = prompt_item.ref_spk_embedding
    print(f"  Speaker embedding shape: {spk_emb.shape}")
    print(f"  Speaker embedding dtype: {spk_emb.dtype}")
    
    spk_emb_path = output_dir / "ref_audio_embedding.bin"
    meta = save_tensor_as_bin(spk_emb.float(), str(spk_emb_path), "ref_audio_embedding")
    metadata["outputs"].append(meta)
    print(f"  Saved: {spk_emb_path}")
    
    # Save reference codes (from speech tokenizer) - only if available
    ref_code = prompt_item.ref_code
    if ref_code is not None:
        print(f"  Reference codes shape: {ref_code.shape}")
        print(f"  Reference codes dtype: {ref_code.dtype}")
        
        ref_code_path = output_dir / "ref_speech_codes.bin"
        meta = save_tensor_as_bin(ref_code, str(ref_code_path), "ref_speech_codes")
        metadata["outputs"].append(meta)
        print(f"  Saved: {ref_code_path}")
    else:
        print("  Reference codes: None (x_vector_only_mode=True)")
    
    # Step 4: Generate speech codes
    print("\n[4/6] Generating speech codes...")
    
    # We need to hook into the generate process to capture intermediate outputs
    # The generate_voice_clone method returns wavs, but we need the codes
    
    # Build voice clone prompt dict
    voice_clone_prompt_dict = model._prompt_items_to_voice_clone_prompt(prompt_items)
    
    # Prepare inputs like generate_voice_clone does
    texts = [synth_text]
    languages = ["English"]
    
    input_texts = [model._build_assistant_text(t) for t in texts]
    input_ids_list = model._tokenize_texts(input_texts)
    
    # ref_ids only needed for ICL mode
    ref_ids_list = None
    if prompt_item.ref_text and not prompt_item.x_vector_only_mode:
        ref_ids_list = [model._tokenize_texts([model._build_ref_text(prompt_item.ref_text)])[0]]
    
    gen_kwargs = model._merge_generate_kwargs(
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.05,
        max_new_tokens=64,  # Very short for faster CPU generation
    )
    
    import time
    print("  Starting generation (this may take a while on CPU)...")
    start_time = time.time()
    
    # Call the underlying model's generate
    talker_codes_list, talker_hidden_states_list = model.model.generate(
        input_ids=input_ids_list,
        ref_ids=ref_ids_list,
        voice_clone_prompt=voice_clone_prompt_dict,
        languages=languages,
        non_streaming_mode=False,
        **gen_kwargs,
    )
    
    # Get the generated codes
    elapsed = time.time() - start_time
    print(f"  Generation completed in {elapsed:.1f}s")
    
    speech_codes = talker_codes_list[0]
    print(f"  Generated speech codes shape: {speech_codes.shape}")
    print(f"  Generated speech codes dtype: {speech_codes.dtype}")
    
    speech_codes_path = output_dir / "speech_codes.bin"
    meta = save_tensor_as_bin(speech_codes, str(speech_codes_path), "speech_codes")
    metadata["outputs"].append(meta)
    print(f"  Saved: {speech_codes_path}")
    
    # Also save hidden states
    hidden_states = talker_hidden_states_list[0]
    print(f"  Hidden states shape: {hidden_states.shape}")
    
    hidden_states_path = output_dir / "hidden_states.bin"
    meta = save_tensor_as_bin(hidden_states.float(), str(hidden_states_path), "hidden_states")
    metadata["outputs"].append(meta)
    print(f"  Saved: {hidden_states_path}")
    
    # Step 5: Decode to audio
    print("\n[5/6] Decoding speech codes to audio...")
    
    # Concatenate ref codes with generated codes for decoding (as done in generate_voice_clone)
    ref_code_for_decode = voice_clone_prompt_dict["ref_code"][0]
    
    if ref_code_for_decode is not None:
        codes_for_decode = torch.cat([ref_code_for_decode.to(speech_codes.device), speech_codes], dim=0)
        ref_len = ref_code_for_decode.shape[0]
    else:
        codes_for_decode = speech_codes
        ref_len = 0
    
    print(f"  Codes for decode shape: {codes_for_decode.shape}")
    
    # Decode using speech tokenizer
    wavs_all, sample_rate = model.model.speech_tokenizer.decode([{"audio_codes": codes_for_decode}])
    
    # Cut off the reference audio portion if present
    total_len = codes_for_decode.shape[0]
    cut = int(ref_len / max(total_len, 1) * wavs_all[0].shape[0])
    decoded_audio = wavs_all[0][cut:]
    
    print(f"  Decoded audio shape: {decoded_audio.shape}")
    print(f"  Sample rate: {sample_rate}")
    print(f"  Duration: {len(decoded_audio) / sample_rate:.2f}s")
    
    # Save decoded audio as raw float32
    decoded_audio_path = output_dir / "decoded_audio.bin"
    meta = save_tensor_as_bin(decoded_audio, str(decoded_audio_path), "decoded_audio")
    meta["sample_rate"] = sample_rate
    meta["duration_seconds"] = len(decoded_audio) / sample_rate
    metadata["outputs"].append(meta)
    print(f"  Saved: {decoded_audio_path}")
    
    # Step 6: Save as WAV file
    print("\n[6/6] Saving output WAV file...")
    
    output_wav_path = output_dir / "output.wav"
    sf.write(str(output_wav_path), decoded_audio, sample_rate)
    
    metadata["outputs"].append({
        "name": "output_wav",
        "path": "output.wav",
        "sample_rate": sample_rate,
        "duration_seconds": len(decoded_audio) / sample_rate,
        "format": "WAV PCM float32",
    })
    print(f"  Saved: {output_wav_path}")
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata saved: {metadata_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Reference outputs generated successfully!")
    print("=" * 60)
    print(f"\nOutput files in {output_dir}:")
    for item in metadata["outputs"]:
        if "shape" in item:
            print(f"  - {item['path']}: {item['shape']} ({item['dtype']})")
        else:
            print(f"  - {item['path']}: {item.get('format', 'audio')}")
    
    # Verify output audio
    print(f"\nOutput audio duration: {len(decoded_audio) / sample_rate:.2f}s")
    if len(decoded_audio) / sample_rate < 0.5:
        print("WARNING: Output audio is very short!")
        return 1
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
