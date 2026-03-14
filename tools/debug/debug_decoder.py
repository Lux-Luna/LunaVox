#!/usr/bin/env python3
"""
Debug script to save intermediate decoder outputs for comparison with C++.
"""

import torch
import numpy as np
from pathlib import Path
import sys

def save_tensor(tensor, path, name):
    """Save tensor as raw binary."""
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().float().numpy()
    else:
        arr = np.asarray(tensor, dtype=np.float32)
    arr.tofile(path)
    print(f"  Saved {name}: shape={arr.shape}, dtype={arr.dtype}, path={path}")
    print(f"    First 5 values: {arr.flatten()[:5]}")
    return arr

def main():
    from qwen_tts import Qwen3TTSModel
    
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "reference" / "decoder_debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    model_path = project_root / "models" / "Qwen3-TTS-12Hz-0.6B-Base"
    model = Qwen3TTSModel.from_pretrained(str(model_path), device_map='cpu', torch_dtype=torch.float32)
    
    # Get decoder components
    tokenizer_model = model.model.speech_tokenizer.model
    decoder = tokenizer_model.decoder
    
    print(f"\nDecoder structure:")
    print(f"  quantizer: {type(decoder.quantizer)}")
    print(f"  pre_conv: {type(decoder.pre_conv)}")
    print(f"  pre_transformer: {type(decoder.pre_transformer)}")
    print(f"  upsample: {len(decoder.upsample)} blocks")
    print(f"  decoder: {len(decoder.decoder)} blocks")
    
    # Load speech codes
    codes_path = project_root / "reference" / "speech_codes.bin"
    codes = np.fromfile(str(codes_path), dtype=np.int64).reshape(63, 16)
    codes_tensor = torch.from_numpy(codes).long().unsqueeze(0)  # [1, 63, 16]
    print(f"\nInput codes shape: {codes_tensor.shape}")
    
    # Step 1: VQ dequantization
    print("\n=== Step 1: VQ Dequantization ===")
    
    # The quantizer has rvq_first and rvq_rest
    quantizer = decoder.quantizer
    print(f"  rvq_first: {type(quantizer.rvq_first)}")
    print(f"  rvq_rest: {type(quantizer.rvq_rest)}")
    
    # Manually dequantize like the model does
    # codes shape: [B, T, n_codebooks] = [1, 63, 16]
    codes_input = codes_tensor.permute(0, 2, 1)  # [B, n_codebooks, T] = [1, 16, 63]
    
    # Split into first and rest
    first_codes = codes_input[:, 0:1, :]  # [1, 1, 63]
    rest_codes = codes_input[:, 1:, :]    # [1, 15, 63]
    
    print(f"  first_codes shape: {first_codes.shape}")
    print(f"  rest_codes shape: {rest_codes.shape}")
    
    # Dequantize first codebook - decode returns [B, C, T] = [1, 256, 63]
    first_emb = quantizer.rvq_first.vq.layers[0].decode(first_codes[:, 0, :])
    first_proj = quantizer.rvq_first.output_proj(first_emb)  # [1, 512, 63]
    
    print(f"  first_emb shape: {first_emb.shape}")
    print(f"  first_proj shape: {first_proj.shape}")
    save_tensor(first_emb[0], output_dir / "first_emb.bin", "first_emb")
    save_tensor(first_proj[0], output_dir / "first_proj.bin", "first_proj")
    
    # Dequantize rest codebooks - all in [B, C, T] format
    rest_sum = torch.zeros(1, 256, 63)
    for i in range(15):
        cb_emb = quantizer.rvq_rest.vq.layers[i].decode(rest_codes[:, i, :])  # [1, 256, 63]
        rest_sum = rest_sum + cb_emb
    
    rest_proj = quantizer.rvq_rest.output_proj(rest_sum)  # [1, 512, 63]
    
    print(f"  rest_sum shape: {rest_sum.shape}")
    print(f"  rest_proj shape: {rest_proj.shape}")
    save_tensor(rest_sum[0], output_dir / "rest_sum.bin", "rest_sum")
    save_tensor(rest_proj[0], output_dir / "rest_proj.bin", "rest_proj")
    
    # VQ output = first_proj + rest_proj, all in [B, C, T] = [1, 512, 63]
    vq_output = first_proj + rest_proj
    print(f"  vq_output shape: {vq_output.shape}")
    save_tensor(vq_output[0], output_dir / "vq_output.bin", "vq_output")
    
    # Step 2: Pre-conv
    print("\n=== Step 2: Pre-conv ===")
    
    # vq_output is already [B, C, T] = [1, 512, 63]
    pre_conv_output = decoder.pre_conv(vq_output)  # [1, 1024, 63]
    print(f"  pre_conv_output shape: {pre_conv_output.shape}")
    save_tensor(pre_conv_output[0], output_dir / "pre_conv_output.bin", "pre_conv_output")
    
    # Step 3: Pre-transformer
    print("\n=== Step 3: Pre-transformer ===")
    
    # pre_transformer expects inputs_embeds in [B, T, C] format
    pre_conv_btc = pre_conv_output.transpose(1, 2)  # [1, 63, 1024]
    print(f"  pre_conv_btc shape: {pre_conv_btc.shape}")
    pre_tfm_result = decoder.pre_transformer(inputs_embeds=pre_conv_btc)
    pre_tfm_output = pre_tfm_result.last_hidden_state.permute(0, 2, 1)  # [1, 1024, 63]
    print(f"  pre_tfm_output shape: {pre_tfm_output.shape}")
    save_tensor(pre_tfm_output[0], output_dir / "pre_tfm_output.bin", "pre_tfm_output")
    
    # Step 4: Upsample blocks (nested structure: list of lists)
    print("\n=== Step 4: Upsample blocks ===")
    
    cur = pre_tfm_output
    for i, upsample_blocks in enumerate(decoder.upsample):
        for j, block in enumerate(upsample_blocks):
            cur = block(cur)
            print(f"  After upsample[{i}][{j}]: shape={cur.shape}")
        save_tensor(cur[0], output_dir / f"upsample_{i}_output.bin", f"upsample_{i}_output")
    
    # Step 5: Decoder blocks
    print("\n=== Step 5: Decoder blocks ===")
    
    for i, dec_block in enumerate(decoder.decoder):
        cur = dec_block(cur)
        print(f"  After decoder[{i}]: shape={cur.shape}")
        save_tensor(cur[0], output_dir / f"decoder_{i}_output.bin", f"decoder_{i}_output")
    
    # Final output
    print("\n=== Final output ===")
    print(f"  Final shape: {cur.shape}")
    save_tensor(cur[0], output_dir / "final_output.bin", "final_output")
    
    # Also run the full decode to compare
    print("\n=== Full decode for comparison ===")
    with torch.no_grad():
        full_result = tokenizer_model.decode(codes_tensor)
        full_output = full_result.audio_values if hasattr(full_result, 'audio_values') else full_result
    print(f"  Full decode output shape: {full_output.shape}")
    save_tensor(full_output[0], output_dir / "full_decode_output.bin", "full_decode_output")
    
    print(f"\nAll outputs saved to {output_dir}")

if __name__ == "__main__":
    main()
