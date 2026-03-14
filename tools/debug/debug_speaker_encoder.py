#!/usr/bin/env python3
"""Debug script to save intermediate outputs from the speaker encoder."""

import os
import sys
import numpy as np
import torch
import soundfile as sf
from scipy import signal
from pathlib import Path

# Add qwen_tts to path
sys.path.insert(0, "/root/.venv/lib/python3.11/site-packages")

from librosa.filters import mel as librosa_mel_fn
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSSpeakerEncoder,
    mel_spectrogram,
    dynamic_range_compression_torch,
)
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig

def save_tensor(tensor: torch.Tensor, path: str, name: str):
    """Save tensor as binary file."""
    data = tensor.detach().cpu().float().numpy()
    filepath = os.path.join(path, f"{name}.bin")
    data.tofile(filepath)
    print(f"Saved {name}: shape={data.shape}, dtype={data.dtype}, "
          f"min={data.min():.6f}, max={data.max():.6f}, mean={data.mean():.6f}")
    # Also print first few values
    flat = data.flatten()
    print(f"  First 10 values: {flat[:10]}")
    return data

def main():
    # Paths
    audio_path = "/root/qwen-3-tts-ggml/clone.wav"
    model_path = "/root/qwen-3-tts-ggml/models/Qwen3-TTS-12Hz-0.6B-Base"
    output_dir = "/root/qwen-3-tts-ggml/reference/debug"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio
    print(f"Loading audio from {audio_path}")
    waveform_np, sample_rate = sf.read(audio_path)
    print(f"Original: sample_rate={sample_rate}, shape={waveform_np.shape}")
    
    # Convert to mono if stereo
    if len(waveform_np.shape) > 1:
        waveform_np = waveform_np.mean(axis=1)
    
    # Resample to 24kHz if needed
    target_sr = 24000
    if sample_rate != target_sr:
        print(f"Resampling from {sample_rate} to {target_sr}")
        num_samples = int(len(waveform_np) * target_sr / sample_rate)
        waveform_np = signal.resample(waveform_np, num_samples)
    
    # Convert to torch tensor [1, samples]
    waveform = torch.from_numpy(waveform_np.astype(np.float32)).unsqueeze(0)
    
    print(f"After preprocessing: shape={waveform.shape}")
    
    # Save resampled audio
    save_tensor(waveform, output_dir, "audio_resampled")
    
    # Mel spectrogram parameters - MUST match extract_speaker_embedding() in modeling_qwen3_tts.py
    n_fft = 1024
    num_mels = 128
    hop_size = 256
    win_size = 1024
    fmin = 0
    fmax = 12000
    
    # Compute mel spectrogram step by step
    print("\n=== Mel Spectrogram Computation ===")
    
    device = waveform.device
    y = waveform
    
    # Step 1: Mel filterbank
    mel = librosa_mel_fn(sr=target_sr, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel).float().to(device)
    save_tensor(mel_basis, output_dir, "mel_filterbank")
    
    # Step 2: Hann window
    hann_window = torch.hann_window(win_size).to(device)
    save_tensor(hann_window, output_dir, "hann_window")
    
    # Step 3: Padding
    padding = (n_fft - hop_size) // 2
    print(f"Padding: {padding}")
    y_padded = torch.nn.functional.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
    save_tensor(y_padded, output_dir, "audio_padded")
    
    # Step 4: STFT
    spec = torch.stft(
        y_padded,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    print(f"STFT output shape: {spec.shape}")
    
    # Save real and imag parts
    spec_real = spec.real
    spec_imag = spec.imag
    save_tensor(spec_real, output_dir, "stft_real")
    save_tensor(spec_imag, output_dir, "stft_imag")
    
    # Step 5: Magnitude
    spec_mag = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    save_tensor(spec_mag, output_dir, "stft_magnitude")
    
    # Step 6: Mel spectrogram
    mel_spec = torch.matmul(mel_basis, spec_mag)
    save_tensor(mel_spec, output_dir, "mel_spec_linear")
    
    # Step 7: Log compression
    mel_spec_log = dynamic_range_compression_torch(mel_spec)
    save_tensor(mel_spec_log, output_dir, "mel_spec_log")
    
    print(f"\nMel spectrogram final shape: {mel_spec_log.shape}")
    
    # Now run through the speaker encoder
    print("\n=== Speaker Encoder ===")
    
    # Load the model using qwen_tts package
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    print("Loading Qwen3-TTS model...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(model_path)
    speaker_encoder = model.speaker_encoder
    speaker_encoder.eval()
    
    # Print model structure
    print("\nSpeaker encoder structure:")
    for name, module in speaker_encoder.named_modules():
        if name:
            print(f"  {name}: {type(module).__name__}")
    
    # Run forward pass with hooks to capture intermediate outputs
    intermediates = {}
    
    def make_hook(name):
        def hook(module, input, output):
            intermediates[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    
    # Hook for each block
    for i, block in enumerate(speaker_encoder.blocks):
        hooks.append(block.register_forward_hook(make_hook(f"block_{i}")))
        if hasattr(block, 'tdnn1'):
            hooks.append(block.tdnn1.register_forward_hook(make_hook(f"block_{i}_tdnn1")))
        if hasattr(block, 'res2net_block'):
            hooks.append(block.res2net_block.register_forward_hook(make_hook(f"block_{i}_res2net")))
        if hasattr(block, 'tdnn2'):
            hooks.append(block.tdnn2.register_forward_hook(make_hook(f"block_{i}_tdnn2")))
        if hasattr(block, 'se_block'):
            hooks.append(block.se_block.register_forward_hook(make_hook(f"block_{i}_se")))
    
    # MFA
    hooks.append(speaker_encoder.mfa.register_forward_hook(make_hook("mfa")))
    
    # ASP
    hooks.append(speaker_encoder.asp.register_forward_hook(make_hook("asp")))
    hooks.append(speaker_encoder.asp.tdnn.register_forward_hook(make_hook("asp_tdnn")))
    hooks.append(speaker_encoder.asp.conv.register_forward_hook(make_hook("asp_conv")))
    
    # FC
    hooks.append(speaker_encoder.fc.register_forward_hook(make_hook("fc")))
    
    # Run forward pass
    # Input should be [batch, time, mel_dim]
    mel_input = mel_spec_log.transpose(1, 2)  # [1, time, 128]
    print(f"\nInput to speaker encoder: {mel_input.shape}")
    save_tensor(mel_input, output_dir, "encoder_input")
    
    with torch.no_grad():
        embedding = speaker_encoder(mel_input)
    
    print(f"\nFinal embedding: {embedding.shape}")
    save_tensor(embedding, output_dir, "embedding")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Save all intermediates
    print("\n=== Intermediate Outputs ===")
    for name, tensor in sorted(intermediates.items()):
        save_tensor(tensor, output_dir, name)
    
    # Also save the input to MFA (concatenated block outputs)
    # This is torch.cat(hidden_states_list[1:], dim=1)
    if "block_1" in intermediates and "block_2" in intermediates and "block_3" in intermediates:
        mfa_input = torch.cat([intermediates["block_1"], intermediates["block_2"], intermediates["block_3"]], dim=1)
        save_tensor(mfa_input, output_dir, "mfa_input")
    
    # Compare with reference
    ref_path = "/root/qwen-3-tts-ggml/reference/ref_audio_embedding.bin"
    if os.path.exists(ref_path):
        ref_emb = np.fromfile(ref_path, dtype=np.float32)
        our_emb = embedding.detach().cpu().float().numpy().flatten()
        l2_dist = np.sqrt(np.sum((ref_emb - our_emb) ** 2))
        print(f"\nL2 distance to reference: {l2_dist:.6f}")
    
    print("\nDone! All intermediate outputs saved to:", output_dir)

if __name__ == "__main__":
    main()
