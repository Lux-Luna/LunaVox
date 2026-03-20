#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CodecEncoderExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.config = model.config
        self.config.return_dict = False
        self.config.use_cache = False

        self.mimi_model = model.encoder
        self.cnn_encoder = self.mimi_model.encoder
        self.transformer = self.mimi_model.encoder_transformer
        self.downsample = self.mimi_model.downsample
        self.quantizer = self.mimi_model.quantizer

        self.eval()
        for m in self.modules():
            if hasattr(m, "config"):
                m.config.return_dict = False
                m.config.use_cache = False

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        x = input_values.unsqueeze(1)

        cnn_out = self.cnn_encoder(x)
        if isinstance(cnn_out, (tuple, list)):
            hidden_states = cnn_out[0]
        else:
            hidden_states = cnn_out

        hidden_states = hidden_states.transpose(1, 2)
        trans_out = self.transformer(hidden_states, return_dict=False)
        if isinstance(trans_out, (tuple, list)):
            hidden_states = trans_out[0]
        else:
            hidden_states = trans_out

        hidden_states = hidden_states.transpose(1, 2)
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        codes = self.quantizer.encode(hidden_states)
        if isinstance(codes, (tuple, list)):
            codes = codes[0]
        elif hasattr(codes, "audio_codes"):
            codes = codes.audio_codes

        valid_q = self.config.encoder_valid_num_quantizers
        codes = codes[:valid_q, :, :]
        return codes.transpose(0, 1).transpose(1, 2)


class SpeakerEncoderExportWrapper(nn.Module):
    def __init__(self, speaker_encoder: nn.Module):
        super().__init__()
        self.speaker_encoder = speaker_encoder
        self.speaker_encoder.eval()

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        return self.speaker_encoder(mels)


class TraceableKVStack:
    def __init__(self, keys, values, window_size: int):
        self.key_cache = keys
        self.value_cache = values
        self.window_size = window_size

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k_combined = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
        v_combined = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        self.key_cache[layer_idx] = k_combined[:, :, -self.window_size :, :]
        self.value_cache[layer_idx] = v_combined[:, :, -self.window_size :, :]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0):
        return self.key_cache[layer_idx].size(2)

    def __len__(self):
        return len(self.key_cache)


class DecoderPart1PreConv(nn.Module):
    def __init__(self, decoder_model):
        super().__init__()
        self.quantizer = decoder_model.quantizer
        self.pre_conv = decoder_model.pre_conv
        self.pre_conv_history_window = 2

    def forward(self, audio_codes: torch.Tensor, pre_conv_history: torch.Tensor):
        codes = audio_codes.transpose(1, 2)
        quantized = self.quantizer.decode(codes)

        quant_full = torch.cat([pre_conv_history, quantized], dim=-1)
        hidden_all = self.pre_conv(quant_full)

        hist_len = pre_conv_history.size(2)
        hidden = hidden_all[:, :, hist_len:]
        hidden = hidden.transpose(1, 2)
        next_pre_conv_hist = quant_full[:, :, -self.pre_conv_history_window :]
        return hidden, next_pre_conv_hist


class DecoderPart2Transformer(nn.Module):
    def __init__(self, decoder_model):
        super().__init__()
        self.trans = decoder_model.pre_transformer
        self.num_layers = self.trans.config.num_hidden_layers
        self.window_size = self.trans.config.sliding_window

    def forward(self, hidden, *past_kv_flat):
        bsz, n_tokens, _ = hidden.shape
        device = hidden.device

        keys_in = list(past_kv_flat[: self.num_layers])
        values_in = list(past_kv_flat[self.num_layers :])
        kv_stack = TraceableKVStack(keys_in, values_in, self.window_size)

        past_len = kv_stack.get_seq_length()
        total_len = past_len + n_tokens

        h = self.trans.input_proj(hidden)
        position_ids = torch.arange(past_len, total_len, device=device).unsqueeze(0)
        pos_embeddings = self.trans.rotary_emb(h, position_ids)

        q_idx = torch.arange(n_tokens, device=device).unsqueeze(1)
        full_k_idx = torch.arange(total_len, device=device).unsqueeze(0)
        k_idx = full_k_idx[:, -self.window_size :]
        mask_cond = (k_idx <= (past_len + q_idx)) & (k_idx > (past_len + q_idx - self.window_size))
        attn_mask = torch.where(mask_cond, 0.0, -10000.0).unsqueeze(0).unsqueeze(0)

        for layer in self.trans.layers:
            h = layer(
                h,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=kv_stack,
                use_cache=True,
                position_embeddings=pos_embeddings,
            )

        h = self.trans.norm(h)
        new_hidden = self.trans.output_proj(h).transpose(1, 2)
        return (new_hidden,) + tuple(kv_stack.key_cache) + tuple(kv_stack.value_cache)


class DecoderPart3Upsample(nn.Module):
    def __init__(self, decoder_model):
        super().__init__()
        self.upsample = decoder_model.upsample
        self.decoder = decoder_model.decoder
        self.samples_per_frame = decoder_model.total_upsample
        self.lookahead_frames = 4
        self.conv_history_window = 4

    def forward(
        self,
        new_hidden: torch.Tensor,
        latent_buffer: torch.Tensor,
        conv_history: torch.Tensor,
        is_last: torch.Tensor,
    ):
        device = new_hidden.device
        accumulated = torch.cat([latent_buffer, new_hidden], dim=-1)

        total_acc_t = torch.zeros(1, device=device, dtype=torch.long) + accumulated.size(2)
        lookahead_t = torch.zeros(1, device=device, dtype=torch.long) + self.lookahead_frames

        total_acc_f = total_acc_t.to(torch.float32)
        lookahead_f = lookahead_t.to(torch.float32)
        num_finalize_f = is_last * total_acc_f + (1.0 - is_last) * torch.clamp(total_acc_f - lookahead_f, min=0.0)
        num_finalize = num_finalize_f.to(torch.long)
        num_finalize_idx = num_finalize[0]

        conv_chain_input = torch.cat([conv_history, accumulated], dim=-1)
        curr = conv_chain_input
        for blocks in self.upsample:
            for block in blocks:
                curr = block(curr)
        for block in self.decoder:
            curr = block(curr)

        wav = curr.squeeze(1).clamp(min=-1, max=1)
        start_samples_idx = conv_history.size(2) * self.samples_per_frame
        valid_samples = (num_finalize * self.samples_per_frame).view(1)
        final_wav = wav[:, start_samples_idx:]

        next_latent_buf = accumulated[:, :, -self.lookahead_frames :]
        bsz, channels = accumulated.size(0), accumulated.size(1)
        indices = torch.arange(self.conv_history_window, device=device, dtype=torch.long)
        target_indices = (num_finalize_idx - self.conv_history_window) + indices
        gather_indices = torch.clamp(target_indices, min=0).unsqueeze(0).unsqueeze(0).expand(bsz, channels, -1)
        next_conv_hist = torch.gather(accumulated, 2, gather_indices)
        return final_wav, valid_samples, next_latent_buf, next_conv_hist


class StatefulDecoderDynamoCombined(nn.Module):
    def __init__(self, decoder_model):
        super().__init__()
        self.part1 = DecoderPart1PreConv(decoder_model)
        self.part2 = DecoderPart2Transformer(decoder_model)
        self.part3 = DecoderPart3Upsample(decoder_model)
        self.num_layers = self.part2.num_layers

    def forward(
        self,
        audio_codes: torch.Tensor,
        pre_conv_history: torch.Tensor,
        latent_buffer: torch.Tensor,
        conv_history: torch.Tensor,
        is_last: torch.Tensor,
        *past_kv_flat,
    ):
        hidden, next_pre_conv_hist = self.part1(audio_codes, pre_conv_history)
        trans_outputs = self.part2(hidden, *past_kv_flat)
        new_hidden = trans_outputs[0]
        next_kv_flat = trans_outputs[1:]
        final_wav, valid_samples, next_latent_buf, next_conv_hist = self.part3(
            new_hidden, latent_buffer, conv_history, is_last
        )
        return (
            final_wav,
            valid_samples,
            next_pre_conv_hist,
            next_latent_buf,
            next_conv_hist,
            *next_kv_flat,
        )
