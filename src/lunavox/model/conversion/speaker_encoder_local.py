#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


@dataclass
class SpeakerEncoderConfig:
    mel_dim: int = 128
    enc_dim: int = 1024
    enc_channels: list[int] = field(default_factory=lambda: [512, 512, 512, 512, 1536])
    enc_kernel_sizes: list[int] = field(default_factory=lambda: [5, 3, 3, 3, 1])
    enc_dilations: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 1])
    enc_attention_channels: int = 128
    enc_res2net_scale: int = 8
    enc_se_channels: int = 128
    sample_rate: int = 24000


class TimeDelayNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
            padding_mode="reflect",
        )
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(hidden_states))


class Res2NetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 8,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.blocks = nn.ModuleList(
            [
                TimeDelayNetBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for _ in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i, hidden_part in enumerate(torch.chunk(hidden_states, self.scale, dim=1)):
            if i == 0:
                output_part = hidden_part
            elif i == 1:
                output_part = self.blocks[i - 1](hidden_part)
            else:
                output_part = self.blocks[i - 1](hidden_part + output_part)
            outputs.append(output_part)
        return torch.cat(outputs, dim=1)


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels: int, se_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=se_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            in_channels=se_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_mean = hidden_states.mean(dim=2, keepdim=True)
        hidden_states_mean = self.relu(self.conv1(hidden_states_mean))
        hidden_states_mean = self.sigmoid(self.conv2(hidden_states_mean))
        return hidden_states * hidden_states_mean


class SqueezeExcitationRes2NetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res2net_scale: int = 8,
        se_channels: int = 128,
        kernel_size: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.tdnn1 = TimeDelayNetBlock(in_channels, out_channels, kernel_size=1, dilation=1)
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, kernel_size, dilation
        )
        self.tdnn2 = TimeDelayNetBlock(out_channels, out_channels, kernel_size=1, dilation=1)
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels, out_channels)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.tdnn1(hidden_state)
        hidden_state = self.res2net_block(hidden_state)
        hidden_state = self.tdnn2(hidden_state)
        hidden_state = self.se_block(hidden_state)
        return hidden_state + residual


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels: int, attention_channels: int = 128):
        super().__init__()
        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(
            in_channels=attention_channels,
            out_channels=channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def _length_to_mask(self, length, max_len=None, dtype=None, device=None):
        if max_len is None:
            max_len = length.max().long().item()
        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len
        ) < length.unsqueeze(1)
        return torch.as_tensor(mask, dtype=dtype, device=device)

    def _compute_statistics(self, x, m, dim=2):
        mean = (m * x).sum(dim)
        std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(self.eps))
        return mean, std

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_length = hidden_states.shape[-1]
        lengths = torch.ones(hidden_states.shape[0], device=hidden_states.device)
        mask = self._length_to_mask(
            lengths * seq_length,
            max_len=seq_length,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ).unsqueeze(1)

        total = mask.sum(dim=2, keepdim=True)
        mean, std = self._compute_statistics(hidden_states, mask / total)
        mean = mean.unsqueeze(2).repeat(1, 1, seq_length)
        std = std.unsqueeze(2).repeat(1, 1, seq_length)
        attention = torch.cat([hidden_states, mean, std], dim=1)

        attention = self.conv(self.tanh(self.tdnn(attention)))
        attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(attention, dim=2)

        mean, std = self._compute_statistics(hidden_states, attention)
        pooled_stats = torch.cat((mean, std), dim=1).unsqueeze(2)
        return pooled_stats


class Qwen3TTSSpeakerEncoder(nn.Module):
    def __init__(self, config: SpeakerEncoderConfig):
        super().__init__()
        if len(config.enc_channels) != len(config.enc_kernel_sizes) or len(
            config.enc_channels
        ) != len(config.enc_dilations):
            raise ValueError(
                "enc_channels, enc_kernel_sizes and enc_dilations should have same length"
            )
        self.blocks = nn.ModuleList()

        self.blocks.append(
            TimeDelayNetBlock(
                config.mel_dim,
                config.enc_channels[0],
                config.enc_kernel_sizes[0],
                config.enc_dilations[0],
            )
        )

        for i in range(1, len(config.enc_channels) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    config.enc_channels[i - 1],
                    config.enc_channels[i],
                    res2net_scale=config.enc_res2net_scale,
                    se_channels=config.enc_se_channels,
                    kernel_size=config.enc_kernel_sizes[i],
                    dilation=config.enc_dilations[i],
                )
            )

        self.mfa = TimeDelayNetBlock(
            config.enc_channels[-1],
            config.enc_channels[-1],
            config.enc_kernel_sizes[-1],
            config.enc_dilations[-1],
        )
        self.asp = AttentiveStatisticsPooling(
            config.enc_channels[-1],
            attention_channels=config.enc_attention_channels,
        )
        self.fc = nn.Conv1d(
            in_channels=config.enc_channels[-1] * 2,
            out_channels=config.enc_dim,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states_list = []
        for layer in self.blocks:
            hidden_states = layer(hidden_states)
            hidden_states_list.append(hidden_states)

        hidden_states = torch.cat(hidden_states_list[1:], dim=1)
        hidden_states = self.mfa(hidden_states)
        hidden_states = self.asp(hidden_states)
        hidden_states = self.fc(hidden_states)
        return hidden_states.squeeze(-1)


def load_speaker_encoder_from_base_dir(base_dir: Path) -> Qwen3TTSSpeakerEncoder:
    config_path = base_dir / "config.json"
    model_path = base_dir / "model.safetensors"
    if not config_path.exists():
        raise RuntimeError(f"Missing config: {config_path}")
    if not model_path.exists():
        raise RuntimeError(f"Missing model weights: {model_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    speaker_cfg_raw = cfg.get("speaker_encoder_config", {})
    speaker_cfg = SpeakerEncoderConfig(**speaker_cfg_raw)

    model = Qwen3TTSSpeakerEncoder(speaker_cfg)
    tensors = load_file(str(model_path))
    state_dict = {}
    for k, v in tensors.items():
        if k.startswith("speaker_encoder."):
            state_dict[k[len("speaker_encoder.") :]] = v
    if not state_dict:
        raise RuntimeError("No speaker encoder tensors found in model.safetensors")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Speaker encoder state_dict missing keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Speaker encoder state_dict unexpected keys: {unexpected}")
    model.eval()
    return model
