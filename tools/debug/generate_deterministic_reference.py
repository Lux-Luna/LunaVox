#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportMissingTypeArgument=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportUnusedCallResult=false
"""
Generate deterministic reference artifacts for Qwen3-TTS GGML validation.

This script forces float32 + greedy decoding and dumps intermediate tensors,
including the exact prefill embedding fed into the talker transformer.

Usage:
    /root/.local/bin/uv run python scripts/generate_deterministic_reference.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-Base"
REF_AUDIO_PATH = PROJECT_ROOT / "clone.wav"
REF_TEXT_PATH = PROJECT_ROOT / "reference_text.txt"
OUTPUT_DIR = PROJECT_ROOT / "reference"

SYNTH_TEXT = "Hello."
LANGUAGE = "English"
MAX_NEW_TOKENS = 64


def _as_numpy(
    value: torch.Tensor | np.ndarray, dtype: np.dtype | None = None
) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().contiguous().numpy()
    else:
        arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _save_bin(
    value: torch.Tensor | np.ndarray, path: Path, dtype: np.dtype | None = None
) -> dict:
    arr = _as_numpy(value, dtype=dtype)
    arr.tofile(path)
    return {
        "path": path.name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size_bytes": int(arr.nbytes),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_prefill_embeddings(
    model, input_id: torch.Tensor, voice_clone_prompt: dict, language: str
):
    tts_model = model.model
    talker = tts_model.talker

    voice_clone_spk_embeds = tts_model.generate_speaker_prompt(voice_clone_prompt)
    speaker_embed = voice_clone_spk_embeds[0]

    language_id = tts_model.config.talker_config.codec_language_id[language.lower()]

    tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(
            torch.tensor(
                [
                    [
                        tts_model.config.tts_bos_token_id,
                        tts_model.config.tts_eos_token_id,
                        tts_model.config.tts_pad_token_id,
                    ]
                ],
                device=talker.device,
                dtype=input_id.dtype,
            )
        )
    ).chunk(3, dim=1)

    codec_prefill_tokens = [
        tts_model.config.talker_config.codec_think_id,
        tts_model.config.talker_config.codec_think_bos_id,
        language_id,
        tts_model.config.talker_config.codec_think_eos_id,
    ]
    codec_input_embedding_0 = talker.get_input_embeddings()(
        torch.tensor([codec_prefill_tokens], device=talker.device, dtype=input_id.dtype)
    )
    codec_input_embedding_1 = talker.get_input_embeddings()(
        torch.tensor(
            [
                [
                    tts_model.config.talker_config.codec_pad_id,
                    tts_model.config.talker_config.codec_bos_id,
                ]
            ],
            device=talker.device,
            dtype=input_id.dtype,
        )
    )

    codec_input_embedding = torch.cat(
        [
            codec_input_embedding_0,
            speaker_embed.view(1, 1, -1),
            codec_input_embedding_1,
        ],
        dim=1,
    )

    role_embed = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
    tts_overlay = torch.cat(
        [
            tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1),
            tts_bos_embed,
        ],
        dim=1,
    )
    codec_prefill_embed = codec_input_embedding[:, :-1]
    codec_plus_overlay = tts_overlay + codec_prefill_embed

    first_text_embed = (
        talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:4]))
        + codec_input_embedding[:, -1:]
    )
    prefill_embedding = torch.cat(
        [role_embed, codec_plus_overlay, first_text_embed], dim=1
    )

    trailing_text_hidden = torch.cat(
        [
            talker.text_projection(talker.get_text_embeddings()(input_id[:, 4:-5])),
            tts_eos_embed,
        ],
        dim=1,
    )

    token_types = [
        {
            "position": 0,
            "type": "role_token",
            "token_id": int(input_id[0, 0].item()),
            "description": "text_projection(<|im_start|>)",
        },
        {
            "position": 1,
            "type": "role_token",
            "token_id": int(input_id[0, 1].item()),
            "description": "text_projection(assistant)",
        },
        {
            "position": 2,
            "type": "role_token",
            "token_id": int(input_id[0, 2].item()),
            "description": "text_projection(newline)",
        },
        {
            "position": 3,
            "type": "codec_overlay",
            "token_id": int(tts_model.config.talker_config.codec_think_id),
            "description": "tts_pad + codec_embedding(codec_think_id)",
        },
        {
            "position": 4,
            "type": "codec_overlay",
            "token_id": int(tts_model.config.talker_config.codec_think_bos_id),
            "description": "tts_pad + codec_embedding(codec_think_bos_id)",
        },
        {
            "position": 5,
            "type": "codec_overlay",
            "token_id": int(language_id),
            "description": "tts_pad + codec_embedding(language_id)",
        },
        {
            "position": 6,
            "type": "codec_overlay",
            "token_id": int(tts_model.config.talker_config.codec_think_eos_id),
            "description": "tts_pad + codec_embedding(codec_think_eos_id)",
        },
        {
            "position": 7,
            "type": "speaker_overlay",
            "token_id": None,
            "description": "tts_pad + projected_speaker_embedding",
        },
        {
            "position": 8,
            "type": "codec_overlay",
            "token_id": int(tts_model.config.talker_config.codec_pad_id),
            "description": "tts_bos + codec_embedding(codec_pad_id)",
        },
        {
            "position": 9,
            "type": "text_plus_codec_bos",
            "token_id": int(input_id[0, 3].item()),
            "codec_token_id": int(tts_model.config.talker_config.codec_bos_id),
            "description": "text_projection(first_text_token) + codec_embedding(codec_bos_id)",
        },
    ]

    return {
        "speaker_embed": speaker_embed,
        "tts_bos_embed": tts_bos_embed,
        "tts_eos_embed": tts_eos_embed,
        "tts_pad_embed": tts_pad_embed,
        "role_embed": role_embed,
        "codec_prefill_embed": codec_prefill_embed,
        "tts_overlay": tts_overlay,
        "prefill_embedding": prefill_embedding,
        "trailing_text_hidden": trailing_text_hidden,
        "token_types": token_types,
    }


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.use_deterministic_algorithms(True)

    print("=" * 72)
    print("Qwen3-TTS Deterministic Reference Generator")
    print("=" * 72)

    for required_path in (MODEL_PATH, REF_AUDIO_PATH, REF_TEXT_PATH):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required path: {required_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ref_text = REF_TEXT_PATH.read_text(encoding="utf-8").strip()

    from qwen_tts import Qwen3TTSModel

    print("Loading model in float32 on CPU...")
    model = Qwen3TTSModel.from_pretrained(
        str(MODEL_PATH),
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    model.model = model.model.eval()

    assistant_text = model._build_assistant_text(SYNTH_TEXT)
    input_ids_list = model._tokenize_texts([assistant_text])
    input_id = input_ids_list[0].to(model.device)

    prompt_items = model.create_voice_clone_prompt(
        ref_audio=str(REF_AUDIO_PATH),
        ref_text=ref_text,
        x_vector_only_mode=True,
    )
    voice_clone_prompt = model._prompt_items_to_voice_clone_prompt(prompt_items)

    prefill_data = _build_prefill_embeddings(
        model,
        input_id=input_id,
        voice_clone_prompt=voice_clone_prompt,
        language=LANGUAGE,
    )
    prefill_embedding = prefill_data["prefill_embedding"]
    trailing_text_hidden = prefill_data["trailing_text_hidden"]
    tts_pad_embed = prefill_data["tts_pad_embed"]

    tts_model = model.model
    talker = tts_model.talker
    prefill_attention_mask = torch.ones(
        (prefill_embedding.shape[0], prefill_embedding.shape[1]),
        dtype=torch.long,
        device=prefill_embedding.device,
    )

    with torch.no_grad():
        first_step = talker(
            inputs_embeds=prefill_embedding,
            attention_mask=prefill_attention_mask,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            output_hidden_states=True,
            use_cache=True,
            subtalker_dosample=False,
            subtalker_top_p=None,
            subtalker_top_k=None,
            subtalker_temperature=None,
        )
    first_frame_logits = first_step.logits[:, -1, :]

    with torch.no_grad():
        talker_codes_list, talker_hidden_states_list = tts_model.generate(
            input_ids=[input_id],
            ref_ids=None,
            voice_clone_prompt=voice_clone_prompt,
            languages=[LANGUAGE],
            non_streaming_mode=False,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            top_k=None,
            top_p=None,
            temperature=None,
            subtalker_dosample=False,
            subtalker_top_k=None,
            subtalker_top_p=None,
            subtalker_temperature=None,
        )

    speech_codes = talker_codes_list[0]
    hidden_states = talker_hidden_states_list[0]

    wavs, sample_rate = tts_model.speech_tokenizer.decode(
        [{"audio_codes": speech_codes}]
    )
    decoded_audio = np.asarray(wavs[0], dtype=np.float32)

    outputs = {}
    outputs["det_text_tokens.bin"] = _save_bin(
        input_id, OUTPUT_DIR / "det_text_tokens.bin", dtype=np.int64
    )
    outputs["det_speaker_embedding.bin"] = _save_bin(
        prompt_items[0].ref_spk_embedding,
        OUTPUT_DIR / "det_speaker_embedding.bin",
        dtype=np.float32,
    )
    outputs["det_prefill_embedding.bin"] = _save_bin(
        prefill_embedding, OUTPUT_DIR / "det_prefill_embedding.bin", dtype=np.float32
    )
    outputs["det_speech_codes.bin"] = _save_bin(
        speech_codes, OUTPUT_DIR / "det_speech_codes.bin", dtype=np.int64
    )
    outputs["det_hidden_states.bin"] = _save_bin(
        hidden_states, OUTPUT_DIR / "det_hidden_states.bin", dtype=np.float32
    )
    outputs["det_first_frame_logits.bin"] = _save_bin(
        first_frame_logits, OUTPUT_DIR / "det_first_frame_logits.bin", dtype=np.float32
    )
    outputs["det_decoded_audio.bin"] = _save_bin(
        decoded_audio, OUTPUT_DIR / "det_decoded_audio.bin", dtype=np.float32
    )
    outputs["det_trailing_text.bin"] = _save_bin(
        trailing_text_hidden, OUTPUT_DIR / "det_trailing_text.bin", dtype=np.float32
    )
    outputs["det_tts_pad_embed.bin"] = _save_bin(
        tts_pad_embed, OUTPUT_DIR / "det_tts_pad_embed.bin", dtype=np.float32
    )
    outputs["det_role_embed.bin"] = _save_bin(
        prefill_data["role_embed"], OUTPUT_DIR / "det_role_embed.bin", dtype=np.float32
    )
    outputs["det_codec_prefill_embed.bin"] = _save_bin(
        prefill_data["codec_prefill_embed"],
        OUTPUT_DIR / "det_codec_prefill_embed.bin",
        dtype=np.float32,
    )
    outputs["det_tts_overlay.bin"] = _save_bin(
        prefill_data["tts_overlay"],
        OUTPUT_DIR / "det_tts_overlay.bin",
        dtype=np.float32,
    )

    sf.write(str(OUTPUT_DIR / "det_output.wav"), decoded_audio, sample_rate)

    token_types_payload = {
        "language": LANGUAGE,
        "x_vector_only_mode": True,
        "non_streaming_mode": False,
        "prefill_sequence_length": int(prefill_embedding.shape[1]),
        "positions": prefill_data["token_types"],
    }
    (OUTPUT_DIR / "det_prefill_token_types.json").write_text(
        json.dumps(token_types_payload, indent=2),
        encoding="utf-8",
    )

    num_samples = len(decoded_audio)

    metadata = {
        "model_path": str(MODEL_PATH),
        "reference_audio": str(REF_AUDIO_PATH),
        "reference_text": ref_text,
        "synthesis_text": SYNTH_TEXT,
        "language": LANGUAGE,
        "dtype": "float32",
        "device": str(model.device),
        "x_vector_only_mode": True,
        "non_streaming_mode": False,
        "generation": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "subtalker_dosample": False,
            "top_k": None,
            "top_p": None,
            "temperature": None,
            "subtalker_top_k": None,
            "subtalker_top_p": None,
            "subtalker_temperature": None,
        },
        "token_ids": {
            "tts_bos_token_id": int(tts_model.config.tts_bos_token_id),
            "tts_eos_token_id": int(tts_model.config.tts_eos_token_id),
            "tts_pad_token_id": int(tts_model.config.tts_pad_token_id),
            "codec_bos_id": int(tts_model.config.talker_config.codec_bos_id),
            "codec_eos_token_id": int(
                tts_model.config.talker_config.codec_eos_token_id
            ),
            "codec_pad_id": int(tts_model.config.talker_config.codec_pad_id),
            "codec_think_id": int(tts_model.config.talker_config.codec_think_id),
            "codec_think_bos_id": int(
                tts_model.config.talker_config.codec_think_bos_id
            ),
            "codec_think_eos_id": int(
                tts_model.config.talker_config.codec_think_eos_id
            ),
            "language_id": int(
                tts_model.config.talker_config.codec_language_id[LANGUAGE.lower()]
            ),
        },
        "shapes": {
            "input_ids": list(input_id.shape),
            "speaker_embedding": list(prompt_items[0].ref_spk_embedding.shape),
            "prefill_embedding": list(prefill_embedding.shape),
            "trailing_text_hidden": list(trailing_text_hidden.shape),
            "tts_pad_embed": list(tts_pad_embed.shape),
            "speech_codes": list(speech_codes.shape),
            "hidden_states": list(hidden_states.shape),
            "first_frame_logits": list(first_frame_logits.shape),
            "decoded_audio": list(decoded_audio.shape),
        },
        "outputs": outputs,
        "wav": {
            "path": "det_output.wav",
            "sha256": _hash_file(OUTPUT_DIR / "det_output.wav"),
            "sample_rate": int(sample_rate),
            "num_samples": num_samples,
            "duration_seconds": float(num_samples / sample_rate),
        },
        "prefill_token_types": "det_prefill_token_types.json",
    }
    (OUTPUT_DIR / "det_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("First 5 speech-code frames [n_frames x 16]:")
    print(_as_numpy(speech_codes[:5], dtype=np.int64))
    print("First 10 values of det_prefill_embedding.bin:")
    print(_as_numpy(prefill_embedding, dtype=np.float32).reshape(-1)[:10])
    print(
        f"Generated {speech_codes.shape[0]} frames and {decoded_audio.shape[0]} audio samples @ {sample_rate} Hz"
    )
    print(f"Wrote deterministic artifacts to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
