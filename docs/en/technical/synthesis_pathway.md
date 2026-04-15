# Synthesis Pathway: When Do We Need the Encoders?

`codec_encoder.fp16.onnx` and `speaker_encoder.fp16.onnx` are only required when the engine has to extract speaker / codec features from a raw WAV in real time. Every other path is "cold inference" — it consumes text, instructions, a built-in speaker ID, or a pre-computed JSON.

## Cold-inference pathways (no encoder needed)

Only `talker` (GGUF), `predictor` (GGUF), and `decoder` (ONNX) are touched.

| Model type | Pathway | Notes |
| --- | --- | --- |
| Base (0.6B / 1.7B) | Standard TTS | Text only; `speaker_embedding` / `ref_codes` are null. |
| Base (0.6B / 1.7B) | Clone from JSON | `--reference foo.json` — features already cached. |
| Custom (0.6B / 1.7B) | Standard TTS | Default voice. |
| Custom (0.6B / 1.7B) | Custom voice | `--speaker Vivian` — embeddings loaded from `embeddings/`. |
| Design (1.7B) | Standard TTS | Text only. |
| Design (1.7B) | Voice design | `--instruct` only. |

## The one path that still needs encoders

| Model type | Pathway | Why |
| --- | --- | --- |
| Base (0.6B / 1.7B) | Clone from WAV | Real-time extraction of acoustic codes + speaker embedding from `--reference foo.wav`. |

`custom` and `design` models **do not support clone** at all (the loader rejects `--reference`).

## Cleanup tip

If you only ever use cold-inference pathways, both `speaker_encoder.fp16.onnx` and `codec_encoder.fp16.onnx` can be deleted from any `models/<name>/` directory. Saves ~130–140 MB per model. The runtime tolerates their absence.
