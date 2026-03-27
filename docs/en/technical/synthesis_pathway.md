# LunaVox Synthesis Pathway Encoder Requirements Analysis

This report evaluates the dependency of LunaVox synthesis pathways on `codec_encoder.fp16.onnx` and `speaker_encoder.fp16.onnx`.

## 1. Core Conclusion

Due to rule updates, **Voice Cloning (clone)** mode is **no longer supported** for `custom` and `design` model types. `custom` models only support built-in expert speakers or standard synthesis, while `design` models only support voice design via text instructions.

Most valid pathways **do not require** loading or running `codec_encoder` and `speaker_encoder`.

These are referred to as **"Cold Inference"** pathways because they don't require real-time feature extraction from raw audio, but instead use text, instructions, predefined speaker IDs, or pre-computed JSON data.

## 2. Pathway List Without Encoder Dependencies

The following pathways rely only on `talker` (GGUF), `predictor` (GGUF), and `decoder` (ONNX):

### 2.1 Base Models
*   **Base (1.7B/0.6B)**: `Standard TTS` (Standard synthesis - from text)
*   **Base (1.7B/0.6B)**: `Clone (JSON)` (Using pre-extracted JSON data, no audio encoding needed)

### 2.2 Custom Models
*   **Custom (1.7B/0.6B)**: `Standard TTS` (Using default voice)
*   **Custom (1.7B/0.6B)**: `Custom Voice` (Using Expert speaker names like Vivian)
    *   System loads pre-computed matrices from `embeddings/`.

### 2.3 Design Models
*   **Design (1.7B)**: `Standard TTS`
*   **Design (1.7B)**: `Voice Design` (Using `--instruct` only)

---

## 3. Why These Pathways Don't Need Encoders?

1.  **Standard TTS / Design**: `speaker_embedding` and `ref_codes` are set to `nullptr` or use model defaults.
2.  **Custom Voice**: Uses solidified embedding vectors from Experts, bypassing real-time extraction.
3.  **Clone (JSON)**: Features are already stored in the JSON file.

## 4. Pathways Requiring Encoders

Currently, **only Base models doing WAV reference cloning** require encoders:

*   **Base (1.7B/0.6B)**: `Clone (WAV)`
    *   Extracts acoustic features (Codes) and Speaker Embedding from WAV.

---

## 5. Cleaning & Recommendations

> [!TIP]
> **Optimization**:
> 1.  **Custom/Design Models**: Since they don't support `clone`, you **can safely delete** all `speaker_encoder` and `codec_encoder` files in these model directories.
> 2.  **Base Models**: If you only use `JSON` or `Standard TTS`, you can also delete them.
> 3.  **Benefit**: Saves about **130MB - 140MB** per model directory.
