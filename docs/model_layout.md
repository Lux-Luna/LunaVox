# LunaVox Model Layout (Current)

Default model root:

- `models/base_small`

Required files:

- `qwen3_tts_talker.q5_k.gguf`
- `qwen3_tts_predictor.q8_0.gguf`
- `qwen3_tts_speaker_encoder.gguf`
- `qwen3_tts_codec_encoder.gguf`
- `qwen3_tts_codec_decoder.gguf`
- `tokenizer.json`
- `embeddings/text_embedding_projected.npy`
- `embeddings/codec_embedding_0.npy` ... `embeddings/codec_embedding_15.npy`

Optional files:

- `embeddings/proj_weight.npy`
- `embeddings/proj_bias.npy`

Setup command:

```bash
python manage.py setup
```

Convert only (reuse local base assets):

```bash
python manage.py convert --skip-download --force
```
