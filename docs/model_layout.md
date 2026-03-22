# LunaVox Model Layout (Current)

Default model root:

- `models/base_small`

Required files:

- `qwen3_tts_talker.q5_k.gguf`
- `qwen3_tts_predictor.q8_0.gguf`
- `qwen3_tts_codec_encoder.fp16.onnx`
- `qwen3_tts_speaker_encoder.fp16.onnx`
- `qwen3_tts_decoder.fp16.onnx`
- `tokenizer.json`
- `embeddings/text_embedding_projected.npy`
- `embeddings/codec_embedding_0.npy` ... `embeddings/codec_embedding_15.npy`

Optional files:

- `embeddings/proj_weight.npy`
- `embeddings/proj_bias.npy`
