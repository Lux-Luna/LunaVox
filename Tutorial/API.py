"""
🔮 LunaVox API Tutorial & Documentation
--------------------------------------

This file provides a comprehensive guide on how to use the LunaVox TTS engine via its Python API.
LunaVox is a lightweight, high-performance inference engine for GPT-SoVITS models.

Supported Model Versions:
- GPT-SoVITS V2 (Standard)
- GPT-SoVITS V2 Pro
- GPT-SoVITS V2 Pro Plus
"""

import lunavox_tts as lunavox
import asyncio

def basic_usage():
    """
    Demonstrates the simplest way to use LunaVox:
    1. Load a character model.
    2. Set a reference audio (for voice cloning).
    3. Perform TTS synthesis.
    """
    print("--- Basic Usage ---")
    
    # 1. Load a character model bundle
    # A bundle is a directory containing the converted ONNX/BIN models.
    # Supported versions (v2, v2Pro, v2ProPlus) are detected automatically.
    character_dir = "./CharacterData/character_model/v2/pretrained"
    lunavox.load_character("miku", character_dir)
    
    # 2. Set reference audio for cloning
    # You must provide the audio path and its transcription.
    # The language of the reference audio can be specified (default is "auto").
    lunavox.set_reference_audio(
        character_name="miku",
        audio_path="./ref.wav",
        audio_text="这是一段参考音频的内容文本。",
        audio_language="zh"  # Language: zh, en, ja
    )
    
    # 3. Synchronous TTS
    # This call blocks until the entire sentence is synthesized.
    # 'play=True' will play the result through your speakers.
    # 'save_path' will save the result to a .wav file.
    lunavox.tts(
        character_name="miku",
        text="你好，欢迎使用 LunaVox 语音合成引擎！",
        play=True,
        save_path="./output.wav",
        language="zh"
    )

async def streaming_usage():
    """
    Demonstrates asynchronous streaming TTS.
    Ideal for real-time applications where low latency is critical.
    """
    print("\n--- Streaming Usage ---")
    
    # lunavox.tts_async returns an AsyncIterator[bytes]
    # It yields audio chunks (PCM data) as they are generated.
    text = "Hello, this is a demonstration of streaming inference. It is very fast!"
    
    async for chunk in lunavox.tts_async(
        character_name="miku",
        text=text,
        play=True,        # Play chunks immediately as they arrive
        language="en"
    ):
        # Process the raw bytes here (e.g., send over network, write to buffer)
        # Each chunk is a segment of the generated waveform.
        pass

def multi_reference_usage():
    """
    V2 Pro and V2 Pro Plus models support averaging multiple reference audios
    to achieve a more stable and accurate character timbre.
    """
    print("\n--- Multi-Reference Usage ---")
    
    # For Pro/ProPlus models, use 'set_multi_reference_audio'
    paths = ["./ref1.wav", "./ref2.wav", "./ref3.wav"]
    texts = ["Text of ref 1", "Text of ref 2", "Text of ref 3"]
    
    # This will average the Speaker Vectors from all provided samples.
    lunavox.set_multi_reference_audio(
        character_name="miku",
        audio_paths=paths,
        audio_texts=texts,
        audio_languages=["en", "en", "en"]
    )
    
    lunavox.tts("miku", "This uses averaged speaker embeddings for better stability.")

def advanced_features():
    """
    Other useful API functions for resource management and services.
    """
    print("\n--- Advanced Features ---")
    
    # Unload a character to free GPU/RAM memory
    lunavox.unload_character("miku")
    
    # Clear internal cache for reference audio features
    lunavox.clear_reference_audio_cache()
    
    # Start a FastAPI server for external access (Blocks the thread)
    # Default: host="127.0.0.1", port=8000, workers=1
    # lunavox.start_server(host="0.0.0.0", port=8000)
    
    # Convert original PyTorch models (.ckpt/.pth) to LunaVox format
    # lunavox.convert_to_onnx(
    #     torch_ckpt_path="path/to/t2s.ckpt",
    #     torch_pth_path="path/to/vits.pth",
    #     output_dir="./ConvertedModel"
    # )

if __name__ == "__main__":
    # Note: Ensure you have the necessary models and reference audio files
    # at the specified paths before running these examples.
    
    # 1. Basic Synchronous Example
    # basic_usage()
    
    # 2. Async Streaming Example
    # asyncio.run(streaming_usage())
    
    # 3. Multi-ref Example (Pro/ProPlus only)
    # multi_reference_usage()
    
    # 4. Utilities
    # advanced_features()
    
    print("Check the code in Tutorial/API.py to learn more about the LunaVox API!")
