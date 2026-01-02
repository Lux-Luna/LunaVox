# 📦 Dependency & Size Analysis

LunaVox is designed with a **modular architecture**, allowing for extreme "Lazy Loading". You only pay the disk space cost for the features you actually use.

This document breaks down the **theoretical minimum disk footprint** for various deployment scenarios (based on ONNX Runtime CPU).

---

## 📊 Summary: Minimum Disk Footprint

| Mode | English (En) | Japanese (Ja) | Chinese (Zh) | Key Additions |
| :--- | :--- | :--- | :--- | :--- |
| **V2 Persona** | **~307 MB** | **~306 MB** | **~710 MB** | Core + G2P (Zh adds RoBERTa) |
| **V2 Reference** | **~687 MB** | **~686 MB** | **~1.1 GB** | + HuBERT (~380MB) |
| **V2PP Persona** | **~312 MB** | **~311 MB** | **~715 MB** | V2PP Model is slightly larger |
| **V2PP Reference** | **~772 MB** | **~771 MB** | **~1.2 GB** | + PromptEncoder & SpeakerVector |

> **Note**: These are estimates for the **runtime environment** (Python + Libraries + Models). System libraries (like CUDA) are not included.

---

## 🧩 Component Breakdown

### 1. Base Engine (~60 MB)
Required for ALL modes.
*   Python Source Code
*   `numpy`, `sounddevice`, etc.
*   `onnxruntime` (CPU)

### 2. Core Models
*   **V2 Standard**: ~240 MB (VITS + TextEncoder + Decoders)
*   **V2 Pro Plus**: ~245 MB (V2PP VITS + TextEncoder + Decoders)

### 3. Language Assets
*   **English/Japanese**: < 10 MB (G2P Dictionaries)
*   **Chinese**: ~400 MB (RoBERTa Model - Required for prosody)

### 4. Zero-Shot / Cloning Assets
*   **HuBERT**: ~380 MB (Required for any Reference Audio mode)
*   **Speaker Vector**: ~20 MB (Required for V2PP Reference mode)
*   **Prompt Encoder**: ~60 MB (Required for V2PP Reference mode)

---

## 💡 Configuration Guide

### Scenario A: Lightweight English TTS App
**Goal**: Offline English TTS with preset characters (Persona). No voice cloning.
*   **Base**: 60 MB
*   **V2 Model**: 240 MB
*   **English G2P**: 7 MB
*   **Total**: **~307 MB** 🚀

### Scenario B: High-End Chinese Voice Cloning
**Goal**: High-quality Chinese TTS with ability to clone voices from audio.
*   **Base**: 60 MB
*   **V2PP Model**: 245 MB
*   **Chinese G2P + RoBERTa**: 410 MB
*   **HuBERT**: 380 MB
*   **Prompt Enc + SV**: 80 MB
*   **Total**: **~1.2 GB**

### Scenario C: GPU Acceleration
If you need GPU speed (CUDA):
*   Add `onnxruntime-gpu`: +200 MB
*   Add CUDA Libraries: +500 MB to 1 GB (unless already installed on system)
