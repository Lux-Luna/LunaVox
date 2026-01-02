# LunaVox Project Structure Guide 🏗️

This document details the code organization and core module design of LunaVox. LunaVox adopts a modular layered architecture aimed at low coupling, high cohesion, and extreme inference performance.

## 📂 Top-Level Overview

The core source code is located in the `src/lunavox_tts` directory.

```
src/lunavox_tts/
├── API/              # 📢 API Layer: Stable public interfaces exposed to users
├── Core/             # ⚙️ Core Engine: TTS business logic, inference pipelines, frontend
├── Interface/        # 🖥️ Interface Layer: CLI and HTTP server
├── Languages/        # 🌍 Languages Layer: G2P and normalization logic for ZH/EN/JA
├── Resources/        # 📦 Resources Layer: Data structures (Audio, Persona) & static assets
├── Utils/            # 🛠️ Utils Layer: Env management, downloads, lifecycle utilities
├── ModelManager.py   # 🧠 Character Model Lifecycle Management (Facade)
└── ResourceManager.py# 💎 Global Resource Owner (Singleton)
```

---

## 🧩 Core Module Breakdown

### 1. API Layer (`/API`)
This is the outermost layer touched by users and developers. It abstracts away complex object management, providing simple, intuitive functions.
*   `synthesis.py`: Core synthesis functions (`tts`, `tts_async`).
*   `characters.py`: Character loading/unloading, reference audio setup.
*   `state.py`: Maintains lightweight runtime configuration state.

### 2. Core Engine (`/Core`)
The heart of the TTS system, responsible for converting text into audio streams.
*   `TTSPlayer.py`: The central controller. Manages playback queues, synthesis threads, and callback events.
*   `Session.py`: Defines the context state for a single synthesis task (`SynthesisSession`), ensuring thread safety.
*   **Processors/**: Feature extractors (e.g., `feature_extractor.py` for SSL/HuBERT features) run before inference.
*   **Model/**: ONNX model loaders and execution policies.
*   **Frontend/**: Text frontend pipeline responsible for tokenization, phonemization, and prosody prediction.

### 3. Resource & Model Management (The Managers)
This is the most critical part of LunaVox's architecture, balancing performance and memory usage.

*   **`ResourceManager.py` (New)**:
    *   **Role**: The **Sole Legal Owner** of Global Shared Resources.
    *   **Manages**: HuBERT models (SSL), BERT models (Semantic features).
    *   **Trait**: True Singleton, ensuring heavy resources exist only once in memory.

*   **`ModelManager.py`**:
    *   **Role**: Facade for Character Model lifecycle.
    *   **Features**: Handles loading of `v2/v2pp` models, LRU caching/eviction, and delegates queries for global resources to `ResourceManager`.

*   **`Utils/RuntimeManager.py`**:
    *   **Role**: Runtime State Orchestrator.
    *   **Features**: Does not hold resources directly. Acts as a "Commander" that calls standard `unload()` interfaces on various modules to perform deep memory cleanup (e.g., for benchmarking).

*   **`Utils/AssetManager.py`**:
    *   **Role**: On-Demand Downloader.
    *   **Features**: Lazy-loads resource packs based on language needs (e.g., "Chinese only" or "Japanese only") from HuggingFace Hub, minimizing initial install size.

### 4. Resources Layer (`/Resources`)
*   **Audio/**: Audio-related data structures.
    *   `ReferenceAudio.py`: Core data class encapsulating raw audio, 16k resampling, and cached feature embeddings.
    *   `SpeakerVector.py`: Contains the Speaker Encoder model (ERes2NetV2).
*   **Persona/**: Configuration schema and manager for "Persona Mode" (Reference-free TTS).

---

## 💡 Key Architectural Decisions

### 1. Explicit Modular Imports
To avoid complex circular dependencies, the project strictly adheres to explicit import principles.
*   Dependency Direction: `RunTimeManager` -> `ModelManager` -> `ResourceManager`.
*   All "heavy modules" are managed in `Utils/RuntimeManager.py` via standard top-level imports, eliminating the need for "Inline Imports".

### 2. Extreme Lazy Loading
LunaVox loads **no models** by default.
*   Character models are loaded only when `load_character("Name")` is called.
*   `ZhBert` and `HuBERT` are triggered for download and load only when synthesizing Chinese text.
*   This design allows LunaVox to start up in extremely low-memory environments.

### 3. Testability First
*   The introduction of `ResourceManager` allows unit tests to easily Mock global resources.
*   The separation of `API` and `Core` allows inference logic to be tested independently of playback hardware.

---
*Last Updated: 2026-01-02*
