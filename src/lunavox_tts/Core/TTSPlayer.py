# 文件: .../Core/TTSPlayer.py

import queue
import os
import threading
import time
import numpy as np
import wave
import logging
from typing import Optional, List, Callable

from .Inference import LunaVoxEngine
from .Session import SynthesisSession
from .Frontend.processor import text_processor
from .Pipeline.WaveSaver import save_wav, preprocess_for_playback
from ..ModelManager import model_manager
from ..Utils.Utils import clear_queue
from ..Resources.Audio.ReferenceAudio import ReferenceAudio
from ..Utils.PerformanceMonitor import monitor
from ..Resources.Audio.AudioEngine import AudioEngine

logger = logging.getLogger(__name__)

STREAM_END = 'STREAM_END'  # 特殊标记，表示文本流结束


class TTSPlayer:
    """
    High-level TTS player that handles text queuing, synthesis, and audio playback.
    """
    
    def __init__(self, sample_rate: int = 32000):
        self.sample_rate: int = sample_rate
        self.channels: int = 1
        self.bytes_per_sample: int = 2  # 16-bit audio
        
        self.inference_engine: LunaVoxEngine = LunaVoxEngine()
        self.audio_engine: AudioEngine = AudioEngine(sample_rate, self.channels)

        self._text_queue: queue.Queue = queue.Queue()
        self._audio_queue: queue.Queue = queue.Queue()

        self._stop_event: threading.Event = threading.Event()
        self._tts_done_event: threading.Event = threading.Event()
        self._tts_done_event.set()
        self._api_lock: threading.Lock = threading.Lock()

        self._tts_worker: Optional[threading.Thread] = None
        self._playback_worker: Optional[threading.Thread] = None

        self._play: bool = False
        self._current_save_path: Optional[str] = None
        self._session_audio_chunks: List[np.ndarray] = []
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._split: bool = False

        self._chunk_callback: Optional[Callable[[Optional[bytes]], None]] = None
        self._current_session: Optional[SynthesisSession] = None

    def _tts_worker_loop(self):
        """TTS processing worker: pulls text, runs inference, dispatches audio."""
        while not self._stop_event.is_set():
            try:
                sentence = self._text_queue.get(timeout=1)
                if sentence is None or self._stop_event.is_set():
                    break
            except queue.Empty:
                continue

            try:
                if sentence is STREAM_END:
                    if self._current_save_path and self._session_audio_chunks:
                        save_wav(
                            self._session_audio_chunks,
                            self._current_save_path,
                            self.sample_rate,
                            self.channels,
                            self.bytes_per_sample
                        )
                        self._session_audio_chunks = []
                        self._current_save_path = None

                    if self._chunk_callback:
                        self._chunk_callback(None)

                    if self._start_time:
                        total_duration = time.perf_counter() - self._start_time
                        monitor.log_metric("Total TTS session time", f"{total_duration:.3f}", "s")

                    self._tts_done_event.set()
                    self._current_session = None
                    continue

                if not self._current_session:
                    logger.error("No active session for TTS processing.")
                    continue

                # Run inference via engine
                audio_chunk = self.inference_engine.generate(
                    text=sentence,
                    session=self._current_session,
                    stop_event=self._stop_event
                )

                if audio_chunk is not None:
                    if self._end_time is None:
                        self._end_time = time.perf_counter()
                        if self._start_time:
                            duration: float = self._end_time - self._start_time
                            monitor.log_metric("First packet latency", f"{duration:.3f}", "s")

                    if self._play:
                        self._audio_queue.put(audio_chunk)
                    if self._current_save_path:
                        self._session_audio_chunks.append(audio_chunk)

                    if self._chunk_callback:
                        audio_data = preprocess_for_playback(audio_chunk)
                        self._chunk_callback(audio_data)

            except Exception as e:
                logger.error(f"Critical error in TTS worker: {e}", exc_info=True)
                if self._chunk_callback:
                    self._chunk_callback(None)
                self._tts_done_event.set()
                self._current_session = None

    def _playback_worker_loop(self):
        try:
            while not self._stop_event.is_set():
                try:
                    audio_chunk = self._audio_queue.get(timeout=1)
                    if audio_chunk is None or audio_chunk is STREAM_END:
                        break
                    
                    if self._play:
                        self.audio_engine.play(audio_chunk)
                        
                    self._audio_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in audio playback: {e}", exc_info=True)
        finally:
            self.audio_engine.stop()

    def start_session(self,
                      play: bool = False,
                      split: bool = False,
                      save_path: Optional[str] = None,
                      chunk_callback: Optional[Callable[[Optional[bytes]], None]] = None,
                      speaker: Optional[str] = None,
                      prompt_audio: Optional[ReferenceAudio] = None,
                      language: Optional[str] = None
                      ):
        with self._api_lock:
            if self._tts_worker and not self._tts_done_event.is_set():
                raise RuntimeError("A TTS session is already running.")
                
            self._tts_done_event.clear()
            self._chunk_callback = chunk_callback
            self._stop_event.clear()

            # Ensure workers are alive
            if self._tts_worker is None or not self._tts_worker.is_alive():
                self._tts_worker = threading.Thread(target=self._tts_worker_loop, daemon=True)
                self._tts_worker.start()

            if self._playback_worker is None or not self._playback_worker.is_alive():
                self._playback_worker = threading.Thread(target=self._playback_worker_loop, daemon=True)
                self._playback_worker.start()

            clear_queue(self._text_queue)
            clear_queue(self._audio_queue)

            self._play = play
            self._split = split
            self._current_save_path = save_path
            self._session_audio_chunks = []
            self._start_time = None
            self._end_time = None
            
            # Validate required parameters
            if not speaker:
                raise ValueError("speaker is required for TTS session")
            if prompt_audio is None:
                raise ValueError("prompt_audio is required for TTS session")
            
            resolved_language = (language or "ja").lower()
            
            # Create session-based state
            skip_pe = getattr(prompt_audio, 'is_persona_based', False)
            gsv_model = model_manager.get(speaker, skip_prompt_encoder=skip_pe)
            if not gsv_model:
                raise RuntimeError(f"Failed to load model for {speaker}")
                
            self._current_session = SynthesisSession(
                speaker=speaker,
                language=resolved_language,
                prompt_audio=prompt_audio,
                model=gsv_model,
                model_version=model_manager.get_character_version(speaker),
                skip_prompt_encoder=skip_pe
            )
            
            if play:
                self.audio_engine.warmup()

    def feed(self, text_chunk: str):
        with self._api_lock:
            if not text_chunk or not self._current_session:
                return
            
            if self._start_time is None:
                self._start_time = time.perf_counter()

            if self._split:
                sentences = text_processor.split_sentences(text_chunk, self._current_session.language)
                for sentence in sentences:
                    self._text_queue.put(sentence)
            else:
                self._text_queue.put(text_chunk)

    def end_session(self):
        with self._api_lock:
            self._text_queue.put(STREAM_END)

    def stop(self):
        with self._api_lock:
            if self._tts_worker is None and self._playback_worker is None:
                return
            if self._stop_event.is_set():
                return
            self.inference_engine.stop_event.set()
            self._stop_event.set()
            self._tts_done_event.set()
            self._text_queue.put(None)
            self._audio_queue.put(None)
            self._current_session = None
            if self._tts_worker and self._tts_worker.is_alive():
                self._tts_worker.join()
            if self._playback_worker and self._playback_worker.is_alive():
                self._playback_worker.join()
            self._tts_worker = None
            self._playback_worker = None

    def wait_for_tts_completion(self):
        if self._tts_done_event.is_set():
            return
        self._tts_done_event.wait()


tts_player: TTSPlayer = TTSPlayer()
