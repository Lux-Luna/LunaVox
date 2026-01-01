# 文件: .../Core/TTSPlayer.py

import queue
import re
import os
import threading
import time

import numpy as np
import wave
import logging
from typing import Optional, List, Callable

from .Inference import LunaVoxEngine
from ..Japanese.Split import split_japanese_text
from ..ModelManager import model_manager
from ..Utils.Shared import context
from ..Utils.Utils import clear_queue
from ..Audio.ReferenceAudio import ReferenceAudio
from ..Utils.PerformanceMonitor import monitor
from ..Audio.AudioEngine import AudioEngine

logger = logging.getLogger(__name__)

STREAM_END = 'STREAM_END'  # 这是一个特殊的标记，表示文本流结束


class TTSPlayer:
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

        self._pa: Optional[pyaudio.PyAudio] = None
        self._pa_stream: Optional[pyaudio.Stream] = None

        self._play: bool = False
        self._current_save_path: Optional[str] = None
        self._session_audio_chunks: List[np.ndarray] = []
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._split: bool = False

        self._chunk_callback: Optional[Callable[[Optional[bytes]], None]] = None
        self._session_speaker: Optional[str] = None
        self._session_prompt_audio: Optional[ReferenceAudio] = None
        self._session_language: str = "ja"

    @staticmethod
    def _preprocess_for_playback(audio_float: np.ndarray) -> bytes:
        # Check for NaNs or Infs before conversion
        if np.isnan(audio_float).any() or np.isinf(audio_float).any():
            # Replace NaNs/Infs with 0
            audio_float = np.nan_to_num(audio_float, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Clip to valid range to avoid overflow when scaling
        audio_float = np.clip(audio_float, -1.0, 1.0)
        
        audio_int16 = (audio_float.squeeze() * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def _tts_worker_loop(self):
        """从文本队列取句子，生成音频，并通过回调函数或音频队列分发。"""
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
                        self._save_session_audio()

                    # 在TTS工作线程完成时，通过回调发送结束信号
                    if self._chunk_callback:
                        self._chunk_callback(None)

                    if self._start_time:
                        total_duration = time.perf_counter() - self._start_time
                        monitor.log_metric("Total TTS session time", f"{total_duration:.3f}", "s")

                    self._tts_done_event.set()
                    self._session_speaker = None
                    self._session_prompt_audio = None
                    continue

                speaker = self._session_speaker or context.current_speaker
                prompt_audio = self._session_prompt_audio or context.current_prompt_audio
                language = self._session_language or context.current_language

                if not speaker or prompt_audio is None:
                    logger.error("Missing model or reference audio for the current session.")
                    continue

                gsv_model = model_manager.get(speaker)
                if not gsv_model:
                    logger.error("Failed to load model for current speaker.")
                    continue

                self.inference_engine.stop_event.clear()
                audio_chunk = self.inference_engine.tts(
                    text=sentence,
                    prompt_audio=prompt_audio,
                    encoder=gsv_model.T2S_ENCODER,
                    first_stage_decoder=gsv_model.T2S_FIRST_STAGE_DECODER,
                    stage_decoder=gsv_model.T2S_STAGE_DECODER,
                    vocoder=gsv_model.VITS,
                    prompt_encoder=gsv_model.PROMPT_ENCODER,
                    language=language,
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

                    # 使用回调函数处理流式数据
                    if self._chunk_callback:
                        audio_data = self._preprocess_for_playback(audio_chunk)
                        self._chunk_callback(audio_data)

            except Exception as e:
                logger.error(f"A critical error occurred while processing the TTS task: {e}", exc_info=True)
                # 发生错误时，也要确保发送结束信号
                if self._chunk_callback:
                    self._chunk_callback(None)
                self._tts_done_event.set()
                self._session_speaker = None
                self._session_prompt_audio = None

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
                    logger.error(f"A critical error occurred while playing audio: {e}", exc_info=True)
        finally:
            self.audio_engine.stop()

    # _play_with_pyaudio is no longer needed as AudioEngine handles playback
    # def _play_with_pyaudio(self, audio_chunk: np.ndarray):
    #     if self._pa is None:
    #         logger.info("Using PyAudio for playback fallback.")
    #         self._pa = pyaudio.PyAudio()
        
    #     if self._pa_stream is None:
    #         self._pa_stream = self._pa.open(
    #             format=pyaudio.paInt16,
    #             channels=self.channels,
    #             rate=self.sample_rate,
    #             output=True
    #         )
        
    #     audio_data = self._preprocess_for_playback(audio_chunk)
    #     self._pa_stream.write(audio_data)

    def _save_session_audio(self):
        try:
            # Flatten each chunk before concatenating (handles variable-length chunks from dynamic ONNX)
            flattened_chunks = [chunk.flatten() if chunk.ndim > 1 else chunk 
                              for chunk in self._session_audio_chunks]
            full_audio = np.concatenate(flattened_chunks, axis=0)
            with wave.open(self._current_save_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.bytes_per_sample)
                wf.setframerate(self.sample_rate)
                wf.writeframes(self._preprocess_for_playback(full_audio))
            logger.info(f"Audio successfully saved to {os.path.abspath(self._current_save_path)}")
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
        finally:
            self._session_audio_chunks = []
            self._current_save_path = None

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
                raise RuntimeError("A TTS session is already running. Please wait until it completes.")
            self._tts_done_event.clear()
            self._chunk_callback = chunk_callback
            self._stop_event.clear()

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
            resolved_language = (language or context.current_language or "ja").lower()
            if resolved_language not in {"ja", "en", "zh"}:
                resolved_language = "ja"
            self._session_language = resolved_language
            self._session_speaker = speaker or context.current_speaker
            self._session_prompt_audio = prompt_audio or context.current_prompt_audio
            if not self._session_speaker or self._session_prompt_audio is None:
                raise ValueError("Speaker and reference audio must be set before starting a TTS session.")
            
            # Pre-warm audio engine to eliminate cold-start stutter
            if play:
                self.audio_engine.warmup()


    def feed(self, text_chunk: str):
        with self._api_lock:
            if not text_chunk:
                return
            if self._start_time is None:
                self._start_time = time.perf_counter()

            if self._split:
                lang = self._session_language or context.current_language or 'ja'
                if lang == 'en':
                    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_chunk.strip()) if s.strip()]
                else:
                    sentences = split_japanese_text(text_chunk.strip())
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
            self._session_speaker = None
            self._session_prompt_audio = None
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
