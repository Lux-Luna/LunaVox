import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import audio playback libraries
try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False

try:
    import pyaudio
    _PYAUDIO_AVAILABLE = True
except ImportError:
    _PYAUDIO_AVAILABLE = False


class AudioEngine:
    def __init__(self, sample_rate: int = 32000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.pa_instance: Optional[pyaudio.PyAudio] = None
        self.pa_stream: Optional[pyaudio.Stream] = None

    def play(self, audio_chunk: np.ndarray):
        """Play a chunk of audio using available backends."""
        if _PYAUDIO_AVAILABLE:
            self._play_with_pyaudio(audio_chunk)
        elif _SD_AVAILABLE:
            self._play_with_sounddevice(audio_chunk)
        else:
            logger.warning("No audio playback library (PyAudio or SoundDevice) available.")

    def _play_with_pyaudio(self, audio_chunk: np.ndarray):
        if self.pa_instance is None:
            self.pa_instance = pyaudio.PyAudio()
        
        if self.pa_stream is None:
            self.pa_stream = self.pa_instance.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True
            )
        
        # Ensure data is int16 for PyAudio
        if audio_chunk.dtype != np.int16:
            audio_chunk = (audio_chunk * 32767).astype(np.int16)
            
        self.pa_stream.write(audio_chunk.tobytes())

    def _play_with_sounddevice(self, audio_chunk: np.ndarray):
        try:
            sd.play(audio_chunk, self.sample_rate)
            # Note: sd.play is often non-blocking, but for streaming we might need sd.OutputStream
            # For simplicity in this extraction, we keep the basic logic.
        except Exception as e:
            logger.error(f"SoundDevice playback error: {e}")

    def stop(self):
        """Stop and cleanup audio resources."""
        if self.pa_stream:
            self.pa_stream.stop_stream()
            self.pa_stream.close()
            self.pa_stream = None
        if self.pa_instance:
            self.pa_instance.terminate()
            self.pa_instance = None
        if _SD_AVAILABLE:
            try:
                sd.stop()
            except:
                pass
