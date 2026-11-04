import os
import wave

import numpy as np


def create_session_directory(root_dir, timestamp, prefix="meeting_"):
    path = os.path.join(root_dir, f"{prefix}{timestamp}")
    os.makedirs(path, exist_ok=False)
    return path


class AudioRecorder:
    def __init__(self, path, samplerate):
        self.path = path
        self.samplerate = samplerate
        self._wave = wave.open(path, "wb")
        self._wave.setnchannels(1)
        self._wave.setsampwidth(2)
        self._wave.setframerate(samplerate)
        self._closed = False
        self.frames_written = 0

    def write(self, audio):
        if self._closed:
            raise ValueError("Audio recorder is closed")
        samples = np.asarray(audio, dtype=np.float32)
        pcm = np.clip(samples, -1.0, 1.0)
        pcm = (pcm * 32767).astype("<i2")
        self._wave.writeframesraw(pcm.tobytes())
        self.frames_written += len(pcm)

    def write_silence(self, frames):
        if self._closed:
            raise ValueError("Audio recorder is closed")
        remaining = max(0, int(frames))
        silence = np.zeros(min(remaining, self.samplerate), dtype="<i2")
        while remaining:
            count = min(remaining, len(silence))
            self._wave.writeframesraw(silence[:count].tobytes())
            self.frames_written += count
            remaining -= count

    def close(self):
        if not self._closed:
            self._wave.close()
            self._closed = True
