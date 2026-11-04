import numpy as np


class SileroSpeechDetector:
    def __init__(self, sampling_rate=16000, threshold=0.15):
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self._get_speech_timestamps = get_speech_timestamps
        self._options = VadOptions(
            threshold=threshold,
            min_speech_duration_ms=100,
            min_silence_duration_ms=200,
            speech_pad_ms=0,
        )

    def __call__(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        padding = (-len(audio)) % 512
        if padding:
            audio = np.pad(audio, (0, padding))
        speech = self._get_speech_timestamps(
            audio,
            vad_options=self._options,
            sampling_rate=self.sampling_rate,
        )
        return bool(speech)


def contains_voice(audio, energy_threshold, subchunk_frames, speech_detector=None):
    audio = np.asarray(audio, dtype=np.float32)
    has_energy = any(
        float(np.sqrt(np.mean(audio[start:start + subchunk_frames] ** 2)))
        >= energy_threshold
        for start in range(0, len(audio), subchunk_frames)
        if len(audio[start:start + subchunk_frames])
    )
    if has_energy:
        return True
    if speech_detector is None:
        return False
    try:
        return bool(speech_detector(audio))
    except Exception:
        return False
