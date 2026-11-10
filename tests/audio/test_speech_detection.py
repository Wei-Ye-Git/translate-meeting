"""Voice activity and energy gate tests."""

import unittest

import numpy as np

from speech_detection import SileroSpeechDetector, contains_voice


class SpeechDetectionTests(unittest.TestCase):
    def test_silero_detector_pads_partial_audio_to_model_window_size(self):
        class SpeechDetector(SileroSpeechDetector):
            def __init__(self):
                self.sampling_rate = 16000
                self._options = object()
                self.received_audio = None

            def _get_speech_timestamps(self, audio, **kwargs):
                self.received_audio = audio
                return [{"start": 0, "end": 512}]

        detector = SpeechDetector()

        self.assertTrue(detector(np.ones(1600, dtype=np.float32)))
        self.assertEqual(detector.received_audio.shape, (2048,))

    def test_silero_detection_can_recover_quiet_speech_below_energy_gate(self):
        audio = np.full(1600, 0.0001, dtype=np.float32)
        detector_calls = []

        is_voice = contains_voice(
            audio,
            energy_threshold=0.001,
            subchunk_frames=1600,
            speech_detector=lambda samples: detector_calls.append(samples) or True,
        )

        self.assertTrue(is_voice)
        self.assertEqual(len(detector_calls), 1)

    def test_silence_is_rejected_when_energy_and_silero_find_no_speech(self):
        audio = np.zeros(1600, dtype=np.float32)

        is_voice = contains_voice(
            audio,
            energy_threshold=0.001,
            subchunk_frames=1600,
            speech_detector=lambda samples: False,
        )

        self.assertFalse(is_voice)

    def test_energy_gate_still_accepts_speech_without_a_vad_detector(self):
        audio = np.full(1600, 0.01, dtype=np.float32)

        self.assertTrue(
            contains_voice(
                audio,
                energy_threshold=0.001,
                subchunk_frames=1600,
            )
        )

    def test_vad_error_falls_back_to_energy_gate(self):
        audio = np.zeros(1600, dtype=np.float32)

        self.assertFalse(
            contains_voice(
                audio,
                energy_threshold=0.001,
                subchunk_frames=1600,
                speech_detector=lambda samples: (_ for _ in ()).throw(
                    RuntimeError("VAD unavailable")
                ),
            )
        )

if __name__ == "__main__":
    unittest.main()
