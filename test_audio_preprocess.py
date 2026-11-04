import unittest

import numpy as np

from audio_preprocess import highpass


class AudioPreprocessTests(unittest.TestCase):
    def test_highpass_removes_constant_offset_and_keeps_length(self):
        samples = np.full(1600, 0.2, dtype=np.float32)
        samples[800:] = 0.2 + 0.05 * np.sin(np.linspace(0, 40, 800)).astype(np.float32)

        filtered = highpass(samples, samplerate=16000, cutoff_hz=80)

        self.assertEqual(len(filtered), len(samples))
        self.assertLess(abs(float(filtered[:400].mean())), 0.02)
        self.assertGreater(float(np.max(np.abs(filtered[800:]))), 0.01)
