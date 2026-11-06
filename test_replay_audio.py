import os
import tempfile
import unittest
import wave

import numpy as np

from replay_audio import build_replay_hotwords, iter_wav_chunks, write_jsonl_exclusive
from terminology import Terminology


class ReplayAudioTests(unittest.TestCase):
    def test_hotwords_include_only_recently_spoken_terms(self):
        terms = Terminology({
            "AI": {"target": "AI", "asr_always": True},
            "PR": {"target": "PR", "asr_always": True},
            "prompt": {"target": "提示", "asr_always": True},
            "ticket": {"target": "工单", "asr_always": True},
            "prompt engineering": {"target": "提示工程"},
            "PowerPoint": {"target": "PowerPoint"},
            "Teams": {"target": "Teams"},
        })

        hotwords = build_replay_hotwords(
            terms,
            ["Earlier note about prompt engineering.", "The speaker shares PowerPoint in Teams."],
        )

        self.assertEqual(hotwords, "prompt engineering, PowerPoint, Teams")
        self.assertEqual(
            build_replay_hotwords(terms, ["No configured terms are spoken here."]),
            None,
        )

    def test_reads_wav_as_timestamped_float_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "fixture.wav")
            samples = np.array([0, 16384, -16384, 32767, -32768], dtype="<i2")
            with wave.open(path, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(samples.tobytes())

            chunks = list(
                iter_wav_chunks(path, chunk_seconds=0.00025, overlap_seconds=0.0000625)
            )

        self.assertEqual([chunk[0] for chunk in chunks], [0.0, 3 / 16000])
        self.assertEqual([len(chunk[1]) for chunk in chunks], [4, 2])
        self.assertAlmostEqual(float(chunks[0][1][1]), 0.5, places=4)

    def test_refuses_to_overwrite_replay_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "replay.jsonl")
            with open(path, "w", encoding="utf-8") as output:
                output.write("keep")

            with self.assertRaises(FileExistsError):
                write_jsonl_exclusive(path, [])

            with open(path, encoding="utf-8") as source:
                self.assertEqual(source.read(), "keep")


if __name__ == "__main__":
    unittest.main()
