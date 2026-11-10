"""Whisper decoding parameter and prompt tests."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from asr import ASR, build_asr_prompt


class AsrDecodeTests(unittest.TestCase):
    def test_preview_skips_word_timestamps(self):
        asr = ASR.__new__(ASR)
        asr.model = MagicMock()
        asr.model.transcribe.return_value = ([], None)

        asr.transcribe(np.zeros(160, dtype=np.float32), word_timestamps=False)

        self.assertFalse(asr.model.transcribe.call_args.kwargs["word_timestamps"])

    def test_hotwords_are_passed_separately_from_prompt(self):
        asr = ASR.__new__(ASR)
        asr.model = MagicMock()
        asr.model.transcribe.return_value = ([], None)

        asr.transcribe(
            np.zeros(160, dtype=np.float32),
            initial_prompt="previous sentence",
            hotwords="pipeline, prompt engineering",
        )

        kwargs = asr.model.transcribe.call_args.kwargs
        self.assertEqual(kwargs["initial_prompt"], "previous sentence")
        self.assertEqual(kwargs["hotwords"], "pipeline, prompt engineering")

    def test_prompt_is_clipped_to_previous_context_budget(self):
        prompt = build_asr_prompt("x" * 400, "pipeline")

        self.assertLessEqual(len(prompt), 224)
        self.assertNotIn("pipeline", prompt)

    def test_prompt_collapses_multiline_context_before_applying_budget(self):
        prompt = build_asr_prompt("first sentence\n\nsecond   sentence")

        self.assertEqual(prompt, "first sentence second sentence")

    def test_transcribe_passes_anti_hallucination_parameters(self):
        asr = ASR.__new__(ASR)
        asr.model = MagicMock()
        asr.model.transcribe.return_value = ([], None)

        asr.transcribe(np.zeros(160, dtype=np.float32))

        kwargs = asr.model.transcribe.call_args.kwargs
        self.assertEqual(kwargs.get("repetition_penalty"), 1.1)
        self.assertEqual(kwargs.get("max_initial_timestamp"), 0.8)
