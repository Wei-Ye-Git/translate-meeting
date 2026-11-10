"""Sentence boundary and context precision tests."""

import unittest

from text_processing import (
    TimedWord,
    merge_incomplete_sentences,
    words_to_text,
)
from config import Config
from translator import can_use_context_translation


class ContextPrecisionTests(unittest.TestCase):
    def test_merges_sentence_that_ends_with_a_continuation_word(self):
        sentences = [
            [
                TimedWord(0.0, 0.2, "It"),
                TimedWord(0.2, 0.4, "is"),
                TimedWord(0.4, 0.6, "very."),
            ],
            [
                TimedWord(0.6, 0.8, "important"),
                TimedWord(0.8, 1.0, "to"),
                TimedWord(1.0, 1.2, "protect."),
            ],
        ]

        result = merge_incomplete_sentences(sentences)

        self.assertEqual(len(result), 1)
        self.assertEqual(words_to_text(result[0]), "It is very. important to protect.")

    def test_keeps_a_short_complete_acknowledgement_separate(self):
        sentences = [
            [TimedWord(0.0, 0.2, "Okay.")],
            [TimedWord(0.2, 0.4, "Next"), TimedWord(0.4, 0.6, "step.")],
        ]

        result = merge_incomplete_sentences(sentences)

        self.assertEqual(len(result), 2)

    def test_short_sentences_use_batch_translation_alignment(self):
        texts = ["Okay.", "The next step.", "Please continue."]

        self.assertFalse(can_use_context_translation(texts, texts))

    def test_long_sentences_can_use_context_candidate(self):
        texts = [
            "The migration feature is available in the new document viewer.",
            "Please use the new document view for all future revisions.",
        ]

        self.assertTrue(can_use_context_translation(texts, ["a", "b"]))

    def test_long_translation_context_can_be_checked_before_alignment(self):
        texts = [
            "The migration feature is available in the new document viewer.",
            "Please use the new document view for all future revisions.",
        ]

        self.assertTrue(can_use_context_translation(texts, []))

    def test_complete_mode_uses_more_context_for_asr(self):
        self.assertGreater(Config().asr_beam_size, Config(mode="low_latency").asr_beam_size)


if __name__ == "__main__":
    unittest.main()
