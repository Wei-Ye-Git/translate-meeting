import unittest
import numpy as np
from text_processing import (
    TimedWord,
    words_to_text,
    is_sentence_complete,
    clean_repetitions,
)


class DynamicChunkerTests(unittest.TestCase):
    def test_complete_sentence_recognized_as_complete(self):
        words = [
            TimedWord(0, 0.4, "We"),
            TimedWord(0.4, 0.8, "need"),
            TimedWord(0.8, 1.2, "to"),
            TimedWord(1.2, 1.6, "check"),
            TimedWord(1.6, 2.0, "the"),
            TimedWord(2.0, 2.4, "database"),
            TimedWord(2.4, 2.8, "before"),
            TimedWord(2.8, 3.2, "we"),
            TimedWord(3.2, 3.6, "deploy."),
        ]
        self.assertTrue(is_sentence_complete(words, min_words=7))
        text = words_to_text(words)
        self.assertEqual(text, "We need to check the database before we deploy.")

    def test_short_incomplete_fragment_not_recognized_as_complete(self):
        words = [
            TimedWord(0, 0.4, "We"),
            TimedWord(0.4, 0.8, "need"),
            TimedWord(0.8, 1.2, "to."),
        ]
        self.assertFalse(is_sentence_complete(words, min_words=7))

    def test_short_standalone_acknowledgement_is_complete(self):
        words = [
            TimedWord(0, 0.4, "All"),
            TimedWord(0.4, 0.8, "right."),
        ]
        self.assertTrue(is_sentence_complete(words, min_words=7))

        okay_words = [TimedWord(0, 0.4, "Okay.")]
        self.assertTrue(is_sentence_complete(okay_words, min_words=7))


if __name__ == "__main__":
    unittest.main()
