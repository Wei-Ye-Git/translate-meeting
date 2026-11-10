"""Long sentence segmentation tests."""

import unittest

from translation_segmentation import split_long_text


def word_count_tokens(text):
    return len(text.split())


class TranslationSegmentationTests(unittest.TestCase):
    def test_splits_at_clause_boundaries_and_respects_token_budget(self):
        text = (
            "We reviewed the project plan, identified the delivery risks, "
            "and agreed on the next steps before the end of the meeting."
        )

        chunks = split_long_text(text, word_count_tokens, max_tokens=9)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(" ".join(chunks), text)
        self.assertTrue(all(word_count_tokens(chunk) <= 9 for chunk in chunks))

    def test_merges_short_clause_fragments_into_neighboring_chunks(self):
        text = "We reviewed all requirements, and then we approved the final implementation plan."

        chunks = split_long_text(text, word_count_tokens, max_tokens=7, min_words=4)

        self.assertTrue(all(len(chunk.split()) >= 4 for chunk in chunks))
        self.assertEqual(" ".join(chunks), text)

    def test_splits_unpunctuated_text_at_word_boundaries(self):
        text = "one two three four five six seven eight nine ten"

        chunks = split_long_text(text, word_count_tokens, max_tokens=5)

        self.assertEqual(chunks, ["one two three four five", "six seven eight nine ten"])

    def test_keeps_final_remainder_together_when_it_fits_budget(self):
        text = "one two three four five six seven eight nine ten eleven twelve"

        chunks = split_long_text(text, word_count_tokens, max_tokens=7)

        self.assertEqual(
            [len(chunk.split()) for chunk in chunks],
            [7, 5],
        )

    def test_keeps_short_inputs_as_one_translation_unit(self):
        self.assertEqual(split_long_text("Short sentence.", word_count_tokens), ["Short sentence."])


if __name__ == "__main__":
    unittest.main()
