"""Stable live-preview text tests."""

import unittest

from stable_text import StableText


class StableTextTests(unittest.TestCase):
    def test_commits_only_the_prefix_shared_by_two_consecutive_hypotheses(self):
        stable = StableText()

        self.assertEqual(stable.update("The software scan is"), "")
        self.assertEqual(stable.update("The software scan is running"), "The software scan is")

    def test_emits_only_newly_stable_words(self):
        stable = StableText()
        stable.update("The software scan is")

        self.assertEqual(stable.update("The software scan is running"), "The software scan is")
        self.assertEqual(stable.update("The software scan is running today"), "running")

    def test_reset_starts_a_new_utterance(self):
        stable = StableText()
        stable.update("First sentence ends")
        stable.update("First sentence ends now")
        stable.reset()

        self.assertEqual(stable.update("Next sentence starts"), "")

    def test_punctuation_revision_does_not_block_later_stable_words(self):
        stable = StableText()
        stable.update("The software scan is")
        self.assertEqual(stable.update("The software scan is running"), "The software scan is")

        self.assertEqual(
            stable.update("The software scan is running. It found issues"),
            "running.",
        )

    def test_unchanged_hypothesis_stabilizes_once_without_recommitting(self):
        stable = StableText()
        stable.update("The scan finished")

        self.assertEqual(stable.update("The scan finished today"), "The scan finished")
        self.assertEqual(stable.update("The scan finished today"), "today")
        self.assertEqual(stable.update("The scan finished today"), "")


if __name__ == "__main__":
    unittest.main()
