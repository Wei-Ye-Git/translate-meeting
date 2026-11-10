"""Transcript quality, anomaly, and recordability filter tests."""

import unittest

from text_processing import (
    TimedWord,
    asr_output_issue,
    clean_repetitions,
    is_asr_anomaly,
    is_meaningful_sentence,
    is_recordable_sentence,
    is_review_candidate,
    is_sentence_complete,
)


class TranscriptQualityFilterTests(unittest.TestCase):
    def test_asr_output_issue_flags_implausible_word_rate(self):
        words = [TimedWord(i * 0.05, i * 0.05 + 0.04, f"word{i}") for i in range(60)]
        self.assertEqual(asr_output_issue(words, audio_duration=5.0), "implausible_word_rate")

    def test_asr_output_issue_keeps_fast_but_plausible_speech(self):
        words = [TimedWord(i * 0.1, i * 0.1 + 0.08, f"word{i}") for i in range(30)]
        self.assertIsNone(asr_output_issue(words, audio_duration=5.0))

    def test_asr_word_rate_uses_speech_span_not_silent_window_padding(self):
        words = [TimedWord(10 + i * 0.05, 10.04 + i * 0.05, f"word{i}") for i in range(20)]
        self.assertEqual(asr_output_issue(words), "implausible_word_rate")

    def test_is_meaningful_sentence_accepts_normal_words(self):
        from text_processing import is_meaningful_sentence
        self.assertTrue(is_meaningful_sentence("The deployment plan is ready."))

    def test_clean_repetitions_removes_comma_separated_repeats(self):
        self.assertEqual(clean_repetitions("yes, yes, yes, we can do it"), "yes, we can do it")

    def test_rejects_single_word_asr_fragment_from_history(self):
        self.assertFalse(is_recordable_sentence("deployment"))

    def test_keeps_short_complete_acknowledgements(self):
        self.assertTrue(is_recordable_sentence("Thanks."))

    def test_only_marks_short_unpunctuated_non_acknowledgements_for_review(self):
        self.assertTrue(is_review_candidate("Maybe later"))
        self.assertFalse(is_review_candidate("Thank you."))

    def test_detects_extreme_repeated_character_as_asr_anomaly(self):
        self.assertTrue(is_asr_anomaly("a" * 20))

    def test_accepts_reasonably_long_stretched_word(self):
        self.assertFalse(is_asr_anomaly("Nooooo"))

    def test_detects_repeated_phrase_loop_as_asr_anomaly(self):
        self.assertTrue(is_asr_anomaly("we should review the final release candidate we should review the final release candidate we should review the final release candidate"))

    def test_detects_repeated_token_loop_inside_longer_output(self):
        self.assertTrue(is_asr_anomaly("the project is ready ready ready ready ready ready ready ready"))

    def test_detects_single_token_loop_inside_longer_output(self):
        self.assertTrue(is_asr_anomaly("we should review deployment deployment deployment deployment deployment deployment deployment deployment"))

    def test_rejects_sentence_with_zero_duration_asr_timestamps(self):
        words = [TimedWord(1, 1, "The"), TimedWord(1, 1, "deployment")]
        self.assertFalse(is_sentence_complete(words))

    def test_accepts_sentence_with_positive_audio_span(self):
        self.assertTrue(is_sentence_complete([TimedWord(1, 1.2, "Thanks." )]))

if __name__ == "__main__":
    unittest.main()
