import unittest

import numpy as np

from speech_quality import (
    analyze_audio_window,
    analyze_timed_utterance,
    classify_segmentation_boundaries,
)
from text_processing import TimedWord


class SpeechQualityTests(unittest.TestCase):
    def test_audio_window_metrics_include_level_peak_and_clipping(self):
        metrics = analyze_audio_window(
            np.asarray([0.0, 0.5, -1.0, 0.25], dtype=np.float32)
        )

        self.assertEqual(metrics["sample_count"], 4)
        self.assertEqual(metrics["peak"], 1.0)
        self.assertEqual(metrics["clipped_fraction"], 0.25)
        self.assertGreater(metrics["rms"], 0)

    def test_empty_audio_window_has_zeroed_metrics(self):
        self.assertEqual(
            analyze_audio_window(np.empty(0, dtype=np.float32)),
            {"sample_count": 0, "rms": 0.0, "peak": 0.0, "clipped_fraction": 0.0},
        )

    def test_detects_fast_speech_and_low_confidence_terms(self):
        words = [
            TimedWord(index * 0.12, (index + 1) * 0.12, text, probability)
            for index, (text, probability) in enumerate([
                ("We", 0.95), ("need", 0.91), ("review", 0.42),
                ("all", 0.92), ("changes", 0.90), ("before", 0.9),
                ("release", 0.91), ("today.", 0.93),
            ])
        ]

        result = analyze_timed_utterance(words, audio_duration=1.0)

        self.assertIn("possible_fast_speech", result["issues"])
        self.assertIn("low_confidence_words", result["issues"])
        self.assertEqual(result["low_confidence_words"][0]["text"], "review")

    def test_fast_speech_threshold_is_tunable_for_slow_meeting_speakers(self):
        words = [
            TimedWord(index * 0.3, (index + 1) * 0.3, f"word{index}", 0.9)
            for index in range(10)
        ]

        result = analyze_timed_utterance(
            words, audio_duration=3.0, fast_words_per_second=3.0
        )

        self.assertIn("possible_fast_speech", result["issues"])

    def test_detects_filler_heavy_speech_and_internal_pauses(self):
        words = [
            TimedWord(0.0, 0.2, "Um", 0.8),
            TimedWord(0.2, 0.4, "we", 0.9),
            TimedWord(0.4, 0.7, "should", 0.9),
            TimedWord(1.8, 2.0, "you", 0.9),
            TimedWord(2.0, 2.3, "know", 0.9),
            TimedWord(3.5, 3.8, "uh", 0.7),
            TimedWord(3.8, 4.0, "continue.", 0.9),
        ]

        result = analyze_timed_utterance(words, audio_duration=4.0)

        self.assertIn("filler_heavy", result["issues"])
        self.assertGreaterEqual(result["long_pause_count"], 2)
        self.assertEqual(result["filler_count"], 3)

    def test_sparse_asr_is_reported_as_possible_underrecognition(self):
        words = [TimedWord(0.1, 0.4, "Thanks.", 0.9)]

        result = analyze_timed_utterance(words, audio_duration=4.0)

        self.assertIn("possible_sparse_asr", result["issues"])

    def test_marks_unpunctuated_pause_split_as_candidate_boundary(self):
        left = [TimedWord(0.0, 0.3, "We"), TimedWord(0.3, 0.6, "need")]
        right = [TimedWord(1.7, 2.0, "to"), TimedWord(2.0, 2.3, "continue.")]

        boundaries = classify_segmentation_boundaries([left, right])

        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0]["left_text"], "We need")
        self.assertEqual(boundaries[0]["pause_seconds"], 1.1)

    def test_ignores_punctuated_sentence_boundary(self):
        left = [TimedWord(0.0, 0.3, "We"), TimedWord(0.3, 0.6, "finished.")]
        right = [TimedWord(1.7, 2.0, "Next"), TimedWord(2.0, 2.3, "topic.")]

        self.assertEqual(classify_segmentation_boundaries([left, right]), [])

    def test_fillers_can_be_reported_without_cleaning_the_recognized_text(self):
        words = [
            TimedWord(0.0, 0.2, "Um"),
            TimedWord(0.2, 0.4, "we"),
            TimedWord(0.4, 0.6, "can"),
            TimedWord(0.6, 0.8, "actually"),
            TimedWord(0.8, 1.0, "do"),
            TimedWord(1.0, 1.2, "that."),
        ]

        result = analyze_timed_utterance(words, audio_duration=1.2)

        self.assertIn("Um", " ".join(word.text for word in words))
        self.assertEqual(result["filler_count"], 2)

if __name__ == "__main__":
    unittest.main()
