"""Word timing, overlap, and correction validation tests."""

import unittest

import numpy as np

from audio_capture import AudioChunk, extract_audio_range
from text_processing import (
    TimedWord,
    correction_is_confident,
    correction_is_bounded,
    correction_matches_target_range,
    has_time_overlap,
    correction_is_usable,
    is_asr_anomaly,
    is_recordable_sentence,
    remove_chunk_overlap_words,
    remove_sentence_overlap,
    remove_sentence_overlap_words,
    split_timed_sentences,
    words_in_time_range,
    words_before,
    words_to_text,
    shift_word_times,
    asr_output_issue,
    has_valid_time_span,
    select_timed_words_for_target,
    select_timed_words_for_correction,
    should_schedule_correction,
    should_finalize_waiting_continuation,
    finalize_pending_group,
    trim_trailing_silence_chunks,
    filter_uncommitted_words,
    should_hold_after_waiting_silence,
)


class TimedTranscriptTests(unittest.TestCase):
    def test_chunk_overlap_removes_only_duplicate_leading_words(self):
        previous = [
            TimedWord(0.0, 0.2, "we"),
            TimedWord(0.2, 0.4, "need"),
            TimedWord(0.4, 0.6, "tests"),
        ]
        current = [
            TimedWord(3.0, 3.2, "need"),
            TimedWord(3.2, 3.4, "tests"),
            TimedWord(3.4, 3.6, "today."),
        ]

        result = remove_chunk_overlap_words(previous, current)

        self.assertEqual(words_to_text(result), "today.")
        self.assertEqual(result[0].start, 3.4)

    def test_sentence_split_keeps_word_timing_for_each_sentence(self):
        words = [
            TimedWord(0.0, 0.2, "First"),
            TimedWord(0.2, 0.4, "sentence."),
            TimedWord(0.4, 0.6, "Second"),
            TimedWord(0.6, 0.8, "sentence."),
        ]

        completed, remaining = split_timed_sentences(words)

        self.assertEqual([words_to_text(item) for item in completed], [
            "First sentence.",
            "Second sentence.",
        ])
        self.assertEqual(remaining, [])
        self.assertEqual(completed[0][0].start, 0.0)
        self.assertEqual(completed[1][-1].end, 0.8)

    def test_audio_range_is_sliced_from_the_sentence_time_span(self):
        chunks = [
            AudioChunk(0, np.arange(10, dtype=np.float32)),
            AudioChunk(10, np.arange(10, 20, dtype=np.float32)),
        ]

        result = extract_audio_range(chunks, 4, 16)

        np.testing.assert_array_equal(result, np.arange(4, 16, dtype=np.float32))

    def test_audio_range_inserts_silence_for_missing_frames(self):
        chunks = [
            AudioChunk(0, np.ones(4, dtype=np.float32)),
            AudioChunk(8, np.full(4, 2, dtype=np.float32)),
        ]
        result = extract_audio_range(chunks, 0, 12)
        np.testing.assert_array_equal(result, [1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2])

    def test_words_before_boundary_keeps_corrected_sentence_only(self):
        words = [
            TimedWord(0.0, 0.3, "The"),
            TimedWord(0.3, 0.6, "markup"),
            TimedWord(0.6, 0.9, "is"),
            TimedWord(1.0, 1.3, "updated"),
        ]

        result = words_before(words, 0.95)

        self.assertEqual(words_to_text(result), "The markup is")

    def test_context_words_are_limited_to_the_target_time_range(self):
        words = [
            TimedWord(0.0, 0.3, "before"),
            TimedWord(1.0, 1.3, "target"),
            TimedWord(1.4, 1.7, "sentence."),
            TimedWord(2.0, 2.3, "after"),
        ]

        result = words_in_time_range(words, 0.8, 1.8)

        self.assertEqual(words_to_text(result), "target sentence.")

    def test_context_transcription_timestamps_can_be_shifted_to_meeting_timeline(self):
        shifted = shift_word_times([TimedWord(0.2, 0.5, "target")], 12.0)

        self.assertEqual((shifted[0].start, shifted[0].end), (12.2, 12.5))

    def test_short_correction_cannot_replace_a_complete_sentence(self):
        original = "The markup migration feature is available in the new document viewer."

        self.assertFalse(correction_is_usable(original, "migration feature"))

    def test_partial_correction_cannot_replace_a_three_word_sentence(self):
        self.assertFalse(correction_is_usable("The feature works.", "feature"))

    def test_correction_cannot_drop_any_words_from_original(self):
        self.assertFalse(
            correction_is_usable(
                "I wanted him to enjoy his stay.",
                "I wanted him to enjoy.",
            )
        )

    def test_short_complete_correction_remains_usable(self):
        self.assertTrue(correction_is_usable("I see", "I see"))

    def test_correction_cannot_replace_sentence_with_unrelated_context(self):
        self.assertFalse(
            correction_is_usable(
                "No major issues with AI.",
                "We have already used AI at work and learned from colleagues.",
            )
        )

    def test_correction_rejects_low_content_overlap(self):
        self.assertFalse(
            correction_is_usable(
                "I had another question.",
                "Tomorrow we will have a meeting.",
            )
        )

    def test_correction_keeps_a_high_overlap_phonetic_repair(self):
        self.assertTrue(
            correction_is_confident(
                "That leads to bad code, that leads to bucks.",
                "That leads to bad code, that leads to bugs.",
                [0.72] * 8,
                [0.84] * 8,
            )
        )

    def test_confidence_gate_rejects_lower_confidence_rewrite(self):
        self.assertFalse(
            correction_is_confident(
                "The work and the verifying.",
                "The work and the very fine.",
                [0.82] * 5,
                [0.61] * 6,
            )
        )

    def test_confidence_gate_allows_transcript_when_wording_is_unchanged(self):
        self.assertTrue(
            correction_is_confident(
                "We use AI every day.",
                "we use AI every day.",
                [0.8] * 5,
                [0.7] * 5,
            )
        )

    def test_recorded_correction_keeps_sentence_when_replay_misses_its_ending(self):
        self.assertFalse(
            correction_is_usable(
                "If it changes 200 lines of code, then the first iteration trying "
                "to understand something, it was obviously wrong.",
                "If it changes 200 lines of codes in the first iteration trying "
                "to understand something it was obviously",
            )
        )

    def test_recorded_correction_keeps_fragment_when_replay_misses_its_start(self):
        self.assertFalse(
            correction_is_usable(
                "is understanding whatever you're telling him before you write "
                "a single line of code.",
                "whatever you're telling him before you write a single line of code.",
            )
        )

    def test_recorded_context_correction_accepts_matching_sentence(self):
        self.assertTrue(
            correction_is_usable(
                "With this plan mode, we will solve exactly that problem.",
                "With this plan mode, we will solve exactly that problem.",
            )
        )

    def test_correction_allows_rewording_with_shared_content(self):
        self.assertTrue(
            correction_is_usable(
                "We use AI daily to generate code.",
                "We use AI every day to generate code.",
            )
        )

    def test_correction_rejects_context_expansion_far_beyond_target(self):
        original = "They know the business."
        corrected = "They know the business and have worked with the company for decades."

        self.assertFalse(correction_is_bounded(original, corrected))

    def test_correction_rejects_words_outside_target_time_window(self):
        words = [TimedWord(9.0, 9.3, "context"), TimedWord(10.0, 10.3, "target.")]

        self.assertFalse(correction_matches_target_range(words, 10.0, 10.3))

    def test_correction_accepts_words_within_target_time_window(self):
        words = [TimedWord(9.9, 10.1, "Target"), TimedWord(10.1, 10.3, "sentence.")]

        self.assertTrue(correction_matches_target_range(words, 10.0, 10.3))

    def test_correction_allows_small_contextual_expansion(self):
        self.assertTrue(
            correction_is_bounded(
                "They know the business.",
                "They really know the business.",
            )
        )

    def test_changed_correction_requires_material_confidence_gain(self):
        self.assertFalse(
            correction_is_confident(
                "RIB sold the SAA business.",
                "RIB sold the SAA distance.",
                [0.80] * 6,
                [0.85] * 6,
            )
        )

    def test_changed_word_requires_its_own_confidence_support(self):
        self.assertFalse(
            correction_is_confident(
                "RIB sold the SAA business.",
                "RIB sold the SAA distance.",
                [0.99, 0.99, 0.99, 0.99, 0.99, 0.95],
                [0.99, 0.99, 0.99, 0.99, 0.99, 0.72],
            )
        )

    def test_sentence_with_terminal_punctuation_is_not_incomplete(self):
        from text_processing import is_likely_incomplete
        self.assertFalse(is_likely_incomplete([TimedWord(0, 1, "for.")]))

    def test_complete_correction_can_replace_the_original_sentence(self):
        original = "The markup migration feature is available in the new document viewer."

        self.assertTrue(
            correction_is_usable(
                original,
                "The markup migration feature is available in the new document viewer.",
            )
        )

    def test_chunk_overlap_matches_across_trailing_punctuation(self):
        previous = [
            TimedWord(0.0, 0.2, "our"),
            TimedWord(0.2, 0.4, "server."),
        ]
        current = [
            TimedWord(1.0, 1.2, "our"),
            TimedWord(1.2, 1.4, "server"),
            TimedWord(1.4, 1.6, "is"),
            TimedWord(1.6, 1.8, "running."),
        ]

        result = remove_chunk_overlap_words(previous, current)
        self.assertEqual(words_to_text(result), "is running.")

    def test_split_timed_sentences_on_natural_pause(self):
        words = [
            TimedWord(0.0, 0.3, "We"),
            TimedWord(0.3, 0.6, "need"),
            TimedWord(0.6, 0.9, "this"),
            # Gap of 1.1s (2.0 - 0.9)
            TimedWord(2.0, 2.3, "Next"),
            TimedWord(2.3, 2.6, "topic"),
        ]

        completed, remaining = split_timed_sentences(words)
        self.assertEqual(len(completed), 1)
        self.assertEqual(words_to_text(completed[0]), "We need this")
        self.assertEqual(words_to_text(remaining), "Next topic")

    def test_keeps_short_unpunctuated_complete_phrase(self):
        self.assertTrue(is_recordable_sentence("I see"))

    def test_removes_repeated_leading_words_between_adjacent_records(self):
        self.assertEqual(
            remove_sentence_overlap(
                "But the question that we want to answer.",
                "But the question that we want to answer through performance testing is...",
            ),
            "through performance testing is...",
        )

    def test_does_not_remove_short_common_phrase_overlap(self):
        self.assertEqual(
            remove_sentence_overlap("Thank you.", "Thank you for joining us."),
            "Thank you for joining us.",
        )

    def test_removes_repeated_word_prefix_without_losing_timestamps(self):
        previous = [TimedWord(i, i + 0.2, word) for i, word in enumerate(
            "the question that we want to answer".split()
        )]
        current = [TimedWord(10 + i, 10.2 + i, word) for i, word in enumerate(
            "the question that we want to answer through testing".split()
        )]

        result = remove_sentence_overlap_words(previous, current)

        self.assertEqual(words_to_text(result), "through testing")
        self.assertEqual(result[0].start, 17)

    def test_overlap_filter_accepts_previous_history_text(self):
        current = [TimedWord(10 + i, 10.2 + i, word) for i, word in enumerate(
            "the question that we want to answer through testing".split()
        )]

        result = remove_sentence_overlap_words(
            "The question that we want to answer.", current
        )

        self.assertEqual(words_to_text(result), "through testing")

    def test_overlap_deduplication_requires_overlapping_time_ranges(self):
        self.assertTrue(has_time_overlap(12.0, 11.9))
        self.assertFalse(has_time_overlap(12.0, 12.0))
        self.assertFalse(has_time_overlap(12.0, 12.2))

    def test_detects_repeated_phrase_longer_than_ten_words(self):
        phrase = "we can see if we look at the sum of SAST issues"

        self.assertTrue(is_asr_anomaly((phrase + " ") * 8))

    def test_rejects_sentence_with_zero_duration_asr_timestamps(self):
        words = [TimedWord(12.0, 12.0, "Thank"), TimedWord(12.0, 12.0, "you.")]

        self.assertFalse(has_valid_time_span(words))

    def test_accepts_sentence_with_positive_audio_span(self):
        words = [TimedWord(12.0, 12.2, "Thank"), TimedWord(12.2, 12.5, "you.")]

        self.assertTrue(has_valid_time_span(words))

    def test_drops_word_timestamp_stretched_across_silent_audio_tail(self):
        words = [
            TimedWord(5481.4, 5481.8, "Thank"),
            TimedWord(5481.8, 5494.9, "you."),
            TimedWord(5494.9, 5494.9, "Bye."),
        ]

        result = select_timed_words_for_target(words, 5480.0, 5483.0)

        self.assertEqual(words_to_text(result), "Thank")
        self.assertTrue(has_valid_time_span(result))

    def test_correction_keeps_words_with_midpoints_near_target_boundaries(self):
        words = [
            TimedWord(9.8, 10.2, "previous."),
            TimedWord(10.2, 10.5, "Target"),
            TimedWord(10.5, 10.9, "sentence."),
            TimedWord(10.95, 11.15, "Next"),
        ]

        result = select_timed_words_for_correction(words, 10.2, 10.9)

        self.assertEqual(words_to_text(result), "previous. Target sentence. Next")

    def test_correction_boundary_padding_excludes_distant_neighbor_words(self):
        words = [
            TimedWord(9.3, 9.7, "earlier."),
            TimedWord(9.8, 10.2, "previous."),
            TimedWord(10.2, 10.5, "Target"),
            TimedWord(10.5, 10.9, "sentence."),
            TimedWord(10.95, 11.15, "Next"),
            TimedWord(11.3, 11.6, "later."),
        ]

        result = select_timed_words_for_correction(words, 10.2, 10.9)

        self.assertEqual(words_to_text(result), "previous. Target sentence. Next")

    def test_correction_rejects_invalid_word_timestamps(self):
        words = [
            TimedWord(10.0, 10.0, "invalid"),
            TimedWord(10.0, 15.1, "stretched"),
            TimedWord(10.0, 10.2, "valid."),
        ]

        result = select_timed_words_for_correction(words, 10.0, 10.3)

        self.assertEqual(words_to_text(result), "valid.")

    def test_context_correction_skips_short_fragment(self):
        words = [TimedWord(10.0, 10.2, "Thanks.")]

        self.assertFalse(should_schedule_correction(words))

    def test_context_correction_runs_for_substantive_sentence(self):
        words = [
            TimedWord(10.0, 10.25, "We"),
            TimedWord(10.25, 10.5, "use"),
            TimedWord(10.5, 10.75, "AI"),
            TimedWord(10.75, 11.0, "daily."),
        ]

        self.assertTrue(should_schedule_correction(words))

    def test_context_correction_skips_high_confidence_sentence(self):
        words = [
            TimedWord(10.0, 10.25, "We", 0.96),
            TimedWord(10.25, 10.5, "use", 0.97),
            TimedWord(10.5, 10.75, "AI", 0.98),
            TimedWord(10.75, 11.0, "daily.", 0.96),
        ]

        self.assertFalse(should_schedule_correction(words))

    def test_context_correction_keeps_low_confidence_sentence(self):
        words = [
            TimedWord(10.0, 10.25, "We", 0.96),
            TimedWord(10.25, 10.5, "use", 0.97),
            TimedWord(10.5, 10.75, "AI", 0.58),
            TimedWord(10.75, 11.0, "daily.", 0.96),
        ]

        self.assertTrue(should_schedule_correction(words))

    def test_finalizes_unfinished_tail_after_silence_while_waiting_for_continuation(self):
        words = [
            TimedWord(4.0, 4.2, "the"),
            TimedWord(4.2, 4.5, "project"),
            TimedWord(4.5, 4.7, "is"),
            TimedWord(4.7, 5.0, "ready."),
        ]

        self.assertTrue(should_finalize_waiting_continuation(words))

    def test_waiting_continuation_keeps_a_grammatically_incomplete_tail_open(self):
        words = [TimedWord(4.0, 4.2, "and")]

        self.assertFalse(should_finalize_waiting_continuation(words))

    def test_does_not_finalize_short_acknowledgement_fragment_while_waiting(self):
        words = [TimedWord(4.0, 4.2, "Thank")]

        self.assertFalse(finalize_pending_group(words, waiting_for_continuation=True))

    def test_finalizes_complete_sentence_while_waiting(self):
        words = [TimedWord(4.0, 4.2, "Thank"), TimedWord(4.2, 4.5, "you.")]

        self.assertTrue(finalize_pending_group(words, waiting_for_continuation=True))

    def test_trims_all_trailing_silence_even_when_only_silence_chunks_remain(self):
        chunks = ["last speech", "silence 1", "silence 2", "silence 3"]

        self.assertEqual(trim_trailing_silence_chunks(chunks, 3), ["last speech"])

    def test_trims_entire_buffer_when_waiting_buffer_is_only_silence(self):
        chunks = ["silence 1", "silence 2"]

        self.assertEqual(trim_trailing_silence_chunks(chunks, 2), [])

    def test_audio_tail_uses_only_words_after_the_committed_prefix(self):
        words = [
            TimedWord(1.0, 1.2, "Thank"),
            TimedWord(1.2, 1.4, "you."),
            TimedWord(1.8, 2.0, "Bye."),
        ]

        result = filter_uncommitted_words(words, committed_until=1.5)

        self.assertEqual(words_to_text(result), "Bye.")

    def test_waits_for_more_silence_when_pending_tail_is_incomplete(self):
        self.assertTrue(
            should_hold_after_waiting_silence(
                [TimedWord(1.0, 1.2, "Thank")],
                silence_duration=0.5,
                required_silence=0.8,
            )
        )

    def test_does_not_hold_completed_acknowledgement(self):
        self.assertFalse(
            should_hold_after_waiting_silence(
                [TimedWord(1.0, 1.2, "Thank"), TimedWord(1.2, 1.5, "you.")],
                silence_duration=0.9,
                required_silence=0.8,
            )
        )

    def test_does_not_hold_unrecordable_greeting_fragment(self):
        self.assertFalse(
            should_hold_after_waiting_silence(
                [TimedWord(1.0, 1.2, "Bye.")],
                silence_duration=0.9,
                required_silence=0.8,
            )
        )

    def test_finalizes_short_complete_thanks_while_waiting_for_continuation(self):
        words = [TimedWord(1.0, 1.2, "Thank"), TimedWord(1.2, 1.4, "you")]

        self.assertTrue(should_finalize_waiting_continuation(words))


    def test_does_not_flag_natural_speech_repetition(self):
        self.assertFalse(is_asr_anomaly("I really, really want to show you this."))
        self.assertFalse(is_asr_anomaly("We will test it, test it again, and verify."))
        self.assertFalse(is_asr_anomaly("Thank you. Bye. Thank you. Bye."))

if __name__ == "__main__":
    unittest.main()
