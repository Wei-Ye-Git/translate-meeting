"""Persistent translation review reason lifecycle tests."""

import unittest

from translator import (
    apply_translation_quality_status,
    clear_review_reason,
    set_review_reason,
)


class TranslationQualityStatusTests(unittest.TestCase):
    def test_persistent_translation_quality_issues_are_marked_for_review(self):
        status = apply_translation_quality_status({"version": 2}, ["missing_numbers"])
        self.assertTrue(status["review"])
        self.assertTrue(status["translation_review"])
        self.assertEqual(status["translation_issues"], ["missing_numbers"])

    def test_successful_translation_retry_clears_translation_only_review_state(self):
        status = apply_translation_quality_status(
            {"version": 2, "review": True, "translation_review": True,
             "translation_issues": ["missing_numbers"]},
            [],
        )
        self.assertNotIn("review", status)
        self.assertNotIn("translation_review", status)

    def test_successful_translation_keeps_existing_asr_review_marker(self):
        status = apply_translation_quality_status(
            {"review": True, "asr_suspect": True,
             "review_reasons": ["asr_implausible_word_rate"]}, []
        )
        self.assertTrue(status["review"])

    def test_failed_quality_retry_remains_a_persistent_review_reason(self):
        status = apply_translation_quality_status(
            {"version": 2}, ["missing_numbers", "translation_retry_failed"]
        )
        self.assertTrue(status["review"])
        self.assertEqual(
            status["review_reasons"],
            ["translation_missing_numbers", "translation_retry_failed"],
        )

    def test_translation_recovery_clears_previous_translation_error(self):
        status = apply_translation_quality_status(set_review_reason({}, "translation_error"), [])
        self.assertNotIn("review", status)

    def test_quality_pass_preserves_legacy_review_without_reason(self):
        self.assertTrue(apply_translation_quality_status({"review": True}, []) ["review"])

    def test_quality_pass_preserves_asr_review_when_translation_review_clears(self):
        status = apply_translation_quality_status(
            {"review": True, "review_reasons": ["asr_implausible_word_rate"],
             "translation_review": True, "translation_issues": ["missing_numbers"]},
            [],
        )
        self.assertEqual(status["review_reasons"], ["asr_implausible_word_rate"])

    def test_review_reasons_clear_independently(self):
        status = set_review_reason({"version": 2}, "asr_implausible_word_rate")
        status = set_review_reason(status, "translation_missing_numbers")
        status = clear_review_reason(status, "asr_implausible_word_rate")
        self.assertEqual(status["review_reasons"], ["translation_missing_numbers"])

    def test_correction_clears_asr_reasons_but_preserves_translation_review(self):
        status = {
            "review": True,
            "review_reasons": ["asr_implausible_word_rate", "translation_missing_numbers"],
            "translation_review": True,
            "translation_issues": ["missing_numbers"],
        }
        corrected = clear_review_reason(status, "asr_implausible_word_rate")
        self.assertTrue(corrected["review"])
        self.assertEqual(corrected["review_reasons"], ["translation_missing_numbers"])


if __name__ == "__main__":
    unittest.main()
