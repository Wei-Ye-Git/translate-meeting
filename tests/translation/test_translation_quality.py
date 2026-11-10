"""Heuristic translation-quality issue detection tests."""

import unittest

from translator import translation_quality_issues


class TranslationQualityTests(unittest.TestCase):
    def test_quality_issues_keep_short_incomplete_phrases_out_of_omission_warning(self):
        self.assertNotIn(
            "possible_content_omission",
            translation_quality_issues(
                "Please confirm the final deployment plan.", "请确认最终部署计划。"
            ),
        )

    def test_detects_lost_numeric_values(self):
        self.assertIn(
            "missing_numbers",
            translation_quality_issues("We need 3 servers by 2027.", "我们需要服务器。"),
        )

    def test_detects_untranslated_or_empty_output(self):
        self.assertIn("empty_translation", translation_quality_issues("Please confirm.", ""))
        self.assertIn(
            "mostly_untranslated",
            translation_quality_issues(
                "Please confirm the final deployment plan.",
                "Please confirm the final deployment plan.",
            ),
        )

    def test_accepts_chinese_translation_with_preserved_numbers(self):
        self.assertEqual(
            translation_quality_issues(
                "We need 3 servers by 2027.", "我们需要 3 台服务器，截止时间为 2027 年。"
            ),
            [],
        )

    def test_detects_possible_missing_negation(self):
        self.assertIn(
            "possible_missing_negation",
            translation_quality_issues("We do not approve the proposal.", "我们批准了该提案。"),
        )

    def test_detects_semantically_incomplete_chinese_translation(self):
        issues = translation_quality_issues(
            "We can proactively test different levels of load for different scenarios.",
            "我们可以主动测试。",
        )
        self.assertIn("possible_content_omission", issues)

    def test_accepts_concise_chinese_translation_of_short_sentence(self):
        self.assertEqual(
            translation_quality_issues("Will the application remain responsive?", "应用能否保持响应？"),
            [],
        )

    def test_accepts_concise_chinese_translation_of_medium_sentence(self):
        self.assertEqual(
            translation_quality_issues(
                "Instead of saying the application feels slow, we can use data.",
                "与其说应用感觉慢，不如用数据说明。",
            ),
            [],
        )

    def test_detects_target_phrase_omission(self):
        self.assertIn(
            "possible_content_omission",
            translation_quality_issues("We will update the database tomorrow.", "明天。"),
        )


if __name__ == "__main__":
    unittest.main()
