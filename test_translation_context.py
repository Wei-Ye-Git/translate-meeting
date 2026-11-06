import unittest

from main import correction_prompt


class TranslationContextTests(unittest.TestCase):
    def test_correction_prompt_contains_neighboring_history(self):
        records = [
            ["The old dashboard.", ""],
            ["The migration feature.", ""],
            ["Please use the new document view.", ""],
            ["This applies to all modules.", ""],
        ]

        prompt = correction_prompt(records, 1)

        self.assertEqual(
            prompt,
            "The old dashboard.",
        )

    def test_correction_prompt_excludes_target_and_future_records(self):
        records = [
            ["Earlier sentence.", ""],
            ["Prior context one.", ""],
            ["Prior context two.", ""],
            ["Target sentence.", ""],
            ["Future sentence.", ""],
        ]

        prompt = correction_prompt(records, 3)

        self.assertEqual(
            prompt,
            "Earlier sentence. Prior context one. Prior context two.",
        )

    def test_correction_prompt_has_a_hard_character_limit(self):
        records = [["context " * 100, ""], ["target sentence", ""]]

        self.assertLessEqual(len(correction_prompt(records, 1, limit=3)), 360)

    def test_correction_prompt_skips_asr_review_candidates(self):
        records = [
            ["Earlier valid meeting context.", ""],
            ["Maybe later", ""],
            ["The target sentence needs correction.", ""],
        ]
        metadata = [{}, {"review_reasons": ["asr_short_fragment"]}, {}]

        prompt = correction_prompt(records, 2, metadata=metadata)

        self.assertEqual(prompt, "Earlier valid meeting context.")

    def test_correction_snapshot_rejects_a_changed_record_version(self):
        from main import correction_snapshot_is_current

        self.assertFalse(
            correction_snapshot_is_current(
                ["corrected text", ""], {"version": 2}, "original text", 1
            )
        )
        self.assertTrue(
            correction_snapshot_is_current(
                ["original text", ""], {"version": 1}, "original text", 1
            )
        )


if __name__ == "__main__":
    unittest.main()
