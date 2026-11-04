import unittest

from meeting_context import MeetingContext
from terminology import Terminology


class MeetingContextTests(unittest.TestCase):
    def setUp(self):
        self.terms = Terminology({
            "CPU": {"target": "CPU"},
            "document viewer": {"target": "文档查看器"},
        })

    def test_snapshot_retains_recent_sentences_and_known_terms_within_budget(self):
        context = MeetingContext(self.terms, max_sentences=2, max_chars=120)
        context.add("We will update the CPU in the next release.")
        context.add("The document viewer migration applies to all modules.")
        context.add("This is the final confirmation.")

        snapshot = context.snapshot()

        self.assertNotIn("We will update the CPU", snapshot)
        self.assertIn("CPU", snapshot)
        self.assertIn("document viewer", snapshot)
        self.assertLessEqual(len(snapshot), 120)

    def test_empty_or_whitespace_sentences_do_not_pollute_context(self):
        context = MeetingContext(self.terms)
        context.add("   ")

        self.assertEqual(context.snapshot(), "")

    def test_correction_replaces_the_recent_source_used_as_context(self):
        context = MeetingContext(self.terms)
        context.add("We will update the viewer.")
        context.replace_recent("We will update the viewer.", "We will update CPU usage.")

        self.assertIn("update CPU usage", context.snapshot())
        self.assertNotIn("update the viewer", context.snapshot())

    def test_asr_prompt_context_contains_only_recent_transcript_text(self):
        context = MeetingContext(self.terms, max_sentences=4, max_chars=180)
        context.add("We are discussing the document viewer.")
        context.add("The agent only knows the context window.")

        prompt = context.asr_prompt_context()

        self.assertEqual(
            prompt,
            "We are discussing the document viewer. The agent only knows the context window.",
        )
        self.assertNotIn("Terms:", prompt)
        self.assertNotIn("Recent meeting context:", prompt)


if __name__ == "__main__":
    unittest.main()
