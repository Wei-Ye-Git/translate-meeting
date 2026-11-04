import unittest

from translator import translation_quality_issues


class TranslationQualityTests(unittest.TestCase):
    class Tokenizer:
        class Batch(dict):
            def to(self, device):
                return self

        def encode(self, text, add_special_tokens=True):
            return list(range(len(text.split()) + int(add_special_tokens)))

        def __call__(self, texts, **kwargs):
            self.seen = list(texts)
            return self.Batch(input_ids=[[1]])

        def decode(self, output, skip_special_tokens=True):
            seen = getattr(self, "seen", [""])
            if seen and "⟦T0⟧" in seen[0]:
                return "⟦T0⟧ 服务器支持迁移。"
            return "我们已经批准该方案。"

    class Model:
        def __init__(self, output=None):
            self.output = output or [[1]]
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return self.output

    def test_context_translation_preserves_only_target_sentence_terms(self):
        from translator import Translator

        translator = Translator.__new__(Translator)
        translator.tokenizer = self.Tokenizer()
        translator.model = self.Model()
        translator.device = "cpu"
        translator.terminology = None
        translator.cache = {}

        translated = translator.translate_with_context(
            "Please approve this proposal.",
            ["We discussed the procurement plan."],
        )

        self.assertEqual(translated, "我们已经批准该方案。")

    def test_context_translation_falls_back_when_source_numbers_are_lost(self):
        from translator import Translator

        translator = Translator.__new__(Translator)
        translator.tokenizer = self.Tokenizer()
        translator.model = self.Model()
        translator.device = "cpu"
        translator.terminology = None
        translator.cache = {}
        translator.translate = lambda text, quality=False: "兜底译文 3"

        translated = translator.translate_with_context(
            "Please approve 3 servers.", ["We discussed the proposal."]
        )

        self.assertEqual(translated, "兜底译文 3")

    def test_context_translation_reverts_to_sentence_batch_when_alignment_fails(self):
        from translator import Translator

        translator = Translator.__new__(Translator)
        translator.tokenizer = self.Tokenizer()
        translator.model = self.Model()
        translator.device = "cpu"
        translator.terminology = None
        translator.cache = {}
        translator.translate_batch = lambda texts, quality=True: [
            "上一句。", "我们需要 3 台服务器。"
        ]

        result = translator.translate_context(
            ["We discussed the rollout.", "We need 3 servers."]
        )

        self.assertEqual(result, ["上一句。", "我们需要 3 台服务器。"])

    def test_context_translation_uses_neighboring_sentences_for_disambiguation(self):
        from translator import Translator

        class ContextTokenizer(self.Tokenizer):
            def decode(self, output, skip_special_tokens=True):
                return "我们讨论了部署计划。 我们将部署 3 台服务器。"

        translator = Translator.__new__(Translator)
        translator.tokenizer = ContextTokenizer()
        translator.model = self.Model()
        translator.device = "cpu"
        translator.terminology = None
        translator.cache = {}

        translated = translator.translate_context([
            "We discussed the deployment plan.",
            "We will deploy 3 servers.",
        ])

        self.assertEqual(translated, ["我们讨论了部署计划。", "我们将部署 3 台服务器。"])
        self.assertEqual(
            translator.tokenizer.seen,
            ["We discussed the deployment plan.\nWe will deploy 3 servers."],
        )

    def test_context_translation_rejects_output_when_one_context_sentence_is_missing(self):
        from translator import Translator

        class ContextTokenizer(self.Tokenizer):
            def decode(self, output, skip_special_tokens=True):
                return "我们将部署 3 台服务器。"

        translator = Translator.__new__(Translator)
        translator.tokenizer = ContextTokenizer()
        translator.model = self.Model()
        translator.device = "cpu"
        translator.terminology = None
        translator.cache = {}
        fallback_calls = []
        translator.translate_batch = lambda texts, quality=True: fallback_calls.extend(texts) or [
            "我们讨论了计划。", "我们将部署 3 台服务器。"
        ]

        result = translator.translate_context([
            "We discussed the deployment plan.",
            "We will deploy 3 servers.",
        ])

        self.assertEqual(fallback_calls, [
            "We discussed the deployment plan.",
            "We will deploy 3 servers.",
        ])
        self.assertEqual(result, ["我们讨论了计划。", "我们将部署 3 台服务器。"])

    def test_context_translation_falls_back_for_over_budget_context(self):
        from translator import Translator

        translator = Translator.__new__(Translator)
        translator.tokenizer = self.Tokenizer()
        translator.model = self.Model()
        translator.device = "cpu"
        translator.terminology = None
        translator.cache = {}
        fallback_calls = []
        translator.translate = lambda text, quality=False: fallback_calls.append(text) or "单句译文"

        result = translator.translate_with_context("target words", ["context " * 600])

        self.assertEqual(result, "单句译文")
        self.assertEqual(fallback_calls, ["target words"])

    def test_context_translation_uses_single_sentence_for_short_fragments(self):
        from translator import Translator

        translator = Translator.__new__(Translator)
        translator.tokenizer = self.Tokenizer()
        translator.model = self.Model()
        translator.device = "cpu"
        translator.terminology = None
        translator.cache = {}
        batch_calls = []
        translator.translate_batch = lambda texts, quality=True: batch_calls.extend(texts) or ["短句"]

        result = translator.translate_context(["Okay.", "We should continue the meeting."])

        self.assertEqual(result, ["短句"])
        self.assertEqual(batch_calls, ["Okay.", "We should continue the meeting."])

    def test_context_translation_keeps_english_terms_readable_to_the_model(self):
        from translator import Translator
        from terminology import Terminology

        class ContextTokenizer(self.Tokenizer):
            def decode(self, output, skip_special_tokens=True):
                if "\n" in self.seen[0]:
                    return "我们讨论了时间线。⟦T0⟧ 服务器支持迁移。"
                return super().decode(output, skip_special_tokens)

        translator = Translator.__new__(Translator)
        translator.tokenizer = ContextTokenizer()
        translator.model = self.Model()
        translator.device = "cpu"
        translator.terminology = Terminology({"CPU": {"target": "CPU"}})
        translator.cache = {}

        translator.translate_with_context(
            "The CPU server supports the migration.", ["We discussed the timeline."]
        )

        self.assertIn("timeline", translator.tokenizer.seen[0])
        self.assertIn("CPU", translator.tokenizer.seen[0])

    def test_quality_issues_keep_short_incomplete_phrases_out_of_omission_warning(self):
        self.assertNotIn(
            "possible_content_omission",
            translation_quality_issues("Please confirm the final deployment plan.", "请确认最终部署计划。"),
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
            translation_quality_issues("Please confirm the final deployment plan.", "Please confirm the final deployment plan."),
        )

    def test_accepts_chinese_translation_with_preserved_numbers(self):
        self.assertEqual(
            translation_quality_issues("We need 3 servers by 2027.", "我们需要 3 台服务器，截止时间为 2027 年。"),
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
            translation_quality_issues(
                "Will the application remain responsive?",
                "应用能否保持响应？",
            ),
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

    def test_chinese_translation_is_not_retried_as_a_one_word_translation(self):
        from translator import Translator

        self.assertFalse(
            Translator.should_retranslate(
                "We should review this pull request carefully.",
                "我们应该仔细审查这个拉取请求。",
            )
        )

    def test_term_cache_keeps_distinct_source_sentences_separate(self):
        from translator import Translator
        from terminology import Terminology

        translator = Translator.__new__(Translator)
        translator.device = "cpu"
        translator.terminology = Terminology({
            "AI": {"target": "AI", "aliases": ["人工智能"]},
            "PR": {"target": "拉取请求", "aliases": ["公关"]},
        })
        translator.cache = {}
        translator.ct2_model = object()
        translator.model = None
        translator.tokenizer = self.Tokenizer()
        translator.last_latency_seconds = 0.0
        translator._generate_ct2 = lambda text, num_beams=1, max_new_tokens=256: (
            "人工智能很有用。" if text.startswith("AI") else "公关很有用。"
        )

        ai_translation = translator.translate("AI is useful.")
        pr_translation = translator.translate("PR is useful.")

        self.assertEqual(ai_translation, "AI很有用。")
        self.assertEqual(pr_translation, "拉取请求很有用。")

    def test_quality_translation_uses_original_english_and_normalizes_terms(self):
        from translator import Translator
        from terminology import Terminology

        translator = Translator.__new__(Translator)
        translator.device = "cpu"
        translator.terminology = Terminology({
                "AI": {"target": "AI", "aliases": ["人工智能"]},
        })
        translator.cache = {}
        translator.ct2_model = object()
        translator.model = None
        translator.tokenizer = self.Tokenizer()
        translator.last_latency_seconds = 0.0
        translator._generate_ct2 = lambda text, num_beams=1, max_new_tokens=256: (
            "人工智能改善了代码审查。"
        )
        quality_sources = []
        translator.quality_issues = lambda source, translated: quality_sources.append(source) or []

        result = translator.translate("AI improves code review.", quality=True)

        self.assertEqual(result, "AI改善了代码审查。")
        self.assertEqual(quality_sources, ["AI improves code review."])

    def test_quality_translation_does_not_replace_chinese_output_with_english_prompt(self):
        from translator import Translator
        from terminology import Terminology

        translator = Translator.__new__(Translator)
        translator.device = "cpu"
        translator.terminology = Terminology({
            "AI": {"target": "AI", "aliases": ["人工智能"]},
        })
        translator.cache = {}
        translator.ct2_model = object()
        translator.model = None
        translator.tokenizer = self.Tokenizer()
        translator.last_latency_seconds = 0.0
        translator._generate_ct2 = lambda text, num_beams=1, max_new_tokens=256: "AI改善了代码审查。"

        result = translator.translate("AI improves code review.", quality=True)

        self.assertEqual(result, "AI改善了代码审查。")

if __name__ == "__main__":
    unittest.main()
