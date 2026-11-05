import unittest

from terminology import Terminology


class TerminologyTests(unittest.TestCase):
    def setUp(self):
        self.terms = Terminology({
            "pipeline": {
                "target": "流水线",
                "aliases": ["管道", "输油管", "输油管道"],
            },
            "prompt engineering": {
                "target": "提示工程",
                "aliases": ["即时工程"],
            },
        })

    def test_replaces_model_alias_when_source_contains_term(self):
        result = self.terms.apply(
            "The pipeline runs the tests.",
            "输油管运行这些测试。",
        )

        self.assertEqual(result, "流水线运行这些测试。")

    def test_replaces_alias_case_insensitively(self):
        result = self.terms.apply("The pipeline runs tests.", "输油管运行测试。")

        self.assertEqual(result, "流水线运行测试。")

    def test_does_not_rewrite_unrelated_translation(self):
        result = self.terms.apply("The pipe is blocked.", "管道堵塞了。")

        self.assertEqual(result, "管道堵塞了。")

    def test_builds_asr_prompt_from_matching_terms(self):
        prompt = self.terms.asr_prompt("The pipeline is maintained by the German team.")

        self.assertEqual(prompt, "pipeline")

    def test_asr_prompt_uses_canonical_term_for_spoken_variant(self):
        terms = Terminology({
            "API": {"target": "API", "asr_variants": ["A P I"]},
        })

        self.assertEqual(terms.asr_prompt("Please ask A P I tomorrow."), "API")

    def test_normalizes_asr_variant_when_it_survives_translation(self):
        terms = Terminology({
            "API": {"target": "API", "asr_variants": ["A P I"]},
        })

        self.assertEqual(
            terms.apply("The A P I server is ready.", "A P I 服务器已就绪。"),
            "API 服务器已就绪。",
        )

    def test_nested_variant_does_not_replace_a_different_glossary_phrase(self):
        terms = Terminology({
            "markup": {"target": "标记", "asr_variants": ["markup migration"]},
            "markup migration": {"target": "标记迁移"},
        })

        self.assertEqual(
            terms.matching_terms("The markup migration is ready."),
            ["markup migration"],
        )

    def test_matching_is_case_insensitive(self):
        self.assertEqual(
            self.terms.apply("PIPELINE", "输油管正在运行。"),
            "流水线正在运行。",
        )

    def test_matching_terms_prefers_longest_phrases_and_respects_word_boundaries(self):
        self.assertEqual(
            self.terms.matching_terms("The pipeline migration uses a pipeline."),
            ["pipeline"],
        )
        self.assertEqual(self.terms.matching_terms("A pipelining task."), [])

    def test_nested_terms_resolve_to_the_longest_configured_entry(self):
        terms = Terminology({
            "markup": {"target": "标记"},
            "markup migration": {"target": "标记迁移"},
        })

        self.assertEqual(terms.matching_terms("The markup migration is ready."), ["markup migration"])

    def test_asr_variants_resolve_to_the_canonical_glossary_term(self):
        terms = Terminology({
            "API": {"target": "API", "asr_variants": ["A P I"]},
        })

        self.assertEqual(terms.matching_terms("Please ask A P I tomorrow."), ["API"])

    def test_global_prompt_prefers_long_phrases(self):
        terms = Terminology({
            "markup": {"target": "标记"},
            "markup migration": {"target": "标记迁移"},
        })

        self.assertEqual(terms.global_prompt(max_terms=1), "markup migration")

    def test_global_prompt_lists_all_configured_terms(self):
        prompt = self.terms.global_prompt()
        self.assertIn("pipeline", prompt)
        self.assertIn("prompt engineering", prompt)

    def test_general_meeting_terms_cover_recognized_variants(self):
        from default_terminology import DEFAULT_TERMINOLOGY

        terms = Terminology(DEFAULT_TERMINOLOGY)

        normalized = terms.normalize_source(
            "We discussed re-prompt engineering and the paw point slides."
        )

        self.assertEqual(
            normalized,
            "We discussed prompt engineering and the PowerPoint slides.",
        )
        self.assertEqual(
            terms.apply("The product owner reviewed the acceptance test.", "产品拥有者检查了验收考试。"),
            "产品负责人检查了验收测试。",
        )
        self.assertEqual(
            terms.matching_terms("The product owner reviewed the acceptance test."),
            ["acceptance test", "product owner"],
        )

    def test_normalizes_general_asr_variants_to_canonical_spelling(self):
        terms = Terminology({
            "PowerPoint": {
                "target": "PowerPoint",
                "asr_variants": ["paw point"],
            },
            "Grafana": {
                "target": "Grafana",
                "asr_variants": ["graphana"],
            },
        })

        self.assertEqual(
            terms.normalize_source("Graphana dashboard and paw point slides."),
            "Grafana dashboard and PowerPoint slides.",
        )

    def test_ai_meeting_terms_normalize_common_translation_errors(self):
        from default_terminology import DEFAULT_TERMINOLOGY

        terms = Terminology(DEFAULT_TERMINOLOGY)

        cases = [
            ("AI improves code review.", "大赦国际改善了代码审查。", "AI改善了代码审查。"),
            ("Please create a PR.", "请创建一个公关。", "请创建一个拉取请求。"),
            ("Write a precise prompt.", "写一个精确的迅速。", "写一个精确的提示词。"),
            ("The ticket needs review.", "机票需要审核。", "工单需要审核。"),
        ]

        for source_text, translated, expected in cases:
            with self.subTest(source=source_text):
                self.assertEqual(terms.apply(source_text, translated), expected)

    def test_asr_prompt_does_not_inject_unspoken_core_terms(self):
        terms = Terminology({
            "AI": {"target": "AI", "asr_always": True},
            "PR": {"target": "PR", "asr_always": True},
            "prompt": {"target": "提示词", "asr_always": True},
            "ticket": {"target": "工单", "asr_always": True},
            "pipeline": {"target": "流水线"},
        })

        self.assertEqual(terms.asr_prompt("", include_always=False), "")
        self.assertEqual(
            terms.asr_prompt("The pipeline is ready.", include_always=False),
            "pipeline",
        )

    def test_asr_prompt_can_skip_always_terms_but_keep_context_matches(self):
        terms = Terminology({
            "AI": {"target": "AI", "asr_always": True},
            "pipeline": {"target": "流水线"},
        })

        self.assertEqual(terms.asr_prompt("The pipeline uses AI."), "AI, pipeline")
        self.assertEqual(
            terms.asr_prompt("The pipeline uses AI.", include_always=False),
            "pipeline, AI",
        )
        self.assertEqual(terms.asr_prompt("", include_always=False), "")

    def test_rejects_invalid_entry_without_target(self):
        with self.assertRaises(ValueError):
            Terminology({"pipeline": {"aliases": ["管道"]}})

    def test_rejects_invalid_variant_configuration(self):
        with self.assertRaises(ValueError):
            Terminology({"API": {"target": "API", "asr_variants": "A P I"}})

    def test_rejects_non_object_glossary(self):
        with self.assertRaises(ValueError):
            Terminology(["pipeline"])


if __name__ == "__main__":
    unittest.main()
