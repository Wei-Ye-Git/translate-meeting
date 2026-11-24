import unittest
"""Terminology validation, matching, and snapshot tests."""

import json
import os
import os
import tempfile

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
        terms = Terminology({
            "prompt engineering": {
                "target": "prompt engineering",
                "asr_variants": ["re-prompt engineering"],
            },
            "PowerPoint": {
                "target": "PowerPoint",
                "asr_variants": ["paw point"],
            },
            "product owner": {
                "target": "产品负责人",
                "aliases": ["产品拥有者"],
            },
            "acceptance test": {
                "target": "验收测试",
                "aliases": ["验收考试"],
            },
        })

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

    def test_reference_report_terms_are_available_in_general_glossary(self):
        terms = Terminology.from_file(
            os.path.join(os.path.dirname(__file__), "..", "..", "glossaries", "general.json")
        )

        self.assertEqual(
            terms.apply("We will use plan mode and check the context window.", "我们使用规划模式并检查上下文范围。"),
            "我们使用计划模式并检查上下文窗口。",
        )

    def test_common_unit_testing_terms_are_available_asr_hotwords(self):
        terms = Terminology.from_file(
            os.path.join(os.path.dirname(__file__), "..", "..", "glossaries", "general.json")
        )
        always = terms.asr_prompt("", include_always=True)
        for term in ("xUnit", "NUnit", "C#", "assert", "test fixture"):
            self.assertIn(term, always)

    def test_ai_meeting_terms_normalize_common_translation_errors(self):
        terms = Terminology({
            "AI": {"target": "AI", "aliases": ["大赦国际"]},
            "PR": {"target": "拉取请求", "aliases": ["公关"]},
            "ticket": {"target": "工单", "aliases": ["机票"]},
        })

        cases = [
            ("AI improves code review.", "大赦国际改善了代码审查。", "AI改善了代码审查。"),
            ("Please create a PR.", "请创建一个公关。", "请创建一个拉取请求。"),
            ("Write a precise prompt.", "写一个精确的提示词。", "写一个精确的提示词。"),
            ("The ticket needs review.", "机票需要审核。", "工单需要审核。"),
        ]

        for source_text, translated, expected in cases:
            with self.subTest(source=source_text):
                self.assertEqual(terms.apply(source_text, translated), expected)

    def test_company_and_aec_terms_use_canonical_product_names(self):
        terms = Terminology({
            "manufacturing execution system": {
                "target": "制造执行系统",
                "aliases": ["制造操作系统"],
            },
        })

        self.assertEqual(
            terms.apply(
                "The manufacturing execution system connects with BIM.",
                "制造操作系统连接。",
            ),
            "制造执行系统连接。",
        )

    def test_technical_terms_do_not_rewrite_unrelated_ordinary_words(self):
        terms = Terminology({
            "prompt engineering": {
                "target": "提示工程",
                "asr_variants": ["quick"],
                "context_terms": ["engineering"],
            },
        })

        self.assertEqual(
            terms.apply("The team works quickly.", "团队工作迅速。"),
            "团队工作迅速。",
        )
        self.assertEqual(
            terms.apply("Please write a prompt.", "请快速回复。"),
            "请快速回复。",
        )

    def test_product_glossary_normalizes_names_and_asr_variants(self):
        terms = Terminology({
            "AcmeSuite": {
                "target": "AcmeSuite",
                "asr_variants": ["A. C. M. E. Suite"],
            },
            "AcmeSuite 4.0": {
                "target": "AcmeSuite 4.0",
                "asr_variants": ["AcmeSuite 4 point 0"],
            },
            "ProductX": {"target": "ProductX", "asr_variants": ["Product X"]},
            "Acme": {
                "target": "Acme",
                "asr_variants": ["ACME"],
                "context_terms": ["software"],
            },
        })

        self.assertEqual(
            terms.normalize_source(
                "A. C. M. E. Suite includes AcmeSuite 4 point 0 and Product X."
            ),
            "AcmeSuite includes AcmeSuite 4.0 and ProductX.",
        )
        self.assertEqual(
            terms.normalize_source("ACME product line is ready."),
            "Acme product line is ready.",
        )
        self.assertEqual(
            terms.normalize_source("Please open the old draft."),
            "Please open the old draft.",
        )

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

    def test_rejects_non_string_aliases_and_asr_always_values(self):
        invalid_entries = [
            {"API": {"target": "API", "aliases": [None]}},
            {"API": {"target": "API", "asr_variants": [" "]}},
            {"API": {"target": "API", "asr_always": "yes"}},
        ]

        for entries in invalid_entries:
            with self.subTest(entries=entries):
                with self.assertRaises(ValueError):
                    Terminology(entries)

    def test_rejects_non_object_glossary(self):
        with self.assertRaises(ValueError):
            Terminology(["pipeline"])

    def test_manager_publishes_incremental_updates_as_new_snapshots(self):
        from terminology import TerminologyManager

        manager = TerminologyManager(Terminology({"pipeline": {"target": "流水线"}}))
        old_snapshot = manager.snapshot()
        manager.queue_update("add", "streaming", {"target": "流式"})

        new_snapshot = manager.apply_pending()

        self.assertEqual(old_snapshot.version, 1)
        self.assertEqual(new_snapshot.version, 2)
        self.assertNotIn("streaming", old_snapshot.entries)
        self.assertEqual(new_snapshot.entries["streaming"]["target"], "流式")

    def test_manager_rejects_invalid_update_without_losing_existing_snapshot(self):
        from terminology import TerminologyManager

        manager = TerminologyManager(Terminology({"pipeline": {"target": "流水线"}}))
        with self.assertRaises(ValueError):
            manager.queue_update("add", "invalid", {"aliases": ["bad"]})

        unchanged = manager.apply_pending()

        self.assertEqual(unchanged.version, 1)
        self.assertIn("pipeline", unchanged.entries)
        self.assertNotIn("invalid", unchanged.entries)
        self.assertIsNone(manager.last_update_error)

    def test_manager_hot_reloads_glossary_files_at_commit_boundary(self):
        from terminology import TerminologyManager

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "terms.json")
            with open(path, "w", encoding="utf-8") as output:
                json.dump({"pipeline": {"target": "流水线"}}, output)
            manager = TerminologyManager.from_files([path])
            with open(path, "w", encoding="utf-8") as output:
                json.dump({"streaming": {"target": "流式"}}, output)

            self.assertTrue(manager.poll_files())
            self.assertNotIn("streaming", manager.snapshot().entries)
            snapshot = manager.apply_pending()

            self.assertEqual(snapshot.version, 2)
            self.assertIn("streaming", snapshot.entries)

    def test_manager_does_not_repeat_the_same_failed_file_reload(self):
        from terminology import TerminologyManager

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "terms.json")
            with open(path, "w", encoding="utf-8") as output:
                json.dump({"pipeline": {"target": "流水线"}}, output)
            manager = TerminologyManager.from_files([path])
            with open(path, "w", encoding="utf-8") as output:
                output.write("{")

            self.assertFalse(manager.poll_files())
            first_error = manager.last_reload_error
            self.assertFalse(manager.poll_files())
            self.assertIs(manager.last_reload_error, first_error)
            self.assertEqual(manager.snapshot().version, 1)

    def test_manager_reports_duplicate_add_and_keeps_later_valid_updates(self):
        from terminology import TerminologyManager

        manager = TerminologyManager(Terminology({"pipeline": {"target": "流水线"}}))
        manager.queue_update("add", "pipeline", {"target": "新译法"})
        manager.queue_update("add", "streaming", {"target": "流式"})

        snapshot = manager.apply_pending()

        self.assertEqual(snapshot.version, 2)
        self.assertEqual(snapshot.entries["pipeline"]["target"], "流水线")
        self.assertIn("streaming", snapshot.entries)
        self.assertIsNotNone(manager.last_update_error)

    def test_manager_file_reload_does_not_erase_queued_manual_update(self):
        from terminology import TerminologyManager

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "terms.json")
            with open(path, "w", encoding="utf-8") as output:
                json.dump({"pipeline": {"target": "流水线"}}, output)
            manager = TerminologyManager.from_files([path])
            manager.queue_update("add", "streaming", {"target": "流式"})
            with open(path, "w", encoding="utf-8") as output:
                json.dump({"pipeline": {"target": "管线"}}, output)

            self.assertTrue(manager.poll_files())
            snapshot = manager.apply_pending()

            self.assertEqual(snapshot.entries["pipeline"]["target"], "管线")
            self.assertEqual(snapshot.entries["streaming"]["target"], "流式")


if __name__ == "__main__":
    unittest.main()
