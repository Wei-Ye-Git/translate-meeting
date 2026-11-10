"""Contextual translation, batching, and long-input routing tests."""

import unittest

from translator import Translator


class TranslationContextFlowTests(unittest.TestCase):
    class Tokenizer:
        class Batch(dict):
            def to(self, device):
                return self

        def encode(self, text, add_special_tokens=True):
            return list(range(len(text.split()) + int(add_special_tokens)))

        def __call__(self, texts, **kwargs):
            self.seen = list(texts)
            return self.Batch(input_ids=[[1] for _ in texts])

        def decode(self, output, skip_special_tokens=True):
            seen = getattr(self, "seen", [""])
            if seen and "⟦T0⟧" in seen[0]:
                return "⟦T0⟧ 服务器支持迁移。"
            return "我们已经批准该方案。"

    class Model:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return [[1]]

    @classmethod
    def make_translator(cls):
        translator = Translator.__new__(Translator)
        translator.tokenizer = cls.Tokenizer()
        translator.model = cls.Model()
        translator.device = "cpu"
        translator.terminology = None
        translator.cache = {}
        translator.last_latency_seconds = 0.0
        return translator

    def test_context_translation_preserves_only_target_sentence_terms(self):
        translator = self.make_translator()

        translated = translator.translate_with_context(
            "Please approve this proposal.",
            ["We discussed the procurement plan."],
        )

        self.assertEqual(translated, "我们已经批准该方案。")

    def test_context_translation_falls_back_when_source_numbers_are_lost(self):
        translator = self.make_translator()
        translator.translate = lambda text, quality=False: "兜底译文 3"

        translated = translator.translate_with_context(
            "Please approve 3 servers.", ["We discussed the proposal."]
        )

        self.assertEqual(translated, "兜底译文 3")

    def test_context_translation_reverts_to_sentence_batch_when_alignment_fails(self):
        translator = self.make_translator()
        translator.translate_batch = lambda texts, quality=True: [
            "上一句。", "我们需要 3 台服务器。"
        ]

        result = translator.translate_context(
            ["We discussed the rollout.", "We need 3 servers."]
        )

        self.assertEqual(result, ["上一句。", "我们需要 3 台服务器。"])

    def test_context_translation_uses_neighboring_sentences_for_disambiguation(self):
        translator = self.make_translator()
        translator.tokenizer.decode = lambda output, skip_special_tokens=True: (
            "我们讨论了部署计划。 我们将部署 3 台服务器。"
        )

        translated = translator.translate_context([
            "We discussed the deployment plan.", "We will deploy 3 servers."
        ])

        self.assertEqual(translated, ["我们讨论了部署计划。", "我们将部署 3 台服务器。"])
        self.assertEqual(
            translator.tokenizer.seen,
            ["We discussed the deployment plan.\nWe will deploy 3 servers."],
        )

    def test_context_translation_rejects_output_when_a_context_sentence_is_missing(self):
        translator = self.make_translator()
        translator.tokenizer.decode = lambda output, skip_special_tokens=True: "我们将部署 3 台服务器。"
        fallback_calls = []
        translator.translate_batch = lambda texts, quality=True: (
            fallback_calls.extend(texts) or ["我们讨论了计划。", "我们将部署 3 台服务器。"]
        )

        result = translator.translate_context([
            "We discussed the deployment plan.", "We will deploy 3 servers."
        ])

        self.assertEqual(len(fallback_calls), 2)
        self.assertEqual(result, ["我们讨论了计划。", "我们将部署 3 台服务器。"])

    def test_context_translation_falls_back_for_over_budget_context(self):
        translator = self.make_translator()
        fallback_calls = []
        translator.translate = lambda text, quality=False: fallback_calls.append(text) or "单句译文"

        result = translator.translate_with_context("target words", ["context " * 600])

        self.assertEqual(result, "单句译文")
        self.assertEqual(fallback_calls, ["target words"])

    def test_context_translation_keeps_english_terms_readable_to_the_model(self):
        from terminology import Terminology

        translator = self.make_translator()
        translator.terminology = Terminology({"CPU": {"target": "CPU"}})
        translator.tokenizer.decode = lambda output, skip_special_tokens=True: (
            "我们讨论了时间线。⟦T0⟧ 服务器支持迁移。"
            if "\n" in translator.tokenizer.seen[0]
            else "我们已经批准该方案。"
        )

        translator.translate_with_context(
            "The CPU server supports the migration.", ["We discussed the timeline."]
        )

        self.assertIn("timeline", translator.tokenizer.seen[0])
        self.assertIn("CPU", translator.tokenizer.seen[0])

    def test_long_sentence_translation_splits_and_preserves_clause_order(self):
        translator = self.make_translator()
        translator.ct2_model = object()
        translated_segments = []
        translator._generate_batch_ct2 = lambda texts, num_beams=1, max_new_tokens=256: (
            translated_segments.extend(texts)
            or [f"片段{len(translated_segments) + index + 1}。" for index in range(len(texts))]
        )
        translator._token_count = lambda text: len(text.split()) * 30
        source = (
            "We reviewed the project plan, identified the delivery risks, "
            "and agreed on the next steps before the end of the meeting."
        )

        translated = translator.translate(source, quality=True)

        self.assertGreater(len(translated_segments), 1)
        self.assertEqual(" ".join(translated_segments), source)

    def test_unpunctuated_long_phrase_avoids_excessive_tiny_chunks(self):
        translator = self.make_translator()
        translator.ct2_model = object()
        segments = []
        translator._generate_batch_ct2 = lambda texts, num_beams=1, max_new_tokens=256: (
            segments.extend(texts) or ["译文。"] * len(texts)
        )
        translator._token_count = lambda text: len(text.split()) * 50
        source = " ".join(["requirement"] * 31)

        translator.translate(source, quality=True)

        self.assertLessEqual(len(segments), 7)
        self.assertTrue(all(len(item.split()) >= 5 for item in segments))
        self.assertEqual(" ".join(segments), source)

    def test_context_with_too_many_sentences_uses_sentence_batch(self):
        translator = self.make_translator()
        translator._token_count = lambda text: 200 if "\n" in text else len(text.split())
        calls = []
        translator.translate_batch = lambda texts, quality=True: (
            calls.append((list(texts), quality)) or ["第一句。", "第二句。"]
        )

        result = translator.translate_context([
            "The deployment is ready for review.",
            "The release will follow the approved schedule.",
        ])

        self.assertEqual(result, ["第一句。", "第二句。"])
        self.assertEqual(len(calls), 1)

    def test_long_batch_item_keeps_short_neighbors_in_vectorized_batch(self):
        translator = self.make_translator()
        translator.ct2_model = None
        translator.model = self.Model()
        short_batch_sizes = []
        generation_calls = {"count": 0}
        translator.model.generate = lambda **kwargs: (
            short_batch_sizes.append(len(kwargs["input_ids"]))
            or [[1] for _ in kwargs["input_ids"]]
        )
        def decode(output, skip_special_tokens=True):
            generation_calls["count"] += 1
            return "短项译文。" if generation_calls["count"] <= 2 else "长项译文。"

        translator.tokenizer.decode = decode
        translator._token_count = lambda text: (
            513 if text.startswith("long") and len(text.split()) > 10
            else len(text.split())
        )
        long_text = "long " + "sentence " * 20

        result = translator.translate_batch(["short one", "short two", long_text], quality=True)

        self.assertEqual(result[:2], ["短项译文。", "短项译文。"])
        self.assertTrue(result[2].startswith("长项译文。"))
        self.assertEqual(short_batch_sizes[0], 2)


if __name__ == "__main__":
    unittest.main()
