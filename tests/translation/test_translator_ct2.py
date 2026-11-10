"""CTranslate2 backend tests."""

import unittest
from unittest.mock import MagicMock

from translator import Translator


class TranslatorCT2Tests(unittest.TestCase):
    def test_ct2_model_is_used_when_available(self):
        translator = Translator.__new__(Translator)
        translator.device = "cpu"
        translator.terminology = None
        translator.cache = {}
        translator.model = None

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 0]
        mock_tokenizer.convert_ids_to_tokens.return_value = ["▁hello", "</s>"]
        mock_tokenizer.convert_tokens_to_ids.return_value = [10, 11]
        mock_tokenizer.decode.return_value = "你好"
        translator.tokenizer = mock_tokenizer

        mock_hypothesis = MagicMock()
        mock_hypothesis.hypotheses = [["▁ni", "▁hao"]]
        mock_ct2 = MagicMock()
        mock_ct2.translate_batch.return_value = [mock_hypothesis]
        translator.ct2_model = mock_ct2

        result = translator.translate("hello", quality=True)

        self.assertEqual(result, "你好")
        mock_ct2.translate_batch.assert_called_once()
        args, kwargs = mock_ct2.translate_batch.call_args
        self.assertEqual(args[0], [["▁hello", "</s>"]])

    def test_ct2_batch_translates_multiple_texts(self):
        translator = Translator.__new__(Translator)
        translator.device = "cpu"
        translator.terminology = None
        translator.cache = {}
        translator.model = None

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 0]
        mock_tokenizer.convert_ids_to_tokens.return_value = ["▁hello", "</s>"]
        mock_tokenizer.convert_tokens_to_ids.return_value = [10, 11]
        mock_tokenizer.decode.return_value = "你好"
        translator.tokenizer = mock_tokenizer

        mock_hyp1 = MagicMock()
        mock_hyp1.hypotheses = [["▁ni", "▁hao"]]
        mock_hyp2 = MagicMock()
        mock_hyp2.hypotheses = [["▁ni", "▁hao"]]
        mock_ct2 = MagicMock()
        mock_ct2.translate_batch.return_value = [mock_hyp1, mock_hyp2]
        translator.ct2_model = mock_ct2

        results = translator.translate_batch(["hello", "world"])

        self.assertEqual(results, ["你好", "你好"])
        self.assertEqual(len(mock_ct2.translate_batch.call_args[0][0]), 2)

    def test_ct2_context_translation_uses_the_ct2_runtime(self):
        translator = Translator.__new__(Translator)
        translator.device = "cpu"
        translator.terminology = None
        translator.cache = {}
        translator.model = None

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 0]
        mock_tokenizer.convert_ids_to_tokens.return_value = ["▁context", "</s>"]
        mock_tokenizer.convert_tokens_to_ids.return_value = [10, 11]
        mock_tokenizer.decode.return_value = "我们讨论了计划。 我们将继续执行。"
        translator.tokenizer = mock_tokenizer

        hypotheses = [MagicMock(), MagicMock()]
        hypotheses[0].hypotheses = [["▁we", "▁discussed"]]
        hypotheses[1].hypotheses = [["▁we", "▁continue"]]
        mock_ct2 = MagicMock()
        mock_ct2.translate_batch.return_value = hypotheses
        translator.ct2_model = mock_ct2

        result = translator.translate_context([
            "We discussed the deployment plan.",
            "We will continue the deployment tomorrow.",
        ])

        self.assertEqual(len(result), 2)
        mock_ct2.translate_batch.assert_called_once()
        self.assertEqual(len(mock_ct2.translate_batch.call_args.args[0]), 1)

    def test_ct2_translator_configures_threads(self):
        from unittest.mock import patch
        with patch("ctranslate2.Translator") as mock_cls, \
             patch("transformers.MarianTokenizer.from_pretrained"), \
             patch("transformers.MarianMTModel.from_pretrained"):
            mock_cls.return_value = MagicMock()
            Translator(model_path="models/opus_mt_en_zh_ct2", intra_threads=2, inter_threads=1)
            mock_cls.assert_called_once()
            _, kwargs = mock_cls.call_args
            self.assertEqual(kwargs.get("intra_threads"), 2)
            self.assertEqual(kwargs.get("inter_threads"), 1)



if __name__ == "__main__":
    unittest.main()
