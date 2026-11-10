"""Localization resource loading tests."""

import json
import os
import unittest

from i18n import configured_locales, data, set_locale, t


class LocalizationTests(unittest.TestCase):
    def tearDown(self):
        set_locale("zh-CN")

    def test_chinese_messages_are_loaded_from_categorized_resource(self):
        set_locale("zh-CN")

        self.assertEqual(t("ui.copy_bilingual"), "复制双语")
        self.assertEqual(t("status.audio_gap", count=3), "音频流出现第 3 个间隙")

    def test_machine_readable_translation_rules_are_resource_data(self):
        markers = data("translation.negation_markers")

        self.assertIn("不", markers)
        self.assertIn("没有", markers)

    def test_all_locale_files_are_valid_json_and_configured(self):
        root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        locale_dir = os.path.join(root, "locales")
        files = [name for name in os.listdir(locale_dir) if name.endswith(".json")]

        for filename in files:
            with self.subTest(filename=filename):
                with open(os.path.join(locale_dir, filename), encoding="utf-8") as source:
                    resource = json.load(source)
                if filename == "meeting-candidate-zh-CN.json":
                    self.assertTrue(resource)
                else:
                    self.assertIsInstance(resource.get("messages"), dict)
                    self.assertIn(filename[:-5], configured_locales())


if __name__ == "__main__":
    unittest.main()
