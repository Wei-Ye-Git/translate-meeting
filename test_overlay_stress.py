import unittest
from overlay import Overlay


class OverlayStressTests(unittest.TestCase):
    def test_current_transcript_is_labeled_as_provisional_preview(self):
        overlay = Overlay(font_size=14)
        overlay.root.withdraw()
        try:
            self.assertEqual(overlay.current_label.cget("text"), "实时预览（内容可能变化）")
        finally:
            overlay.close()

    def test_overlay_keeps_only_configured_history_window(self):
        overlay = Overlay(font_size=14, max_history_records=2)
        overlay.root.withdraw()
        try:
            overlay.append_history(0, "One.", "一。")
            overlay.append_history(1, "Two.", "二。")
            overlay.append_history(2, "Three.", "三。")

            self.assertEqual(overlay.history, [("Two.", "二。"), ("Three.", "三。")])
            self.assertNotIn("One.", overlay.history_text.get("1.0", "end"))
        finally:
            overlay.close()

    def test_history_replacement_preserves_timestamps_for_visible_rows(self):
        overlay = Overlay(font_size=14)
        overlay.root.withdraw()
        try:
            overlay.append_history(
                0, "Sentence.", "译文。",
                metadata={"start": 61.25, "end": 62.5, "version": 1},
            )
            overlay.replace_history(list(overlay.history))

            self.assertIn("[01:01.250-01:02.500]", overlay.history_text.get("1.0", "end"))
        finally:
            overlay.close()

    def test_history_revision_is_marked_and_previous_text_is_preserved(self):
        overlay = Overlay(font_size=14)
        overlay.root.withdraw()
        try:
            overlay.append_history(
                0, "Cold site is ready.", "冷藏地点已就绪。",
                metadata={"start": 1.0, "end": 2.0, "version": 1},
            )
            overlay.update_history_original(
                0,
                "The scanner is ready.",
                metadata={
                    "start": 1.0,
                    "end": 2.0,
                    "version": 2,
                    "corrected": True,
                    "previous_original": "Cold site is ready.",
                    "previous_translation": "冷藏地点已就绪。",
                },
                version=2,
            )

            self.assertIn("[已修订] The scanner is ready.", overlay.history_text.get("1.0", "end"))
            self.assertEqual(
                overlay.history_metadata[0]["previous_original"],
                "Cold site is ready.",
            )
            self.assertEqual(
                overlay.history_metadata[0]["previous_translation"],
                "冷藏地点已就绪。",
            )
        finally:
            overlay.close()

    def test_new_preview_clears_translation_from_previous_phrase(self):
        overlay = Overlay(font_size=14)
        overlay.root.withdraw()
        try:
            overlay.update_current("First phrase", "第一句")
            overlay.update_current("Next phrase", "")

            self.assertEqual(overlay.translation.cget("text"), "")
        finally:
            overlay.close()

    def test_previous_revision_is_included_in_bilingual_export(self):
        overlay = Overlay(font_size=14)
        overlay.root.withdraw()
        try:
            overlay.append_history(
                0,
                "The scanner is available.",
                "扫描工具可用。",
                metadata={
                    "start": 1.0,
                    "end": 2.0,
                    "version": 2,
                    "corrected": True,
                    "previous_original": "The site is available.",
                    "previous_translation": "冷藏地点可用。",
                },
            )

            export = overlay.bilingual_text()

            self.assertIn("[EN - 修订前] The site is available.", export)
            self.assertIn("[ZH - 修订前] 冷藏地点可用。", export)
        finally:
            overlay.close()

    def test_export_includes_relative_timestamps_and_review_marker(self):
        overlay = Overlay(font_size=14)
        overlay.root.withdraw()
        try:
            overlay.append_history(
                0,
                "Grafana helps teams.",
                "Grafana 帮助团队。",
                metadata={"start": 12.5, "end": 14.0, "version": 1},
            )

            text = overlay.bilingual_text()

            self.assertIn("[00:12.500-00:14.000] [EN] Grafana helps teams.", text)
            self.assertIn("[ZH] Grafana 帮助团队。", text)
            overlay.mark_history_for_review(0)
            self.assertIn("[需要复核]", overlay.bilingual_text())
            self.assertIn("[需要复核]", overlay.history_text.get("1.0", "end"))
        finally:
            overlay.close()

    def test_copy_english_excludes_unconfirmed_live_preview(self):
        overlay = Overlay(font_size=14)
        overlay.root.withdraw()
        try:
            overlay.append_english("temporary preview")
            overlay.copy_english()

            self.assertEqual(overlay.status.cget("text"), "还没有可复制的英文记录")
        finally:
            overlay.close()

    def test_copy_english_includes_timestamps_and_review_marker(self):
        overlay = Overlay(font_size=14)
        overlay.root.withdraw()
        try:
            overlay.append_history(
                0, "Please review this result.", "请复核此结果。",
                metadata={"start": 8.25, "end": 9.5, "version": 1, "review": True},
            )
            overlay.copy_english()

            copied = overlay.root.clipboard_get()
            self.assertIn("[00:08.250-00:09.500]", copied)
            self.assertIn("[需要复核] Please review this result.", copied)
        finally:
            overlay.close()

    def test_overlay_handles_400_sentences_without_crashing(self):
        overlay = Overlay(font_size=14)
        overlay.root.withdraw()
        try:
            for i in range(400):
                overlay.append_history(i, f"English sentence {i}.", f"中文翻译 {i}。")

            self.assertEqual(len(overlay.history), 400)

            # Test updating translation and original at various points
            overlay.update_history(0, "首句翻译更新")
            overlay.update_history(250, "中段翻译更新")
            overlay.update_history(399, "末句翻译更新")
            overlay.update_history_original(250, "Updated English 250.")

            self.assertEqual(overlay.history[0][1], "首句翻译更新")
            self.assertEqual(overlay.history[250][0], "Updated English 250.")
            self.assertEqual(overlay.history[250][1], "翻译中...")
            self.assertEqual(overlay.history[399][1], "末句翻译更新")

            overlay.append_history(
                400, "Timed sentence.", "定时句。",
                metadata={"start": 12.5, "end": 14.0, "version": 2},
            )
            self.assertIn("[00:12.500-00:14.000]", overlay.history_text.get("1.0", "end"))
            self.assertFalse(overlay.update_history(400, "过期翻译", version=1))
            self.assertTrue(overlay.update_history(400, "新翻译", version=2))
            self.assertEqual(overlay.history[400][1], "新翻译")
            self.assertFalse(
                overlay.update_history_original(
                    400, "Stale original.", metadata={"version": 1}
                )
            )

            overlay.replace_history([("Replacement sentence.", "新句子。")])
            self.assertEqual(len(overlay.history_metadata), 1)
            self.assertEqual(overlay.history[0], ("Replacement sentence.", "新句子。"))

            # Check that the text widget content exists and is not empty/blank
            text_content = overlay.history_text.get("1.0", "end")
            self.assertIn("Replacement sentence.", text_content)
            self.assertIn("Replacement sentence.", text_content)
            self.assertIn("新句子。", text_content)
        finally:
            overlay.close()


if __name__ == "__main__":
    unittest.main()
