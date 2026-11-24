import unittest

from pipeline_metrics import PipelineMetrics


class PipelineMetricsTests(unittest.TestCase):
    def test_review_status_does_not_replace_record_capture_timestamp(self):
        from pipeline_metrics import record_capture_time_for_event

        metadata = {"end": 12.0, "captured_at_monotonic": 50.0}

        self.assertIsNone(record_capture_time_for_event("history_review", metadata, lambda _: 0.0))
        self.assertEqual(
            record_capture_time_for_event("history_add", metadata, lambda _: 0.0),
            50.0,
        )

    def test_revision_without_capture_metadata_does_not_fall_back_to_session_start(self):
        from pipeline_metrics import record_capture_time_for_event

        self.assertIsNone(
            record_capture_time_for_event("history_original_update", {}, lambda _: 0.0)
        )

    def test_committed_transcript_measures_audio_to_translation_and_ui_queue(self):
        metrics = PipelineMetrics()
        metrics.remember_record(7, 2, 10.0, published_at=10.4)

        english = metrics.presented_committed_english((7, 2), 10.7)
        self.assertAlmostEqual(english["audio_to_ui_seconds"], 0.7)
        self.assertAlmostEqual(english["ui_queue_seconds"], 0.3)

        metrics.translation_published("committed_chinese", (7, 2), 11.2)
        chinese = metrics.presented("committed_chinese", (7, 2), 11.4)
        self.assertAlmostEqual(chinese["audio_to_ui_seconds"], 1.4)
        self.assertAlmostEqual(chinese["ui_queue_seconds"], 0.2)

        summary = metrics.percentile_summary(metrics.summary())
        self.assertEqual(summary["committed_chinese"]["count"], 1)
        self.assertAlmostEqual(
            summary["committed_chinese"]["audio_to_ui_p95_seconds"], 1.4
        )

    def test_preview_and_live_translation_share_capture_to_ui_clock(self):
        metrics = PipelineMetrics()
        metrics.remember_preview(3, 5.0)
        preview = metrics.presented_english("preview_english", 3, 5.25)
        self.assertAlmostEqual(preview["audio_to_ui_seconds"], 0.25)

        metrics.remember_live(4, 5.0)
        metrics.translation_published("live_chinese", 4, 5.8)
        result = metrics.presented("live_chinese", 4, 5.9)
        self.assertAlmostEqual(result["audio_to_ui_seconds"], 0.9)
        self.assertAlmostEqual(result["ui_queue_seconds"], 0.1)


if __name__ == "__main__":
    unittest.main()
