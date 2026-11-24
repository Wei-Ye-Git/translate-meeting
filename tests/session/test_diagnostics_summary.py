import json
import os
import tempfile
import unittest

from summarize_diagnostics import summarize


class DiagnosticsSummaryTests(unittest.TestCase):
    def test_summarizes_issue_categories_and_latency_percentiles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "diagnostics.jsonl")
            rows = [
                {"event": "asr_quality", "issue": "possible_fast_speech"},
                {"event": "asr_quality", "issue": "low_confidence_words"},
                {"event": "translation_quality", "issue": "missing_numbers",
                 "record_index": 1},
                {"event": "history_translation", "index": 1,
                 "issues": ["missing_numbers"]},
                {"event": "translation_started", "queue_wait_seconds": 0.1},
                {"event": "translation_started", "queue_wait_seconds": 0.9},
                {"event": "translation_started"},
                {"event": "asr_result", "seconds": 0.2, "audio_seconds": 1.0},
                {"event": "asr_result", "seconds": 0.4, "audio_seconds": 1.0},
                {"event": "subtitle_presented", "subtitle_kind": "preview_english",
                 "audio_to_ui_seconds": 0.35, "ui_queue_seconds": 0.05},
                {"event": "subtitle_presented", "subtitle_kind": "preview_english",
                 "audio_to_ui_seconds": 0.65, "ui_queue_seconds": 0.15},
                {"event": "ui_queue_backlog"},
                {"event": "subtitle_presented", "subtitle_kind": "live_chinese",
                 "audio_to_ui_seconds": 1.2, "ui_queue_seconds": 0.1},
                {"event": "subtitle_presented", "subtitle_kind": "committed_chinese",
                 "audio_to_ui_seconds": 2.4, "ui_queue_seconds": 0.2},
                {"event": "asr_window_no_text"},
                {"event": "transcript_segment_dropped"},
                {"event": "session_finished", "audio_capture_timeline_gap_frames": 160,
                 "audio_capture_timeline_gap_events": 2,
                 "asr_detected_gap_events": 1,
                 "low_latency_queue_dropped_frames": 80, "stale_audio_frames": 240,
                 "low_latency_queue_dropped_chunks": 3,
                 "audio_capture_underflow_warnings": 1,
                 "audio_capture_overflow_warnings": 2,
                 "audio_capture_discontinuity_warnings": 5,
                  "quality_stats": {"audio_timeline_gap_frames": 320,
                                    "audio_timeline_gap_events": 4,
                                    "audio_queue_dropped_chunks": 3,
                                    "audio_stale_chunks": 6},
                 "runtime_warning_counts": {"SoundcardRuntimeWarning:data discontinuity": 4}},
                {"event": "copilot_request_started", "input_characters": 100,
                 "elapsed_seconds": 10.0},
                {"event": "copilot_request_completed", "outcome": "applied",
                 "latency_seconds": 0.8, "provider_usage": {"events": [
                     {"tokens": {"input": 40, "output": 12}, "cost": 0.01}
                 ]}, "elapsed_seconds": 11.0},
                {"event": "copilot_request_started", "input_characters": 60,
                 "elapsed_seconds": 20.0},
                {"event": "copilot_request_completed", "outcome": "failed_or_discarded",
                 "latency_seconds": 1.2, "elapsed_seconds": 22.0},
            ]
            with open(path, "w", encoding="utf-8") as output:
                for row in rows:
                    output.write(json.dumps(row) + "\n")

            report = summarize(path)

        self.assertEqual(report["issue_counts"]["possible_fast_speech"], 1)
        self.assertEqual(report["issue_counts"]["missing_numbers"], 1)
        self.assertEqual(report["asr_result_count"], 2)
        self.assertEqual(report["asr_latency_p50_seconds"], 0.3)
        self.assertEqual(report["asr_real_time_factor"], 0.3)
        self.assertEqual(report["history_translation_queue_wait_p50_seconds"], 0.5)
        self.assertEqual(report["history_translation_queue_wait_p95_seconds"], 0.86)
        self.assertEqual(report["subtitle_audio_to_ui"]["preview_english"]["p50_seconds"], 0.5)
        self.assertEqual(report["subtitle_audio_to_ui"]["committed_chinese"]["p95_seconds"], 2.4)
        self.assertEqual(report["subtitle_ui_queue"]["preview_english"]["p95_seconds"], 0.145)
        self.assertEqual(report["audio_queue_dropped_frames"], 80)
        self.assertEqual(report["audio_capture_timeline_gap_frames"], 160)
        self.assertEqual(report["audio_capture_timeline_gap_events"], 2)
        self.assertEqual(report["asr_detected_gap_events"], 1)
        self.assertEqual(report["asr_timeline_gap_frames"], 320)
        self.assertEqual(report["asr_timeline_gap_events"], 4)
        self.assertEqual(report["stale_audio_frames"], 240)
        self.assertEqual(report["audio_queue_dropped_chunks"], 3)
        self.assertEqual(report["audio_queue_drop_events"], 3)
        self.assertEqual(report["audio_capture_underflow_warnings"], 1)
        self.assertEqual(report["audio_capture_overflow_warnings"], 2)
        self.assertEqual(report["audio_capture_discontinuity_warnings"], 5)
        self.assertEqual(report["stale_audio_events"], 6)
        self.assertEqual(report["ui_queue_backlog_events"], 1)
        self.assertEqual(report["pipeline_counts"]["asr_window_no_text"], 1)
        self.assertEqual(report["pipeline_counts"]["transcript_segment_dropped"], 1)
        self.assertEqual(
            report["runtime_warning_counts"]["SoundcardRuntimeWarning:data discontinuity"],
            4,
        )
        self.assertEqual(report["copilot_request_count"], 2)
        self.assertEqual(report["copilot_applied_count"], 1)
        self.assertEqual(report["copilot_failed_or_discarded_count"], 1)
        self.assertEqual(report["copilot_requests_per_session_hour"], 327.27)
        self.assertEqual(report["copilot_provider_usage_totals"]["tokens.input"], 40)
        self.assertIsNone(report["copilot_github_credits"])


if __name__ == "__main__":
    unittest.main()
