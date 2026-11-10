"""Structured diagnostic event tests."""

import json
import os
import tempfile
import unittest
import warnings

from audio_capture import AudioWarningRecorder
from diagnostics import SessionDiagnostics, SessionQualityStats


class SessionDiagnosticsTests(unittest.TestCase):
    def test_session_quality_stats_track_failures_and_maximum_backlog(self):
        stats = SessionQualityStats()
        stats.increment("asr_suspect_windows")
        stats.increment("translation_quality_retries")
        stats.increment("translation_quality_unresolved")
        stats.set_max("max_audio_backlog_seconds", 3.2)
        stats.set_max("max_audio_backlog_seconds", 1.4)

        self.assertEqual(
            stats.snapshot(),
            {
                "asr_suspect_windows": 1,
                "translation_quality_retries": 1,
                "translation_quality_unresolved": 1,
                "max_audio_backlog_seconds": 3.2,
            },
        )

    def test_writes_structured_events_and_coalesces_repeated_warnings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "diagnostics.jsonl")
            diagnostics = SessionDiagnostics(path)

            diagnostics.write_event("audio_gap", seconds=0.5)
            diagnostics.record_warning(
                "SoundcardRuntimeWarning", "data discontinuity", "capture.py", 12
            )
            diagnostics.record_warning(
                "SoundcardRuntimeWarning", "data discontinuity", "capture.py", 12
            )
            diagnostics.flush_warning_summary()

            with open(path, encoding="utf-8") as source:
                events = [json.loads(line) for line in source]

        self.assertEqual([event["event"] for event in events], ["audio_gap", "warning_summary"])
        self.assertEqual(events[0]["seconds"], 0.5)
        self.assertEqual(events[1]["count"], 2)

    def test_diagnostics_sink_disables_itself_after_write_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "missing", "diagnostics.jsonl")
            diagnostics = SessionDiagnostics(path)

            self.assertFalse(diagnostics.write_event("first"))
            self.assertFalse(diagnostics.write_event("second"))

    def test_audio_warning_recorder_coalesces_repeated_runtime_warnings(self):
        warnings_seen = []
        recorder = AudioWarningRecorder(lambda *items: warnings_seen.append(items))

        recorder("data discontinuity", RuntimeWarning, "capture.py", 12)
        recorder("data discontinuity", RuntimeWarning, "capture.py", 12)
        recorder.flush()

        self.assertEqual(len(warnings_seen), 1)
        self.assertEqual(warnings_seen[0][-1], 2)
        self.assertIn("first_observed_at", warnings_seen[0][-2])
        self.assertIn("last_observed_at", warnings_seen[0][-2])

    def test_warning_recorder_does_not_mutate_process_global_warning_configuration(self):
        original_handler = warnings.showwarning
        original_filters = list(warnings.filters)
        recorder = AudioWarningRecorder(lambda *items: None)

        recorder("device warning", RuntimeWarning, "capture.py", 12)
        recorder.flush()

        self.assertIs(warnings.showwarning, original_handler)
        self.assertEqual(warnings.filters, original_filters)

    def test_capture_discontinuity_warning_is_distinguished_in_session_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "diagnostics.jsonl")
            diagnostics = SessionDiagnostics(path)
            capture_discontinuities = {"count": 0}

            def record_capture_warning(
                category, message, filename, lineno, observed_range=None, count=1
            ):
                if "discontinuity" in message.lower():
                    capture_discontinuities["count"] += count
                    diagnostics.write_event(
                        "audio_capture_discontinuity_warning",
                        count=count,
                        category=category,
                        message=message,
                        **(observed_range or {}),
                    )
                diagnostics.record_warning(category, message, filename, lineno, count)

            recorder = AudioWarningRecorder(record_capture_warning)
            recorder("data discontinuity", RuntimeWarning, "capture.py", 12)
            recorder("data discontinuity", RuntimeWarning, "capture.py", 12)
            recorder.flush()
            diagnostics.flush_warning_summary()

            with open(path, encoding="utf-8") as source:
                events = [json.loads(line) for line in source]

        self.assertEqual(capture_discontinuities["count"], 2)
        self.assertEqual(events[0]["event"], "audio_capture_discontinuity_warning")
        self.assertEqual(events[0]["count"], 2)
        self.assertIn("first_observed_at", events[0])
        self.assertIn("last_observed_at", events[0])
        self.assertEqual(events[1]["event"], "warning_summary")


if __name__ == "__main__":
    unittest.main()
