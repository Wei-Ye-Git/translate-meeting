"""Result-queue pressure must not discard committed transcript events."""

import queue
import unittest

from ui_events import apply_result_event, drain_worker_errors, enqueue_result_event


class ResultQueueTests(unittest.TestCase):
    def test_committed_subtitle_events_return_presentation_timestamps(self):
        class Overlay:
            history = []
            history_metadata = []
            history_offset = 0

            def append_history(self, index, english, chinese, metadata=None):
                self.history.append((english, chinese))
                self.history_metadata.append(metadata or {})

            def set_status(self, _):
                pass

            def update_history(self, index, chinese, version=None):
                self.history[index] = (self.history[index][0], chinese)

        overlay = Overlay()
        config = type("Config", (), {"max_overlay_records": 10})()
        state = {"revision": 0}
        lock = __import__("threading").Lock()

        english = apply_result_event(
            overlay, config,
            ("history_add", 0, "Hello.", {"version": 1}, 1), state, lock,
        )
        chinese = apply_result_event(
            overlay, config, ("history_update", 0, "你好。", 1, 1), state, lock,
        )

        self.assertEqual(english["kind"], "committed_english")
        self.assertEqual(english["key"], (0, 1))
        self.assertEqual(chinese["kind"], "committed_chinese")
        self.assertGreaterEqual(chinese["applied_at"], english["applied_at"])

    def test_worker_errors_are_displayed_when_the_queue_contains_errors(self):
        errors = queue.Queue()
        errors.put(("translation", RuntimeError("failed")))

        class Overlay:
            status = None

            def set_status(self, text):
                self.status = text

        overlay = Overlay()

        processed = drain_worker_errors(errors, overlay)

        self.assertEqual(processed, 1)
        self.assertIn("translation", overlay.status)
        self.assertIn("failed", overlay.status)
        self.assertTrue(errors.empty())

    def test_worker_error_drain_displays_all_pending_errors(self):
        errors = queue.Queue()
        errors.put(("asr", RuntimeError("decode failed")))
        errors.put(("translation", RuntimeError("model failed")))

        class Overlay:
            messages = None

            def __init__(self):
                self.messages = []

            def set_status(self, text):
                self.messages.append(text)

        overlay = Overlay()
        processed = drain_worker_errors(errors, overlay)

        self.assertEqual(processed, 2)
        self.assertEqual(len(overlay.messages), 2)
        self.assertIn("decode failed", overlay.messages[0])
        self.assertIn("model failed", overlay.messages[1])

    def test_transient_events_are_shed_at_the_soft_limit(self):
        result_queue = queue.Queue()
        result_queue.put(("history_add", 0, "Committed.", {}, 1))

        accepted = enqueue_result_event(
            result_queue, ("english", "Live preview", "", None, None), 1
        )

        self.assertFalse(accepted)
        self.assertEqual(result_queue.qsize(), 1)

    def test_history_events_are_retained_above_the_soft_limit(self):
        result_queue = queue.Queue()
        result_queue.put(("history_add", 0, "First.", {}, 1))
        result_queue.put(("history_update", 0, "第一句。", 1, 1))

        accepted = enqueue_result_event(
            result_queue, ("history_add", 1, "Second.", {}, 1), 2
        )

        self.assertTrue(accepted)
        self.assertEqual(
            [result_queue.get_nowait()[0] for _ in range(3)],
            ["history_add", "history_update", "history_add"],
        )

    def test_history_revision_and_review_events_are_not_disposable(self):
        for event in ("history_original_update", "history_review"):
            with self.subTest(event=event):
                result_queue = queue.Queue()
                result_queue.put(("history_add", 0, "First.", {}, 1))

                self.assertTrue(
                    enqueue_result_event(result_queue, (event, 0, "", None, 1), 1)
                )
                self.assertEqual(result_queue.qsize(), 2)


if __name__ == "__main__":
    unittest.main()
