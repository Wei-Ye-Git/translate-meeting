"""Worker exception containment and peer independence tests."""

import threading
import unittest

from worker_runtime import WorkerRuntime


class WorkerRuntimeTests(unittest.TestCase):
    def test_uncaught_failure_is_reported_without_terminating_peer_worker(self):
        failures = []
        peer_finished = threading.Event()
        runtime = WorkerRuntime(lambda name, error: failures.append((name, str(error))))

        runtime.start("broken", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        runtime.start("peer", peer_finished.set)
        unfinished = runtime.join_all(timeout=2)

        self.assertEqual(unfinished, [])
        self.assertEqual(failures, [("broken", "boom")])
        self.assertTrue(peer_finished.is_set())

    def test_error_reporting_failure_is_also_contained(self):
        finished = threading.Event()
        runtime = WorkerRuntime(
            lambda name, error: (_ for _ in ()).throw(RuntimeError("reporting failed"))
        )

        runtime.start("broken", lambda: (_ for _ in ()).throw(ValueError("worker failed")))
        runtime.start("peer", finished.set)

        self.assertEqual(runtime.join_all(timeout=2), [])
        self.assertTrue(finished.is_set())


if __name__ == "__main__":
    unittest.main()
