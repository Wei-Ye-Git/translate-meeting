"""Run background workers with a per-thread exception boundary."""

import threading


class WorkerRuntime:
    """Own worker threads and route uncaught failures without stopping peers."""

    def __init__(self, on_error):
        self._on_error = on_error
        self._threads = {}
        self._lock = threading.Lock()

    def start(self, name, target, daemon=True):
        with self._lock:
            if name in self._threads:
                raise ValueError(f"Worker already started: {name}")
            thread = threading.Thread(
                target=self._run,
                args=(name, target),
                name=name,
                daemon=daemon,
            )
            self._threads[name] = thread
            thread.start()
            return thread

    def _run(self, name, target):
        try:
            target()
        except Exception as error:
            try:
                self._on_error(name, error)
            except Exception:
                # Error reporting must not escape the worker boundary either.
                pass

    def join_all(self, timeout=None):
        """Join workers and return names that are still alive, if any."""
        with self._lock:
            threads = tuple(self._threads.items())
        for _, thread in threads:
            thread.join(timeout=timeout)
        return [name for name, thread in threads if thread.is_alive()]

    def is_alive(self, name):
        with self._lock:
            thread = self._threads.get(name)
        return bool(thread and thread.is_alive())
