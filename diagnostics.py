import json
import os
import threading
from collections import Counter
from datetime import datetime, timezone

from i18n import t


class SessionQualityStats:
    """Thread-safe counters and high-water marks for session quality signals."""

    def __init__(self):
        self._values = {}
        self._lock = threading.Lock()

    def increment(self, name, amount=1):
        with self._lock:
            self._values[name] = self._values.get(name, 0) + amount

    def set_max(self, name, value):
        with self._lock:
            self._values[name] = max(self._values.get(name, value), value)

    def snapshot(self):
        with self._lock:
            return dict(self._values)


class SessionDiagnostics:
    """Append compact machine-readable events for one meeting session."""

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._warnings = Counter()
        self._warning_details = {}
        self._quality_stats = SessionQualityStats()
        self._quality_stats.increment("asr_suspect_windows", 0)
        self._quality_stats.increment("translation_quality_retries", 0)
        self._quality_stats.increment("translation_retry_failures", 0)
        self._quality_stats.increment("translation_quality_unresolved", 0)
        self._quality_stats.increment("translation_context_fallbacks", 0)
        self._quality_stats.set_max("max_audio_backlog_seconds", 0.0)
        self._disabled = False

    def increment_quality(self, name, amount=1):
        self._quality_stats.increment(name, amount)

    def update_quality_max(self, name, value):
        self._quality_stats.set_max(name, value)

    def quality_snapshot(self):
        return self._quality_stats.snapshot()

    def write_event(self, event, **fields):
        if self._disabled:
            return False
        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            **fields,
        }
        try:
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except OSError as error:
            self._disabled = True
            try:
                print(t("error.diagnostics_write", path=self.path, error=error))
            except Exception:
                pass
            return False

    def record_warning(self, category, message, filename, lineno, count=1):
        key = (category, message, os.path.basename(filename), lineno)
        with self._lock:
            self._warnings[key] += count
            self._warning_details[key] = {
                "category": category,
                "message": message,
                "source": os.path.basename(filename),
                "line": lineno,
            }

    def write_warning_observation(self, category, message, filename, lineno, count, observed_range):
        self.write_event(
            "warning_observation",
            category=category,
            message=message,
            source=os.path.basename(filename),
            line=lineno,
            count=count,
            **(observed_range or {}),
        )

    def write_warning_summary(self, category, message, filename, lineno, count):
        self.write_event(
            "warning_summary",
            category=category,
            message=message,
            source=os.path.basename(filename),
            line=lineno,
            count=count,
        )

    def flush_warning_summary(self):
        with self._lock:
            summaries = [
                {**self._warning_details[key], "count": count}
                for key, count in self._warnings.items()
            ]
            self._warnings.clear()
            self._warning_details.clear()
        for summary in summaries:
            self.write_event("warning_summary", **summary)
