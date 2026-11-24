import json
import os
import threading
import time
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
        self._event_sequence = 0
        self._started_monotonic = time.monotonic()
        self.session_id = os.path.basename(os.path.dirname(path))
        self._quality_stats = SessionQualityStats()
        self._quality_stats.increment("asr_suspect_windows", 0)
        self._quality_stats.increment("translation_quality_retries", 0)
        self._quality_stats.increment("translation_retry_failures", 0)
        self._quality_stats.increment("translation_quality_unresolved", 0)
        self._quality_stats.increment("translation_context_fallbacks", 0)
        self._quality_stats.increment("quality_issue_events", 0)
        self._quality_stats.set_max("max_audio_backlog_seconds", 0.0)
        self._quality_stats.set_max("max_result_queue_items", 0)
        self._quality_stats.set_max("max_pending_history_translations", 0)
        for name in (
            "capture_runtime_discontinuities",
            "capture_runtime_underflows",
            "capture_runtime_overflows",
            "audio_timeline_gap_events",
            "audio_timeline_gap_frames",
            "audio_queue_dropped_chunks",
            "audio_queue_dropped_frames",
            "audio_stale_chunks",
            "audio_stale_chunk_frames",
            "asr_skipped_audio_frames",
        ):
            self._quality_stats.increment(name, 0)
        self._disabled = False

    def increment_quality(self, name, amount=1):
        self._quality_stats.increment(name, amount)

    def update_quality_max(self, name, value):
        self._quality_stats.set_max(name, value)

    def quality_snapshot(self):
        return self._quality_stats.snapshot()

    def write_event(self, event, **fields):
        try:
            with self._lock:
                if self._disabled:
                    return False
                self._event_sequence += 1
                record = {
                    "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "session_id": self.session_id,
                    "event_sequence": self._event_sequence,
                    "elapsed_seconds": round(
                        time.monotonic() - self._started_monotonic, 3
                    ),
                    "thread": threading.current_thread().name,
                    "event": event,
                    **fields,
                }
                with open(self.path, "a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except OSError as error:
            with self._lock:
                self._disabled = True
            try:
                print(t("error.diagnostics_write", path=self.path, error=error))
            except Exception:
                pass
            return False

    def record_quality_issues(
        self, event, issues, start=None, end=None, text=None, metrics=None, **fields
    ):
        """Write one compact event per issue for easy aggregation."""
        written = True
        for issue in dict.fromkeys(issues or []):
            self.increment_quality(issue)
            self.increment_quality("quality_issue_events")
            written = self.write_event(
                event,
                issue=issue,
                start=start,
                end=end,
                text=text,
                metrics=metrics or {},
                **fields,
            ) and written
        return written

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

    def warning_counts(self):
        """Return accumulated warnings for inclusion in the session summary."""
        with self._lock:
            return {
                f"{category}:{message}": count
                for (category, message, _, _), count in self._warnings.items()
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
