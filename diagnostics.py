import json
import os
import threading
from collections import Counter
from datetime import datetime, timezone


class SessionDiagnostics:
    """Append compact machine-readable events for one meeting session."""

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._warnings = Counter()
        self._warning_details = {}

    def write_event(self, event, **fields):
        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            **fields,
        }
        try:
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as error:
            print(f"[diagnostics] 无法写入诊断日志 {self.path}: {error}")

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
