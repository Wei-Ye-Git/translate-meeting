"""Correlate subtitle capture, translation, and UI presentation timestamps."""

import threading
import math


def record_capture_time_for_event(event, metadata, captured_at_for_audio_time):
    """Return capture time only for events with valid record metadata."""
    if event == "history_review":
        return None
    metadata = metadata or {}
    captured_at = metadata.get("captured_at_monotonic")
    if captured_at is not None:
        return captured_at
    end = metadata.get("end")
    return captured_at_for_audio_time(end) if end is not None else None


class PipelineMetrics:
    def __init__(self):
        self._records = {}
        self._live = {}
        self._samples = {
            "preview_english": [], "committed_english": [],
            "committed_chinese": [], "live_chinese": [],
        }
        self._lock = threading.Lock()

    def remember_record(self, record_id, version, captured_at, published_at=None):
        if captured_at is None:
            return
        with self._lock:
            self._records[(record_id, version)] = {
                "captured_at": captured_at,
                "english_published_at": published_at,
                "translation_published_at": None,
            }
            self._prune(self._records)

    def remember_live(self, revision, captured_at):
        if captured_at is None:
            return
        with self._lock:
            self._live[revision] = {
                "captured_at": captured_at,
                "translation_published_at": None,
            }
            self._prune(self._live)

    def remember_preview(self, revision, captured_at):
        if captured_at is None:
            return
        with self._lock:
            self._live[revision] = {
                "captured_at": captured_at,
                "translation_published_at": None,
            }

    def discard_preview(self, revision):
        with self._lock:
            self._live.pop(revision, None)

    def clear_live(self):
        with self._lock:
            self._live.clear()

    def english_published(self, record_id, version, published_at):
        with self._lock:
            timing = self._records.get((record_id, version))
            if timing:
                timing["english_published_at"] = published_at

    def presented_english(self, kind, key, applied_at):
        with self._lock:
            timing = self._find(kind, key)
            if not timing:
                return None
            sample = {
                "subtitle_kind": kind,
                "audio_to_ui_seconds": max(0.0, applied_at - timing["captured_at"]),
                "ui_queue_seconds": (
                    max(0.0, applied_at - timing["english_published_at"])
                    if timing.get("english_published_at") is not None else None
                ),
            }
            self._samples[kind].append(sample)
            if kind == "preview_english":
                self._live.pop(key, None)
            return sample

    def presented_committed_english(self, key, applied_at):
        with self._lock:
            timing = self._records.get(key)
            if not timing:
                return None
            sample = {
                "subtitle_kind": "committed_english",
                "audio_to_ui_seconds": max(0.0, applied_at - timing["captured_at"]),
                "ui_queue_seconds": (
                    max(0.0, applied_at - timing["english_published_at"])
                    if timing.get("english_published_at") is not None else None
                ),
            }
            self._samples["committed_english"].append(sample)
            return sample

    def clear_record(self, key):
        with self._lock:
            self._records.pop(key, None)

    def translation_published(self, kind, key, published_at):
        with self._lock:
            timing = self._find(kind, key)
            if timing:
                timing["translation_published_at"] = published_at

    def presented(self, kind, key, applied_at, *, published_at=None):
        with self._lock:
            timing = self._find(kind, key)
            if not timing:
                return None
            captured_at = timing["captured_at"]
            published = published_at or timing.get("translation_published_at")
            sample = {
                "subtitle_kind": kind,
                "audio_to_ui_seconds": max(0.0, applied_at - captured_at),
                "ui_queue_seconds": max(0.0, applied_at - published) if published else None,
            }
            self._samples[kind].append(sample)
            if kind == "committed_chinese":
                self._records.pop(key, None)
            elif kind == "live_chinese":
                self._live.pop(key, None)
            return sample

    def summary(self):
        with self._lock:
            return {key: list(values) for key, values in self._samples.items()}

    @staticmethod
    def percentile_summary(samples):
        summary = {}
        for kind, values in samples.items():
            latencies = [item["audio_to_ui_seconds"] for item in values]
            ui_queue = [
                item["ui_queue_seconds"] for item in values
                if item["ui_queue_seconds"] is not None
            ]
            summary[kind] = {
                "count": len(latencies),
                "audio_to_ui_p50_seconds": _percentile(latencies, 0.5),
                "audio_to_ui_p95_seconds": _percentile(latencies, 0.95),
                "ui_queue_p50_seconds": _percentile(ui_queue, 0.5),
                "ui_queue_p95_seconds": _percentile(ui_queue, 0.95),
            }
        return summary

    def _find(self, kind, key):
        return (
            self._live.get(key)
            if kind in ("preview_english", "live_chinese")
            else self._records.get(key)
        )

    @staticmethod
    def _prune(values, limit=5000):
        while len(values) > limit:
            values.pop(next(iter(values)))


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 3)
