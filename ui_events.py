"""Translate worker result messages into overlay updates."""

import queue
import time

from i18n import t
from translation_queue import is_live_translation_current


DISPOSABLE_RESULT_EVENTS = frozenset(
    {"english", "status", "live_translation"}
)


def enqueue_result_event(result_queue, item, soft_limit):
    """Keep transcript events lossless while shedding transient UI updates."""
    if (
        soft_limit > 0
        and result_queue.qsize() >= soft_limit
        and item[0] in DISPOSABLE_RESULT_EVENTS
    ):
        return False
    result_queue.put(item)
    return True


def drain_worker_errors(error_queue, overlay):
    """Display every pending worker failure on the UI thread."""
    processed = 0
    while True:
        try:
            name, error = error_queue.get_nowait()
        except queue.Empty:
            return processed
        error_queue.task_done()
        overlay.set_status(t("status.worker_failed", worker=name, error=error))
        processed += 1


def apply_result_event(overlay, config, item, live_state, live_state_lock):
    event, first, second, fourth, event_version = item
    if event == "english":
        overlay.append_english(first)
        overlay.update_current(first, "")
        if isinstance(fourth, dict):
            return {
                "kind": "preview_english", "key": fourth.get("pipeline_id"),
                "applied_at": time.monotonic(),
            }
    elif event == "history_add":
        visible_index = first - overlay.history_offset
        if len(overlay.history) >= config.max_overlay_records:
            overlay.append_history(first, second, "", metadata=fourth)
        elif visible_index < len(overlay.history):
            overlay.update_history_original(
                visible_index, second, metadata=fourth, version=event_version
            )
        elif visible_index == len(overlay.history):
            overlay.append_history(first, second, "", metadata=fourth)
        overlay.set_status(
            t("status.review_fragment")
            if fourth and fourth.get("review")
            else t("status.processing")
        )
        return {
            "kind": "committed_english", "key": (first, event_version),
            "applied_at": time.monotonic(),
        }
    elif event == "history_update":
        visible_index = first - overlay.history_offset
        if visible_index >= 0:
            overlay.update_history(visible_index, second, version=fourth)
        overlay.set_status(t("status.translated_sentence", index=first + 1))
        return {
            "kind": "committed_chinese", "key": (first, event_version),
            "applied_at": time.monotonic(),
        }
    elif event == "history_original_update":
        visible_index = first - overlay.history_offset
        if visible_index >= 0:
            overlay.update_history_original(
                visible_index, second, metadata=fourth, version=event_version
            )
        overlay.set_status(t("status.corrected_sentence", index=first + 1))
        return {
            "kind": "committed_english", "key": (first, event_version),
            "applied_at": time.monotonic(),
        }
    elif event == "history_window_update":
        overlay.update_history_window(first)
    elif event == "history_translation_hold":
        overlay.hold_history_translation(first)
    elif event == "live_translation":
        with live_state_lock:
            is_current = is_live_translation_current(live_state, first, fourth)
        if is_current:
            overlay.update_current(fourth, second)
            return {
                "kind": "live_chinese", "key": first,
                "applied_at": time.monotonic(),
            }
    elif event == "live_reset":
        overlay.update_current("", "")
    elif event == "history_review":
        visible_index = first - overlay.history_offset
        if 0 <= visible_index < len(overlay.history) and (
            event_version is None
            or event_version >= overlay.history_metadata[visible_index].get("version", 1)
        ):
            overlay.mark_history_for_review(visible_index)
            overlay.set_status(t("status.review_sentence", index=first + 1))
    elif event == "status":
        overlay.set_status(first)
