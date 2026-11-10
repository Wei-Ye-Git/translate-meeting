"""Translate worker result messages into overlay updates."""

from i18n import t
from translation_queue import is_live_translation_current


def apply_result_event(overlay, config, item, live_state, live_state_lock):
    event, first, second, fourth, event_version = item
    if event == "english":
        overlay.append_english(first)
        overlay.update_current(first, "")
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
    elif event == "history_update":
        visible_index = first - overlay.history_offset
        if visible_index >= 0:
            overlay.update_history(visible_index, second, version=fourth)
        overlay.set_status(t("status.translated_sentence", index=first + 1))
    elif event == "history_original_update":
        visible_index = first - overlay.history_offset
        if visible_index >= 0:
            overlay.update_history_original(
                visible_index, second, metadata=fourth, version=event_version
            )
        overlay.set_status(t("status.corrected_sentence", index=first + 1))
    elif event == "live_translation":
        with live_state_lock:
            is_current = is_live_translation_current(live_state, first, fourth)
        if is_current:
            overlay.update_current(fourth, second)
    elif event == "live_reset":
        overlay.update_current("", "")
    elif event == "history_review":
        visible_index = first - overlay.history_offset
        if visible_index >= 0:
            overlay.mark_history_for_review(visible_index)
        overlay.set_status(t("status.review_sentence", index=first + 1))
    elif event == "status":
        overlay.set_status(first)
