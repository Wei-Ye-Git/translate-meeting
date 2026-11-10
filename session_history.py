"""Render and persist transcript snapshots and correction history."""

import json
import os


def render_history(records, metadata, overlay, translate):
    lines = []
    for index, (original, translated) in enumerate(records):
        item = metadata[index]
        timestamp = (
            f"[{overlay._format_timestamp(item['start'])}-"
            f"{overlay._format_timestamp(item['end'])}] "
        )
        review = translate("ui.review_marker") if item.get("review") else ""
        corrected = translate("ui.corrected_marker") if item.get("corrected") else ""
        lines.append(
            f"{timestamp}{review}{corrected}{translate('ui.english_label')}"
            f"{original}\n{translate('ui.chinese_label')}{translated}\n"
        )
        if item.get("previous_original"):
            lines.append(
                translate("export.revision_english") + item["previous_original"] + "\n"
            )
        if item.get("previous_translation"):
            lines.append(
                translate("export.revision_chinese") + item["previous_translation"] + "\n"
            )
        lines.append("\n")
    return "".join(lines)


def atomic_write_history(path, records, metadata, overlay, translate):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as output:
        output.write(render_history(records, metadata, overlay, translate))
    os.replace(temporary_path, path)


def append_revision(path, entry):
    with open(path, "a", encoding="utf-8") as output:
        output.write(json.dumps(entry, ensure_ascii=False) + "\n")
