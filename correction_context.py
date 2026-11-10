"""Context selection and stale-result guards for delayed ASR correction."""

from meeting_context import should_add_to_meeting_context


def correction_prompt(records, index, limit=3, metadata=None):
    metadata = metadata or []
    selected = []
    for item_index in range(index - 1, -1, -1):
        item_metadata = metadata[item_index] if item_index < len(metadata) else {}
        if not should_add_to_meeting_context(item_metadata):
            continue
        selected.append(records[item_index][0])
        if len(selected) >= limit:
            break
    return " ".join(reversed(selected))[-360:]


def correction_snapshot_is_current(record, metadata, expected_text, expected_version):
    return (
        record is not None
        and metadata is not None
        and metadata.get("version") == expected_version
        and record[0] == expected_text
    )
