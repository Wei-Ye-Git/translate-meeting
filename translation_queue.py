import queue


def publish_live_translation(text, min_words, state, pending):
    """Queue only changed stable text, replacing any older pending snapshot."""
    if len(text.split()) < min_words or text == state.get("text"):
        return None
    revision = state.get("revision", 0) + 1
    state["revision"] = revision
    state["text"] = text
    while True:
        try:
            pending.get_nowait()
            pending.task_done()
        except queue.Empty:
            break
    pending.put_nowait((revision, text))
    return revision


def take_latest(current, pending):
    """Return the newest queued live translation and discard stale snapshots."""
    latest = current
    while True:
        try:
            latest = pending.get_nowait()
            pending.task_done()
        except queue.Empty:
            return latest
