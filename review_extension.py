"""Load the optional local transcript-review backend for this session."""

from importlib import import_module


def load_review_extension(config):
    if not config.realtime_copilot_enabled:
        return None, None

    try:
        backend = import_module("local_ai.realtime_copilot")
        config.configure_ai_timing()
        cli = backend.resolve_ai_cli(config.realtime_ai_cli)
    except (ImportError, OSError, ValueError) as error:
        print(f"[AI review] unavailable; continuing with local models: {ascii(str(error))}")
        return None, None
    print(f"[AI review] using {cli['name']}")
    return backend, cli
