import json
import os
import threading


_RESOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locales")
_DEFAULT_LOCALE = "zh-CN"
_lock = threading.Lock()
_resources = {}
_locale = os.environ.get("MEETING_LOCALE", _DEFAULT_LOCALE)


def set_locale(locale):
    global _locale
    with _lock:
        _locale = locale


def _load(locale):
    with _lock:
        if locale in _resources:
            return _resources[locale]
        path = os.path.join(_RESOURCE_DIR, f"{locale}.json")
        try:
            with open(path, "r", encoding="utf-8") as source:
                resource = json.load(source)
        except (OSError, ValueError):
            resource = {}
        _resources[locale] = resource
        return resource


def t(key, **params):
    resource = _load(_locale)
    messages = resource.get("messages", {})
    template = messages.get(key)
    if template is None and _locale != _DEFAULT_LOCALE:
        template = _load(_DEFAULT_LOCALE).get("messages", {}).get(key)
    if template is None:
        return key
    try:
        return template.format(**params)
    except KeyError:
        return key
    except ValueError:
        return key


def data(key, default=None):
    resource = _load(_locale)
    rules = resource.get("data", {})
    if key in rules:
        return rules[key]
    return _load(_DEFAULT_LOCALE).get("data", {}).get(key, default)


def configured_locales():
    try:
        return sorted(
            name[:-5]
            for name in os.listdir(_RESOURCE_DIR)
            if name.endswith(".json")
        )
    except OSError:
        return []
