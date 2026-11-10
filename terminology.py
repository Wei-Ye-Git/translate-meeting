import json
import os
import re
import threading
from copy import deepcopy
from queue import Empty, SimpleQueue

from i18n import t


class Terminology:
    def __init__(self, entries):
        if not isinstance(entries, dict):
            raise ValueError(t("error.invalid_glossary"))
        for term, entry in entries.items():
            if (
                not isinstance(term, str)
                or not term.strip()
                or not isinstance(entry, dict)
                or not isinstance(entry.get("target"), str)
                or not entry.get("target", "").strip()
                or not isinstance(entry.get("aliases", []), list)
                or not isinstance(entry.get("asr_variants", []), list)
                or not isinstance(entry.get("context_terms", []), list)
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in entry.get("aliases", [])
                )
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in entry.get("asr_variants", [])
                )
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in entry.get("context_terms", [])
                )
                or (
                    "asr_always" in entry
                    and not isinstance(entry["asr_always"], bool)
                )
            ):
                raise ValueError(t("error.invalid_glossary_entry", term=term))
        self.entries = deepcopy(entries)

    @classmethod
    def from_file(cls, path):
        with open(path, "r", encoding="utf-8") as source:
            return cls(json.load(source))

    @classmethod
    def from_files(cls, paths):
        entries = {}
        for path in paths:
            if not path or not os.path.isfile(path):
                continue
            with open(path, "r", encoding="utf-8") as source:
                loaded = json.load(source)
            if not isinstance(loaded, dict):
                raise ValueError(t("error.invalid_glossary"))
            entries.update(loaded)
        return cls(entries)

    def matching_terms(self, text):
        lowered = text.lower()
        matches = []
        for term, entry in self.entries.items():
            candidates = [(term, term)] + [
                (item, term) for item in entry.get("asr_variants", [])
            ]
            matched = []
            for candidate, _ in candidates:
                if not candidate or not self._context_allows_variant(
                    candidate, term, lowered, observed_text=text
                ):
                    continue
                match = re.search(
                    r"(?<![A-Za-z0-9])" + re.escape(candidate.lower()) + r"(?![A-Za-z0-9])",
                    lowered,
                )
                if match:
                    matched.append((candidate, match.start(), match.end()))
            if matched:
                longest = max(matched, key=lambda item: len(item[0]))
                matches.append((term, len(longest[0]), longest[1], longest[2]))
        ordered = sorted(matches, key=lambda item: item[1], reverse=True)
        selected = []
        for term, match_length, match_start, match_end in ordered:
            shadowed = any(
                selected_term != term
                and other_start == match_start
                and (
                    other_end > match_end
                    or (other_end == match_end and len(selected_term) > len(term))
                )
                for selected_term, _, other_start, other_end in ordered
            )
            if shadowed:
                continue
            selected.append((term, match_length))
        return [term for term, _ in selected]

    def _context_allows_variant(self, candidate, term, lowered_text, observed_text=None):
        entry = self.entries[term]
        if candidate not in entry.get("asr_variants", []):
            return True
        context_terms = entry.get("context_terms", [])
        if not context_terms:
            return True
        return any(
            re.search(
                r"(?<![A-Za-z0-9])" + re.escape(context.lower()) + r"(?![A-Za-z0-9])",
                lowered_text,
            )
            for context in context_terms
        )

    def asr_prompt(self, text, max_terms=30, include_always=True):
        matched = list(self.matching_terms(text))
        selected = []
        if include_always:
            for term, entry in self.entries.items():
                if entry.get("asr_always"):
                    selected.append(term)
        selected.extend(term for term in matched if term not in selected)
        return ", ".join(selected[:max_terms])

    def global_prompt(self, max_terms=30):
        always = [term for term, entry in self.entries.items() if entry.get("asr_always")]
        others = sorted(
            (term for term in self.entries if term not in always),
            key=len,
            reverse=True,
        )
        terms = (always + others)[:max_terms]
        return ", ".join(terms)

    def normalize_source(self, text):
        result = text
        matches = []
        lowered = text.lower()
        for term, entry in self.entries.items():
            candidates = [term] + entry.get("asr_variants", [])
            for candidate in candidates:
                if candidate and self._context_allows_variant(
                    candidate, term, lowered, observed_text=text
                ) and re.search(
                    rf"(?<![A-Za-z0-9]){re.escape(candidate.lower())}(?![A-Za-z0-9])",
                    lowered,
                ):
                    matches.append((candidate, term))
        for candidate, term in sorted(matches, key=lambda item: len(item[0]), reverse=True):
            result = re.sub(
                rf"(?<![A-Za-z0-9]){re.escape(candidate)}(?![A-Za-z0-9])",
                lambda _: term,
                result,
                flags=re.IGNORECASE,
            )
        return result

    def apply(self, source, translated):
        result = translated
        if not source or not translated:
            return result
        for term in self.matching_terms(source):
            entry = self.entries[term]
            target = entry["target"]
            for alias in sorted(entry.get("aliases", []), key=len, reverse=True):
                if alias:
                    result = re.sub(
                        re.escape(alias), lambda _: target, result, flags=re.IGNORECASE
                    )
            for candidate in [term] + entry.get("asr_variants", []):
                if candidate:
                    result = re.sub(
                        rf"(?<![A-Za-z0-9]){re.escape(candidate)}(?![A-Za-z0-9])",
                        lambda _: target,
                        result,
                        flags=re.IGNORECASE,
                    )
        return result


class TerminologyManager:
    """Publish immutable-by-convention glossary snapshots at segment boundaries."""

    def __init__(self, terminology, file_paths=()):
        self._lock = threading.Lock()
        self._snapshot = terminology
        self._snapshot.version = 1
        self._updates = SimpleQueue()
        self.last_reload_error = None
        self._failed_file_signature = None
        self.last_update_error = None
        self.file_paths = tuple(path for path in file_paths if path)
        self._file_signature = self._signature()

    @classmethod
    def from_files(cls, paths):
        paths = tuple(paths)
        return cls(Terminology.from_files(paths), file_paths=paths)

    def snapshot(self):
        with self._lock:
            return self._snapshot

    def __getattr__(self, name):
        return getattr(self.snapshot(), name)

    def queue_update(self, operation, term=None, entry=None):
        """Queue add/replace/delete; updates become visible on apply_pending()."""
        if operation not in {"add", "replace", "delete"}:
            raise ValueError(t("error.unsupported_glossary_operation"))
        if not isinstance(term, str) or not term.strip():
            raise ValueError(t("error.glossary_term_required"))
        if operation != "delete" and not isinstance(entry, dict):
            raise ValueError(t("error.glossary_entry_required"))
        if operation != "delete":
            Terminology({term: entry})
        self._updates.put((operation, term, deepcopy(entry)))

    def _signature(self):
        signature = []
        for path in self.file_paths:
            try:
                stat = os.stat(path)
                signature.append((path, stat.st_mtime_ns, stat.st_size))
            except OSError:
                signature.append((path, None, None))
        return tuple(signature)

    def poll_files(self):
        """Queue a validated full reload when a glossary file changes."""
        signature = self._signature()
        if (
            signature == self._file_signature
            or signature == self._failed_file_signature
        ):
            return False
        try:
            replacement = Terminology.from_files(self.file_paths)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.last_reload_error = error
            self._failed_file_signature = signature
            return False
        self._updates.put(("replace_all", None, replacement.entries))
        self._file_signature = signature
        self._failed_file_signature = None
        self.last_reload_error = None
        return True

    def apply_pending(self):
        with self._lock:
            updates = []
            while True:
                try:
                    updates.append(self._updates.get_nowait())
                except Empty:
                    break
            if not updates:
                return self._snapshot
            entries = deepcopy(self._snapshot.entries)
            rejected = []
            full_reloads = [update for update in updates if update[0] == "replace_all"]
            if full_reloads:
                entries = deepcopy(full_reloads[-1][2])
            for operation, term, entry in updates:
                if operation == "replace_all":
                    continue
                if operation == "delete":
                    entries.pop(term, None)
                elif operation == "add" and term in entries:
                    rejected.append(ValueError(
                        t("error.duplicate_glossary_term", term=term)
                    ))
                else:
                    entries[term] = entry
            try:
                next_snapshot = Terminology(entries)
            except ValueError as error:
                self.last_update_error = error
                return self._snapshot
            self.last_update_error = rejected[0] if rejected else None
            if entries == self._snapshot.entries:
                return self._snapshot
            next_snapshot.version = self._snapshot.version + 1
            self._snapshot = next_snapshot
            return next_snapshot
