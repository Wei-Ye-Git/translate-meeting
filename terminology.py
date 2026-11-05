import json
import re


class Terminology:
    def __init__(self, entries):
        if not isinstance(entries, dict):
            raise ValueError("术语表必须是 JSON 对象")
        for term, entry in entries.items():
            if (
                not isinstance(term, str)
                or not term.strip()
                or not isinstance(entry, dict)
                or not isinstance(entry.get("target"), str)
                or not entry.get("target", "").strip()
                or not isinstance(entry.get("aliases", []), list)
                or not isinstance(entry.get("asr_variants", []), list)
            ):
                raise ValueError(f"术语配置缺少 target: {term}")
        self.entries = entries

    @classmethod
    def from_file(cls, path):
        with open(path, "r", encoding="utf-8") as source:
            return cls(json.load(source))

    def matching_terms(self, text):
        lowered = text.lower()
        matches = []
        for term, entry in self.entries.items():
            candidates = [(term, term)] + [
                (item, term) for item in entry.get("asr_variants", [])
            ]
            matched = []
            for candidate, _ in candidates:
                if not candidate:
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
                if candidate and re.search(
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
