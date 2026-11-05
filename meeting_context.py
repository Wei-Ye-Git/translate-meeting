from collections import deque


def should_add_to_meeting_context(metadata):
    """Keep suspect and fragmentary ASR out of prompts; translation review is orthogonal."""
    metadata = metadata or {}
    reasons = list(metadata.get("review_reasons", []))
    if metadata.get("review_reason"):
        reasons.append(metadata["review_reason"])
    asr_review_reasons = {
        "asr_anomaly",
        "asr_implausible_word_rate",
        "asr_short_fragment",
        "short_asr_fragment",
        "implausible_word_rate",
    }
    return (
        not metadata.get("asr_suspect")
        and not metadata.get("asr_anomaly")
        and not asr_review_reasons.intersection(reasons)
    )


class MeetingContext:
    """Bounded, lightweight meeting state used to guide ASR and disambiguation."""

    def __init__(self, terminology, max_sentences=4, max_chars=480, max_terms=16):
        self.terminology = terminology
        self.max_sentences = max_sentences
        self.max_chars = max_chars
        self.max_terms = max_terms
        self.sentences = deque(maxlen=max_sentences)
        self.terms = deque(maxlen=max_terms)

    def add(self, text):
        text = " ".join((text or "").split())
        if not text:
            return
        self.sentences.append(text)
        for term in self.terminology.matching_terms(text):
            if term in self.terms:
                self.terms.remove(term)
            self.terms.append(term)

    def replace_recent(self, previous, corrected):
        corrected = " ".join((corrected or "").split())
        if not corrected:
            return
        sentences = list(self.sentences)
        try:
            index = len(sentences) - 1 - sentences[::-1].index(previous)
        except ValueError:
            return False
        sentences[index] = corrected
        self.sentences.clear()
        self.sentences.extend(sentences)
        old_terms = list(self.terms)
        self.terms.clear()
        for sentence in self.sentences:
            for term in self.terminology.matching_terms(sentence):
                if term in self.terms:
                    self.terms.remove(term)
                self.terms.append(term)
        for term in old_terms:
            if not any(term in self.terminology.matching_terms(s) for s in self.sentences):
                try:
                    self.terms.remove(term)
                except ValueError:
                    pass
        for term in self.terminology.matching_terms(corrected):
            if term in self.terms:
                self.terms.remove(term)
            self.terms.append(term)
        return True

    def snapshot(self):
        terms = list(self.terms)
        term_budget = max(0, self.max_chars - len("Terms: "))
        while terms and len(", ".join(terms)) > term_budget:
            terms.pop(0)
        term_section = "Terms: " + ", ".join(terms) if terms else ""
        sentences = list(self.sentences)
        prefix = (term_section + "\n") if term_section else ""
        available = max(0, self.max_chars - len(prefix) - len("Recent meeting context: "))
        retained = []
        used = 0
        for sentence in reversed(sentences):
            separator = 1 if retained else 0
            if used + len(sentence) + separator > available:
                break
            retained.insert(0, sentence)
            used += len(sentence) + separator
        context = "Recent meeting context: " + " ".join(retained) if retained else ""
        value = prefix + context
        if len(value) <= self.max_chars:
            return value
        if term_section:
            return term_section[:self.max_chars]
        return value[:self.max_chars]

    def asr_prompt_context(self):
        """Return recent spoken context without glossary or prompt labels."""
        retained = []
        used = 0
        for sentence in reversed(self.sentences):
            separator = 1 if retained else 0
            if used + len(sentence) + separator > self.max_chars:
                break
            retained.insert(0, sentence)
            used += len(sentence) + separator
        return " ".join(retained)

    def recent_sentences(self, limit=2):
        return list(self.sentences)[-limit:]
