import re


class StableText:
    """Commits words only after they survive consecutive ASR hypotheses."""

    def __init__(self):
        self.previous_words = []
        self.committed_words = []

    @staticmethod
    def _comparison_word(word):
        return re.sub(r"^[^\w']+|[^\w']+$", "", word).lower()

    def update(self, hypothesis):
        words = re.findall(r"\S+", (hypothesis or "").strip())
        if len(words) < len(self.committed_words) or any(
            self._comparison_word(old) != self._comparison_word(new)
            for old, new in zip(self.committed_words, words)
        ):
            self.previous_words = words
            return ""
        common_length = 0
        for previous, current in zip(self.previous_words, words):
            if self._comparison_word(previous) != self._comparison_word(current):
                break
            common_length += 1

        stable_length = max(common_length, len(self.committed_words))
        newly_stable = words[len(self.committed_words):stable_length]
        self.committed_words = words[:stable_length]
        self.previous_words = words
        return " ".join(newly_stable).strip()

    def finish(self, hypothesis=None):
        words = re.findall(r"\S+", (hypothesis or "").strip())
        if not words:
            words = self.previous_words
        newly_stable = words[len(self.committed_words):]
        self.committed_words = words
        return " ".join(newly_stable).strip()

    def reset(self):
        self.previous_words = []
        self.committed_words = []
