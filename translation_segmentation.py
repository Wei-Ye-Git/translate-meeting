"""Token-budget-aware source segmentation for long translation inputs."""

import re


DEFAULT_MAX_TOKENS = 384
DEFAULT_MIN_WORDS = 5
_CLAUSE_BOUNDARY = re.compile(
    r"(?<=[,;:])\s+|\s+(?=(?:and|but|because|while|which|that|although|however|therefore)\b)",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_long_text(text, token_count, max_tokens=DEFAULT_MAX_TOKENS,
                    min_words=DEFAULT_MIN_WORDS):
    """Split long text near clause boundaries and avoid tiny translation jobs."""
    text = " ".join((text or "").split())
    if not text:
        return []
    if token_count(text) <= max_tokens:
        return [text]

    clauses = []
    for sentence in _SENTENCE_BOUNDARY.split(text):
        clauses.extend(_CLAUSE_BOUNDARY.split(sentence))
    clauses = [part.strip() for part in clauses if part.strip()]

    chunks = []
    current = ""
    for clause in clauses:
        candidate = f"{current} {clause}".strip() if current else clause
        if token_count(candidate) <= max_tokens:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if token_count(clause) <= max_tokens:
            current = clause
            continue
        chunks.extend(_split_at_words(clause, token_count, max_tokens))
        current = ""
    if current:
        chunks.append(current)

    chunks = _merge_short_chunks(chunks, token_count, max_tokens, min_words)
    if any(token_count(chunk) > max_tokens for chunk in chunks):
        raise ValueError("Unable to split translation input within token budget")
    return chunks


def _split_at_words(text, token_count, max_tokens):
    words = text.split()
    chunks = []
    current = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and token_count(candidate) > max_tokens:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
        if token_count(" ".join(current)) > max_tokens:
            raise ValueError("A single token exceeds the translation budget")
    if current:
        chunks.append(" ".join(current))
    return chunks


def _merge_short_chunks(chunks, token_count, max_tokens, min_words):
    merged = list(chunks)
    index = 0
    while index < len(merged):
        if len(merged[index].split()) >= min_words or len(merged) == 1:
            index += 1
            continue
        neighbor_index = index - 1 if index > 0 else index + 1
        if neighbor_index >= len(merged):
            index += 1
            continue
        left_index = min(index, neighbor_index)
        words = (merged[left_index] + " " + merged[left_index + 1]).split()
        balanced = _balanced_word_split(words, token_count, max_tokens, min_words)
        if balanced:
            merged[left_index:left_index + 2] = balanced
            index = max(0, left_index - 1)
        else:
            index += 1
    return merged


def _balanced_word_split(words, token_count, max_tokens, min_words):
    midpoint = len(words) // 2
    candidates = sorted(
        range(min_words, len(words) - min_words + 1),
        key=lambda index: (abs(index - midpoint), index),
    )
    for split_at in candidates:
        left = " ".join(words[:split_at])
        right = " ".join(words[split_at:])
        if (
            token_count(left) <= max_tokens
            and token_count(right) <= max_tokens
        ):
            return [left, right]
    return None


def join_translated_segments(parts):
    """Join separately translated clauses without injecting phrase-level spaces."""
    result = ""
    for part in (item.strip() for item in parts if item and item.strip()):
        if result and result[-1] not in "，。！？；：,!?;:":
            result += "，"
        result += part
    return result
