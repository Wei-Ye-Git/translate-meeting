import re
from dataclasses import dataclass
from difflib import SequenceMatcher


SENTENCE_END = re.compile(r"(.+?[.!?。！？](?:[\"'”’)]*)?)(?:\s+|$)", re.DOTALL)


@dataclass(frozen=True)
class TimedWord:
    start: float
    end: float
    text: str
    probability: float = None


def words_to_text(words):
    text = " ".join(word.text.strip() for word in words if word.text.strip())
    return re.sub(r"\s+([,.!?;:])", r"\1", text).strip()


def words_before(words, boundary):
    return [word for word in words if word.start < boundary]


def words_in_time_range(words, start, end):
    return [word for word in words if word.end > start and word.start < end]


def shift_word_times(words, offset_seconds):
    return [
        TimedWord(word.start + offset_seconds, word.end + offset_seconds, word.text)
        for word in words
    ]


ACKNOWLEDGEMENTS = {
    "okay", "all right", "yes", "no", "yeah", "yep", "right",
    "thank you", "thanks", "sure", "got it", "good morning",
    "good afternoon", "hello", "hi", "bye", "goodbye",
    "thank you very much", "thank you so much", "no problem", "sounds good",
    "of course", "see you"
}

INCOMPLETE_ENDINGS = {
    "a", "an", "and", "as", "at", "before", "because", "but", "by",
    "for", "from", "her", "his", "if", "in", "into", "is", "it",
    "like", "more", "my", "of", "on", "or", "our", "than", "that",
    "the", "their", "this", "those", "to", "under", "was", "we", "were",
    "when", "where", "which", "while", "with", "without", "you", "your",
}


def is_sentence_complete(words, min_words=7):
    if not words:
        return False
    text = clean_repetitions(words_to_text(words)).strip()
    cleaned = re.sub(r"[^\w']+", " ", text).strip().lower()
    if cleaned in ACKNOWLEDGEMENTS or any(cleaned.startswith(ack + " ") for ack in {"thank you", "good morning", "good afternoon"}):
        return True
    word_count = len(re.findall(r"[A-Za-z0-9']+", text))
    has_end_punct = re.search(r"[.!?。！？]$", text) is not None
    if has_end_punct and word_count >= min_words:
        return True
    if word_count >= 14:
        return True
    return False


def is_likely_incomplete(words):
    """Return whether a pause probably occurred inside the sentence."""
    if not words:
        return True
    text = words[-1].text.strip().lower()
    if re.search(r"[.!?。！？][\"'”’)]*$", text):
        return False
    text = re.sub(r"[^a-z0-9']+$", "", text)
    return text in INCOMPLETE_ENDINGS


def merge_incomplete_sentences(sentences):
    if not sentences:
        return []

    merged = [list(sentences[0])]
    continuation_words = {
        "and", "but", "because", "for", "if", "so", "to", "that", "which",
        "important", "available", "necessary", "possible", "when", "while",
        "then", "there", "this", "these", "those", "with", "without",
        "we", "they", "it", "he", "she", "i", "you", "or", "as", "about",
        "also", "like", "such", "from", "by", "into", "over", "after"
    }

    acknowledgements = {
        "okay", "all right", "yes", "no", "yeah", "yep", "right",
        "thank you", "thanks", "sure", "got it", "good morning",
        "good afternoon", "hello", "hi", "bye", "goodbye"
    }

    for sentence in sentences[1:]:
        previous_text = words_to_text(merged[-1])
        first_word = sentence[0].text.lower().strip(" ,;:") if sentence else ""
        previous_words = re.findall(r"[A-Za-z0-9']+", previous_text)
        last_previous = previous_words[-1].lower() if previous_words else ""
        sentence_length = len(re.findall(r"[A-Za-z0-9']+", words_to_text(sentence)))
        last_previous_tail = previous_text.rstrip(".")[-3:].lower()
        cleaned_previous = re.sub(r"[^\w']+", " ", previous_text).strip().lower()

        if cleaned_previous in acknowledgements:
            merged.append(list(sentence))
            continue

        should_merge = (
            (
                len(previous_words) >= 2
                and first_word in continuation_words
                and sentence_length <= 8
                and (
                    last_previous in {"is", "are", "was", "were", "very", "more", "the", "a", "an", "to", "of", "in", "on", "for", "with", "at", "and"}
                    or last_previous_tail in {"if", "to", "of", "in", "on", "at", "so", "we"}
                )
            )
            or (
                len(previous_words) <= 4
                and (first_word in continuation_words or not re.search(r"[.!?]$", previous_text.strip()))
                and sentence_length <= 16
            )
        )

        if should_merge:
            merged[-1].extend(sentence)
        else:
            merged.append(list(sentence))
    return merged


def correction_is_usable(original, corrected):
    original_words = re.findall(r"[A-Za-z0-9']+", original)
    corrected_words = re.findall(r"[A-Za-z0-9']+", corrected)
    if not corrected.strip() or not corrected_words:
        return False
    similarity = SequenceMatcher(
        None,
        [word.lower() for word in original_words],
        [word.lower() for word in corrected_words],
    ).ratio()
    if similarity < 0.7:
        return False
    # A delayed pass must not replace a complete transcript with a shorter
    # fragment when its time window missed a boundary word.
    minimum_words = len(original_words)
    minimum_chars = max(1, int(len(original.strip()) * 0.75))
    return len(corrected_words) >= minimum_words and len(corrected.strip()) >= minimum_chars


def correction_is_confident(original, corrected, original_confidences, corrected_confidences):
    original_text = re.sub(r"[^\w']+", " ", original).strip().lower()
    corrected_text = re.sub(r"[^\w']+", " ", corrected).strip().lower()
    if original_text == corrected_text:
        return True
    original_scores = [score for score in original_confidences if score is not None]
    corrected_scores = [score for score in corrected_confidences if score is not None]
    if not original_scores or not corrected_scores:
        return False
    original_average = sum(original_scores) / len(original_scores)
    corrected_average = sum(corrected_scores) / len(corrected_scores)
    return corrected_average >= original_average + 0.03


def should_schedule_correction(words, min_words=4, min_duration_seconds=0.8):
    """Avoid contextual re-decoding of fragments too short to constrain ASR."""
    if not words:
        return False
    text = words_to_text(words)
    word_count = len(re.findall(r"[A-Za-z0-9']+", text))
    duration = words[-1].end - words[0].start
    return word_count >= min_words and duration >= min_duration_seconds


def has_valid_time_span(words, max_word_seconds=5.0, max_sentence_seconds=15.0):
    if not words:
        return False
    if any(
        not (word.start == word.start and word.end == word.end)
        or word.end <= word.start
        or word.end - word.start > max_word_seconds
        for word in words
    ):
        return False
    return (
        words[-1].end - words[0].start <= max_sentence_seconds
        and any(word.end <= max_word_seconds + words[0].start for word in words)
    )


def select_timed_words_for_target(
    words,
    target_start,
    target_end,
    max_word_seconds=5.0,
    boundary_padding_seconds=0.25,
):
    selected = [
        word for word in words
        if word.end > word.start
        and word.end - word.start <= max_word_seconds
        and word.end > target_start - boundary_padding_seconds
        and word.start < target_end + boundary_padding_seconds
    ]
    return selected


def select_timed_words_for_correction(
    words,
    target_start,
    target_end,
    max_word_seconds=5.0,
    boundary_padding_seconds=0.2,
):
    """Keep target words and words whose midpoints are within boundary tolerance."""
    return [
        word for word in words
        if word.end > word.start
        and word.end - word.start <= max_word_seconds
        and target_start - boundary_padding_seconds
        <= (word.start + word.end) / 2
        <= target_end + boundary_padding_seconds
    ]


def should_finalize_waiting_continuation(words):
    if not words:
        return False
    text = words_to_text(words)
    if not is_likely_incomplete(words):
        return is_recordable_sentence(text)
    if is_recordable_sentence(text):
        return True
    return False


def finalize_pending_group(words, waiting_for_continuation):
    if not words:
        return False
    if waiting_for_continuation:
        return should_finalize_waiting_continuation(words)
    return not is_likely_incomplete(words)


def trim_trailing_silence_chunks(chunks, silence_count):
    if silence_count <= 0:
        return chunks
    return chunks[:-silence_count]


def filter_uncommitted_words(words, committed_until):
    return [word for word in words if word.end > committed_until]


def should_hold_after_waiting_silence(words, silence_duration, required_silence):
    if not words or silence_duration >= required_silence:
        return False
    text = words_to_text(words)
    return is_likely_incomplete(words) or not is_recordable_sentence(text)


def should_finalize_after_waiting_timeout(words):
    return bool(words) and is_recordable_sentence(words_to_text(words))


def remove_chunk_overlap_words(previous, current):
    previous_words = [re.sub(r"[^\w']", "", word.text.strip().lower()) for word in previous]
    current_words = [re.sub(r"[^\w']", "", word.text.strip().lower()) for word in current]
    if not previous_words or not current_words:
        return current
    max_overlap = min(12, len(previous_words), len(current_words))
    for size in range(max_overlap, 1, -1):
        if previous_words[-size:] == current_words[:size]:
            return current[size:]
    return current


def split_timed_sentences(words):
    sentences = []
    remaining = []
    sentence = []
    for i, word in enumerate(words):
        sentence.append(word)
        if re.search(r"[.!?。！？][\"'”’)]*$", word.text.strip()):
            sentences.append(sentence)
            sentence = []
        elif i + 1 < len(words) and (words[i + 1].start - word.end >= 0.85) and len(sentence) >= 3:
            sentences.append(sentence)
            sentence = []
    if sentence:
        remaining = sentence
    return sentences, remaining


def split_sentences(text):
    sentences = []
    remaining = text.strip()
    while remaining:
        match = SENTENCE_END.match(remaining)
        if not match:
            break
        sentences.append(match.group(1).strip())
        remaining = remaining[match.end():].strip()
    return sentences, remaining


def is_meaningful_sentence(text):
    cleaned = clean_repetitions(text or "").strip()
    if not cleaned:
        return False
    lowered = cleaned.lower()
    filler = re.compile(r"\b(?:uh|um|er|erm|you know|i mean)\b", re.IGNORECASE)
    if filler.search(lowered):
        words = re.findall(r"[A-Za-z0-9']+", cleaned)
        if not words:
            return False
        filler_words = {"uh", "um", "er", "erm"}
        if all(word.lower() in filler_words for word in words):
            return False
    words = re.findall(r"[A-Za-z0-9']+", cleaned)
    if not words:
        return False
    if re.search(r"[.!?]$", cleaned):
        return len(words) >= 1
    return len(words) >= 1


def is_recordable_sentence(text, allow_single_word=False):
    """Reject obvious ASR fragments before they enter permanent history."""
    cleaned = clean_repetitions(text or "").strip()
    if not is_meaningful_sentence(cleaned):
        return False
    normalized = re.sub(r"[^\w']+", " ", cleaned).strip().lower()
    if normalized in ACKNOWLEDGEMENTS:
        return True
    words = re.findall(r"[A-Za-z0-9']+", cleaned)
    if allow_single_word and len(words) == 1:
        return True
    if len(words) < 2:
        return False
    return not is_likely_incomplete([TimedWord(0, 1, words[-1])])


def is_review_candidate(text):
    cleaned = clean_repetitions(text or "").strip()
    normalized = re.sub(r"[^\w']+", " ", cleaned).strip().lower()
    if not cleaned or normalized in ACKNOWLEDGEMENTS:
        return False
    if re.search(r"[.!?。！？]$", cleaned):
        return False
    return len(re.findall(r"[A-Za-z0-9']+", cleaned)) <= 4


def is_asr_anomaly(text):
    """Detect high-confidence decoding loops without rewriting the transcript."""
    if not text or not text.strip():
        return False

    if re.search(r"([A-Za-z])\1{11,}", text, re.IGNORECASE):
        return True

    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    if len(tokens) >= 6:
        for start in range(len(tokens)):
            for size in range(1, min(24, (len(tokens) - start) // 3) + 1):
                phrase = tokens[start:start + size]
                repetitions = 1
                position = start + size
                while position + size <= len(tokens) and tokens[position:position + size] == phrase:
                    repetitions += 1
                    position += size
                repeated_tokens = repetitions * size
                if size == 1 and len(phrase[0]) >= 3 and repetitions >= 8:
                    return True
                if (
                    size > 1
                    and len(tokens) >= 12
                    and repetitions >= 3
                    and repeated_tokens >= len(tokens) * 0.6
                ):
                    return True

    return False


def clean_repetitions(text):
    cleaned = text.strip()
    if not cleaned:
        return ""

    cleaned = re.sub(
        r"\b(?:uh|um|er|erm|like|you know|i mean|basically|actually|literally)\b(?:\s+\b(?:uh|um|er|erm|like|you know|i mean|basically|actually|literally)\b)+",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned)

    for size in range(6, 0, -1):
        atom = r"[A-Za-z0-9']+"
        phrase = rf"({atom}(?:[,\s]+{atom}){{{size - 1}}})" if size > 1 else rf"({atom})"
        cleaned = re.sub(
            rf"\b{phrase}(?:\s*[,.]?\s*\1)+\b", r"\1", cleaned, flags=re.IGNORECASE
        )

    cleaned = re.sub(r"(?:\s+[.,!?;:]){2,}", " ", cleaned)
    cleaned = re.sub(r"([,.!?;:])\1+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"([.?!])(?=\S)", r"\1 ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def remove_chunk_overlap(previous, current):
    previous_words = re.findall(r"[A-Za-z0-9']+", previous.lower())
    current_words = re.findall(r"[A-Za-z0-9']+", current.lower())
    if not previous_words or not current_words:
        return current.strip()
    for size in range(min(12, len(previous_words), len(current_words)), 1, -1):
        if previous_words[-size:] == current_words[:size]:
            match = re.match(rf"(?:\s*\S+){{{size}}}", current.strip())
            if match:
                return current[match.end():].strip(" ,;:-")
    return current.strip()


def remove_sentence_overlap(previous, current, min_overlap=5):
    """Remove a substantial repeated prefix from adjacent transcript records."""
    previous_words = re.findall(r"[A-Za-z0-9']+", previous.lower())
    current_words = re.findall(r"[A-Za-z0-9']+", current.lower())
    max_overlap = min(12, len(previous_words), len(current_words))
    for size in range(max_overlap, min_overlap - 1, -1):
        if previous_words[-size:] != current_words[:size]:
            continue
        match = re.match(rf"(?:\s*\S+){{{size}}}", current.strip())
        if match:
            return current.strip()[match.end():].strip(" ,;:-")
    return current.strip()


def remove_sentence_overlap_words(previous, current, min_overlap=5):
    """Remove a substantial repeated prefix while preserving word timestamps."""
    if isinstance(previous, str):
        previous_words = [
            re.sub(r"[^\w']", "", word) for word in re.findall(r"[A-Za-z0-9']+", previous.lower())
        ]
    else:
        previous_words = [re.sub(r"[^\w']", "", word.text.lower()) for word in previous]
    current_words = [re.sub(r"[^\w']", "", word.text.lower()) for word in current]
    max_overlap = min(12, len(previous_words), len(current_words))
    for size in range(max_overlap, min_overlap - 1, -1):
        if previous_words[-size:] == current_words[:size]:
            return current[size:]
    return current
