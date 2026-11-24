"""Conservative, explainable quality signals for ASR and sentence boundaries."""

import re

import numpy as np


FILLER_PHRASES = (
    "you know",
    "i mean",
    "uh",
    "um",
    "er",
    "erm",
    "ah",
    "hmm",
    "like",
    "basically",
    "actually",
)
WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")
TERMINAL_PATTERN = re.compile(r"[.!?][\"'”’)]*$")


def analyze_audio_window(audio):
    """Return compact signal-level evidence for an ASR input window."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.size == 0:
        return {
            "sample_count": 0,
            "rms": 0.0,
            "peak": 0.0,
            "clipped_fraction": 0.0,
        }
    absolute = np.abs(samples)
    return {
        "sample_count": int(samples.size),
        "rms": round(float(np.sqrt(np.mean(np.square(samples)))), 6),
        "peak": round(float(np.max(absolute)), 6),
        "clipped_fraction": round(float(np.mean(absolute >= 0.999)), 6),
    }


def _filler_count(text):
    count = 0
    for phrase in FILLER_PHRASES:
        count += len(re.findall(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE))
    return count


def analyze_timed_utterance(
    words,
    audio_duration=None,
    fast_words_per_second=3.0,
    long_pause_seconds=0.85,
    low_confidence_threshold=0.55,
):
    """Return metrics and candidate issues without changing ASR text.

    The metrics are diagnostics, not proof of omitted speech or incorrect audio.
    """
    words = list(words or [])
    text = " ".join(word.text.strip() for word in words if word.text.strip())
    tokens = WORD_PATTERN.findall(text)
    valid_words = [word for word in words if word.end > word.start]
    span = (
        max(word.end for word in valid_words) - min(word.start for word in valid_words)
        if valid_words else 0.0
    )
    measured_duration = span if span > 0 else max(0.0, audio_duration or 0.0)
    words_per_second = len(tokens) / measured_duration if measured_duration > 0 else 0.0

    filler_count = _filler_count(text)
    filler_ratio = filler_count / len(tokens) if tokens else 0.0
    pauses = [
        {
            "start": round(previous.end, 3),
            "end": round(current.start, 3),
            "seconds": round(current.start - previous.end, 3),
            "preceded_by_terminal_punctuation": bool(
                TERMINAL_PATTERN.search(previous.text.strip())
            ),
        }
        for previous, current in zip(words, words[1:])
        if current.start - previous.end >= long_pause_seconds
    ]
    low_confidence_words = [
        {"text": word.text, "start": round(word.start, 3),
         "end": round(word.end, 3), "probability": round(word.probability, 3)}
        for word in words
        if word.probability is not None
        and word.probability < low_confidence_threshold
    ]

    issues = []
    if words_per_second >= fast_words_per_second and len(tokens) >= 6:
        issues.append("possible_fast_speech")
    if filler_count >= 3 and filler_ratio >= 0.15:
        issues.append("filler_heavy")
    if pauses:
        issues.append("long_internal_pause")
    if any(not pause["preceded_by_terminal_punctuation"] for pause in pauses):
        issues.append("possible_pause_boundary_fragment")
    if low_confidence_words:
        issues.append("low_confidence_words")
    if len(tokens) <= 3 and (audio_duration or 0) >= 2.5:
        issues.append("possible_sparse_asr")
    if words and len(valid_words) != len(words):
        issues.append("invalid_word_timestamps")

    return {
        "issues": issues,
        "word_count": len(tokens),
        "audio_seconds": round(max(0.0, audio_duration or 0.0), 3),
        "timed_span_seconds": round(span, 3),
        "words_per_second": round(words_per_second, 3),
        "filler_count": filler_count,
        "filler_ratio": round(filler_ratio, 3),
        "long_pause_count": len(pauses),
        "long_pauses": pauses,
        "low_confidence_words": low_confidence_words,
    }


def classify_segmentation_boundaries(sentences, pause_threshold=0.85):
    """Describe likely pause-based cuts that lack terminal punctuation."""
    boundaries = []
    for previous, current in zip(sentences, sentences[1:]):
        if not previous or not current:
            continue
        gap = current[0].start - previous[-1].end
        if gap < pause_threshold:
            continue
        previous_text = " ".join(word.text.strip() for word in previous)
        if TERMINAL_PATTERN.search(previous_text.strip()):
            continue
        boundaries.append({
            "left_end": round(previous[-1].end, 3),
            "right_start": round(current[0].start, 3),
            "pause_seconds": round(gap, 3),
            "left_text": previous_text,
            "right_text": " ".join(word.text.strip() for word in current),
        })
    return boundaries
