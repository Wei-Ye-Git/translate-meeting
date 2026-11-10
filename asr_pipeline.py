"""Bounded ASR-window construction and decoding for a meeting timeline."""

from dataclasses import dataclass

import numpy as np

from asr import build_asr_prompt
from audio_capture import extract_audio_range
from audio_preprocess import highpass
from text_processing import TimedWord, asr_output_issue, select_timed_words_for_target


@dataclass(frozen=True)
class ASRWindowResult:
    words: list
    audio: np.ndarray
    audio_start: int
    glossary_version: int
    latency_seconds: float
    audio_seconds: float
    issue: str | None = None
    primary_error: Exception | None = None
    retry_error: Exception | None = None


def transcribe_audio_window(
    source_chunks,
    recent_chunks,
    committed_until_frame,
    sample_rate,
    boundary_padding_seconds,
    max_word_seconds,
    glossary_snapshot,
    recent_context,
    matching_terms,
    asr,
    asr_lock,
    beam_size,
    repetition_penalty,
    max_initial_timestamp,
):
    """Decode one utterance with bounded context and absolute timestamps."""
    if not source_chunks:
        return ASRWindowResult([], np.empty(0, dtype=np.float32), 0, 0, 0.0, 0.0)

    padding = int(boundary_padding_seconds * sample_rate)
    target_start = source_chunks[0].start_frame
    target_end = source_chunks[-1].end_frame
    if committed_until_frame:
        target_start = max(target_start, committed_until_frame)

    available_chunks = list(recent_chunks)
    known_starts = {chunk.start_frame for chunk in available_chunks}
    available_chunks.extend(
        chunk for chunk in source_chunks if chunk.start_frame not in known_starts
    )
    available_chunks.sort(key=lambda chunk: chunk.start_frame)
    audio_start = max(0, target_start - padding)
    audio_end = target_end + padding
    audio_start = max(audio_start, available_chunks[0].start_frame)
    audio_end = min(audio_end, available_chunks[-1].end_frame)
    audio = extract_audio_range(available_chunks, audio_start, audio_end)
    if len(audio) == 0:
        return ASRWindowResult(
            [], np.empty(0, dtype=np.float32), audio_start,
            getattr(glossary_snapshot, "version", 0), 0.0, 0.0,
        )

    audio = highpass(audio, sample_rate)
    primary_error = None
    retry_error = None
    try:
        with asr_lock:
            words = asr.transcribe(
                audio,
                initial_prompt=build_asr_prompt(recent_context) or None,
                hotwords=matching_terms or None,
                beam_size=beam_size,
                repetition_penalty=repetition_penalty,
                max_initial_timestamp=max_initial_timestamp,
            )
    except Exception as error:
        primary_error = error
        try:
            with asr_lock:
                words = asr.transcribe(
                    audio,
                    initial_prompt=build_asr_prompt(recent_context) or None,
                    hotwords=matching_terms or None,
                    beam_size=1,
                    repetition_penalty=repetition_penalty,
                    max_initial_timestamp=max_initial_timestamp,
                )
        except Exception as retry_error_value:
            retry_error = retry_error_value
            words = []

    latency_seconds = asr.last_latency_seconds
    timed_words = [
        TimedWord(
            (audio_start / sample_rate) + word.start,
            (audio_start / sample_rate) + word.end,
            word.text,
            word.probability,
        )
        for word in words
    ]
    timed_words = select_timed_words_for_target(
        timed_words,
        target_start / sample_rate,
        target_end / sample_rate,
        max_word_seconds=max_word_seconds,
    )
    issue = asr_output_issue(
        timed_words,
        max(0.0, (target_end - target_start) / sample_rate),
    )
    return ASRWindowResult(
        timed_words,
        audio,
        audio_start,
        getattr(glossary_snapshot, "version", 0),
        latency_seconds,
        len(audio) / sample_rate,
        issue,
        primary_error,
        retry_error,
    )
