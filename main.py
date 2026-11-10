import os
import datetime
import queue
import re
import threading
import time
from collections import deque

import numpy as np

from asr import ASR, build_asr_prompt
from audio_preprocess import highpass
from audio_capture import (
    AudioChunk,
    capture_loop,
    extract_audio_range,
    pick_loopback_device,
)
from config import Config
from diagnostics import SessionDiagnostics
from i18n import t
from model_validation import missing_model_files
from overlay import Overlay
from text_processing import (
    TimedWord,
    clean_repetitions,
    split_timed_sentences,
    words_to_text,
    correction_is_usable,
    correction_is_confident,
    correction_is_bounded,
    correction_matches_target_range,
    has_time_overlap,
    merge_incomplete_sentences,
    is_recordable_sentence,
    is_review_candidate,
    is_likely_incomplete,
    remove_sentence_overlap_words,
    is_asr_anomaly,
    has_valid_time_span,
    words_in_time_range,
    finalize_pending_group,
    trim_trailing_silence_chunks,
    filter_uncommitted_words,
    select_timed_words_for_correction,
    should_schedule_correction,
    should_hold_after_waiting_silence,
    should_finalize_after_waiting_timeout,
    asr_output_issue,
)
from translator import (
    Translator,
    apply_translation_quality_status,
    clear_review_reason,
    set_review_reason,
)
from terminology import TerminologyManager
from meeting_context import MeetingContext, should_add_to_meeting_context
from correction_context import correction_prompt, correction_snapshot_is_current
from session_archive import AudioRecorder, create_session_directory
from stable_text import StableText
from speech_detection import SileroSpeechDetector, contains_voice
from translation_queue import (
    is_live_translation_current,
    publish_live_translation,
    reset_live_translation,
    take_latest,
)
from worker_runtime import WorkerRuntime
from session_history import append_revision, atomic_write_history
from ui_events import apply_result_event
from asr_pipeline import transcribe_audio_window


def main():
    cfg = Config()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    queue_size = (
        cfg.max_audio_queue_chunks
        if cfg.mode == "complete"
        else cfg.low_latency_queue_chunks
    )
    audio_queue = queue.Queue(maxsize=queue_size)
    history_translation_queue = queue.Queue(maxsize=1)
    live_translation_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=cfg.max_result_queue_items)
    correction_queue = queue.Queue(maxsize=20)
    stop_event = threading.Event()
    capture_stop_event = threading.Event()
    capture_done_event = threading.Event()
    asr_done_event = threading.Event()
    worker_done_event = threading.Event()
    correction_done_event = threading.Event()
    translation_failed_event = threading.Event()
    glossary_snapshot_lock = threading.Lock()
    glossary_snapshot_state = {"snapshot": None}
    records = []
    record_metadata = []
    record_contexts = []
    records_lock = threading.Lock()
    result_queue_lock = threading.Lock()
    correction_pending = set()
    last_history_flush = time.monotonic()
    last_backlog_warning = 0.0
    history_save_lock = threading.Lock()
    revision_log_lock = threading.Lock()
    live_state_lock = threading.Lock()
    live_state = {"revision": 0}

    error_queue = queue.Queue()
    audio_gap_count = 0
    suspect_asr_ranges = []
    capture_stats = {
        "chunks": 0,
        "frames": 0,
        "reconnects": 0,
        "estimated_missing_frames": 0,
        "runtime_discontinuities": 0,
        "next_frame": 0,
    }
    asr_lock = threading.Lock()

    device = pick_loopback_device(cfg.loopback_device_name)
    chunk_frames = int(cfg.samplerate * cfg.chunk_seconds)
    missing = missing_model_files(os.path.join(root_dir, "models"))
    if missing:
        raise FileNotFoundError(t("error.missing_models", files=", ".join(missing)))
    asr = ASR(
        cfg.whisper_model, cfg.whisper_device, cfg.whisper_compute_type,
        os.path.join(root_dir, "models", "whisper_base_en"),
        cpu_threads=cfg.whisper_cpu_threads,
    )
    terminology_path = os.path.join(root_dir, cfg.terminology_file)
    default_terminology_path = os.path.join(root_dir, cfg.default_terminology_file)
    terminology = TerminologyManager.from_files(
        [
            default_terminology_path,
            os.path.join(root_dir, cfg.company_terminology_file),
            terminology_path,
        ]
    )
    glossary_snapshot_state["snapshot"] = terminology.snapshot()
    if not os.path.isfile(terminology_path):
        print(t("status.local_glossary_missing", path=terminology_path))
    meeting_context = MeetingContext(
        terminology,
        max_sentences=cfg.context_max_sentences,
        max_chars=cfg.context_max_chars,
    )
    context_seeded_records = set()
    translator = Translator(
        os.path.join(root_dir, cfg.translate_model_dir),
        cfg.translate_device,
        terminology,
        intra_threads=cfg.translate_intra_threads,
        inter_threads=cfg.translate_inter_threads,
    )
    try:
        speech_detector = SileroSpeechDetector(cfg.samplerate)
        print(t("status.vad_enabled"))
    except Exception as error:
        speech_detector = None
        print(t("status.vad_fallback", error=error))
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_dir = create_session_directory(
        os.path.join(root_dir, "sessions"),
        session_id,
        prefix=cfg.session_directory_prefix,
    )
    print(t("status.session_directory", path=session_dir))
    diagnostics = SessionDiagnostics(os.path.join(session_dir, "diagnostics.jsonl"))
    diagnostics.write_event(
        "session_started",
        mode=cfg.mode,
        sample_rate=cfg.samplerate,
        chunk_seconds=cfg.chunk_seconds,
        whisper_model=cfg.whisper_model,
        whisper_device=cfg.whisper_device,
        translate_device=cfg.translate_device,
        loopback_device=device.name,
    )

    def snapshot_records():
        with records_lock:
            return [tuple(record) for record in records], [
                metadata.copy() for metadata in record_metadata
            ]

    audio_recorder = AudioRecorder(
        os.path.join(session_dir, cfg.audio_filename), cfg.samplerate
    )
    try:
        overlay = Overlay(
            cfg.overlay_alpha,
            cfg.font_size,
            max_history_records=cfg.max_overlay_records,
        )
    except Exception:
        audio_recorder.close()
        raise
    overlay.set_status(
        t("status.mode_complete")
        if cfg.mode == "complete"
        else t("status.mode_low_latency")
    )

    def report_thread_error(name, error):
        try:
            diagnostics.write_event("worker_error", worker=name, error=repr(error))
        except Exception:
            pass
        try:
            error_queue.put_nowait((name, error))
        except Exception:
            pass
        if name == t("worker.audio_capture"):
            capture_done_event.set()
        elif name == t("worker.asr"):
            capture_stop_event.set()
        elif name == t("worker.correction"):
            correction_done_event.set()
        elif name == t("worker.translation"):
            translation_failed_event.set()

    def publish_result(event, first, second="", fourth=None, version=None):
        item = (event, first, second, fourth, version)
        with result_queue_lock:
            try:
                result_queue.put_nowait(item)
            except queue.Full:
                if event in {"english", "status"}:
                    return
                retained = []
                removed_disposable = False
                while True:
                    try:
                        existing = result_queue.get_nowait()
                    except queue.Empty:
                        break
                    if not removed_disposable and existing[0] in {"english", "status"}:
                        removed_disposable = True
                        continue
                    retained.append(existing)
                for existing in retained:
                    result_queue.put_nowait(existing)
                if removed_disposable:
                    result_queue.put_nowait(item)
                else:
                    # The transcript is persisted independently; never block an
                    # ASR/translation worker on a closed or stalled UI.
                    return

    def publish_live_reset():
        with live_state_lock:
            reset_live_translation(live_state, live_translation_queue)
        publish_result("live_reset", "", "", None)

    def enqueue_translation(index):
        while True:
            try:
                history_translation_queue.put(index, timeout=0.2)
                return True
            except queue.Full:
                if (
                    translation_failed_event.is_set()
                    or stop_event.is_set() and worker_done_event.is_set()
                ):
                    return False
                continue

    def capture_worker():
        try:
            capture_loop(
                device, cfg.samplerate, chunk_frames, audio_queue, capture_stop_event, cfg,
                audio_sink=audio_recorder,
                diagnostic_callback=diagnostics.write_event,
                capture_stats=capture_stats,
                audio_timeline_sink=audio_recorder,
                consumer_done_event=asr_done_event,
                failure_callback=lambda error: report_thread_error(
                    t("worker.audio_capture"), error
                ),
            )
        except Exception as error:
            report_thread_error(t("worker.audio_capture"), error)
        finally:
            capture_done_event.set()

    def add_history_sentence(words, audio):
        if not has_valid_time_span(
            words,
            max_word_seconds=cfg.max_asr_word_duration_seconds,
            max_sentence_seconds=cfg.max_asr_sentence_duration_seconds,
        ):
            return None
        with records_lock:
            previous = records[-1][0] if records else ""
            previous_end = record_metadata[-1]["end"] if record_metadata else None
        current_start = words[0].start if words else 0.0
        current_end = words[-1].end if words else current_start
        suspect_asr = any(
            current_start < suspect_end and current_end > suspect_start
            for suspect_start, suspect_end in suspect_asr_ranges
        )
        if previous and has_time_overlap(previous_end, current_start):
            words = remove_sentence_overlap_words(previous, words, min_overlap=2)
        recognized_text = words_to_text(words)
        asr_anomaly = is_asr_anomaly(recognized_text)
        with glossary_snapshot_lock:
            glossary_snapshot = glossary_snapshot_state["snapshot"]
        original = glossary_snapshot.normalize_source(
            recognized_text if asr_anomaly else clean_repetitions(recognized_text)
        )
        if not asr_anomaly and not is_recordable_sentence(
            original
        ):
            return None
        with records_lock:
            if len(records) >= cfg.max_stored_records:
                return None
        with records_lock:
            if records:
                last_orig = re.sub(r"[^\w']+", " ", records[-1][0]).strip().lower()
                curr_orig = re.sub(r"[^\w']+", " ", original).strip().lower()
                if not asr_anomaly and last_orig and last_orig == curr_orig:
                    return None
            records.append([original, ""])
            start = words[0].start if words else 0.0
            end = words[-1].end if words else start
            metadata = {"start": start, "end": end, "version": 1}
            if asr_anomaly:
                metadata["asr_anomaly"] = True
            if asr_anomaly or is_review_candidate(original) or suspect_asr:
                reason = (
                    "asr_implausible_word_rate" if suspect_asr
                    else "asr_anomaly" if asr_anomaly
                    else "asr_short_fragment"
                )
                metadata = set_review_reason(metadata, reason)
            if suspect_asr:
                metadata["asr_suspect"] = True
            record_metadata.append(metadata)
            record_contexts.append(meeting_context.recent_sentences(limit=3))
            index = len(records) - 1
            if should_add_to_meeting_context(metadata):
                meeting_context.add(original)
                context_seeded_records.add(index)
        publish_result("history_add", index, original, metadata, metadata["version"])
        if not enqueue_translation(index):
            publish_result("status", t("status.translation_queue_full"), "", None)
        return index

    def save_history_file():
        save_history_snapshot(os.path.join(session_dir, cfg.history_file))

    def write_timestamped_history():
        timestamped_file = f"meeting_transcript_{session_id}.txt"
        save_history_snapshot(os.path.join(session_dir, timestamped_file))

    def save_history_snapshot(history_path):
        with history_save_lock:
            saved_records, saved_metadata = snapshot_records()
            atomic_write_history(
                history_path, saved_records, saved_metadata, Overlay, t
            )

    def append_revision_log(entry):
        revision_path = os.path.join(session_dir, "meeting_revisions.jsonl")
        with revision_log_lock:
            append_revision(revision_path, entry)

    def finalize_history_sentence(words):
        text = clean_repetitions(words_to_text(words))
        if not is_recordable_sentence(text):
            return None
        return text

    def worker():
        nonlocal audio_gap_count, last_history_flush, last_backlog_warning
        glossary_snapshot = terminology.snapshot()
        utterance_chunks = []
        silence_consecutive_chunks = 0
        last_preview_time = time.monotonic() - cfg.asr_preview_interval_seconds
        last_preview_text = ""
        last_chunk_end_frame = None
        waiting_for_continuation = False
        waiting_since = None
        last_forced_transcription_end = None
        last_history_flush = time.monotonic()
        correction_candidates = []
        recent_chunks = deque()
        stable_preview = StableText()
        stable_preview_text = ""
        committed_until_frame = 0

        def transcribe_utterance(source_chunks, beam_size):
            """Transcribe with nearby audio so quiet boundary words survive VAD."""
            if not source_chunks:
                return [], None, None
            with glossary_snapshot_lock:
                glossary_snapshot = glossary_snapshot_state["snapshot"]
            with records_lock:
                recent_context = meeting_context.asr_prompt_context()
            recent_context = glossary_snapshot.normalize_source(recent_context)
            matching_terms = glossary_snapshot.asr_prompt(
                recent_context,
                max_terms=cfg.asr_glossary_prompt_terms,
                include_always=False,
            )
            result = transcribe_audio_window(
                source_chunks,
                list(recent_chunks),
                committed_until_frame,
                cfg.samplerate,
                cfg.audio_boundary_padding_seconds,
                cfg.max_asr_word_duration_seconds,
                glossary_snapshot,
                recent_context,
                matching_terms,
                asr,
                asr_lock,
                beam_size,
                cfg.asr_repetition_penalty,
                cfg.asr_max_initial_timestamp,
            )
            if result.primary_error:
                diagnostics.write_event(
                    "asr_error",
                    stage="primary",
                    beam_size=beam_size,
                    error=repr(result.primary_error),
                )
                if result.retry_error:
                    diagnostics.write_event(
                        "asr_error", stage="retry", beam_size=1,
                        error=repr(result.retry_error),
                    )
                    publish_result(
                        "status",
                        t("status.asr_retry_failed", error=result.retry_error),
                        "", None,
                    )
                    return [], result.audio, result.audio_start
                publish_result(
                    "status", t("status.asr_retry", error=result.primary_error), "", None
                )
            timed_words = result.words
            diagnostics.write_event(
                "asr_result",
                glossary_version=result.glossary_version,
                words=len(timed_words),
                beam_size=beam_size,
                seconds=round(result.latency_seconds, 3),
                audio_seconds=round(result.audio_seconds, 3),
            )
            if result.latency_seconds > cfg.asr_timeout_seconds and beam_size > 1:
                publish_result(
                    "status",
                    t("status.asr_over_budget", seconds=result.latency_seconds),
                    "",
                    None,
                )
            issue = result.issue
            if issue:
                diagnostics.increment_quality("asr_suspect_windows")
                target_start = source_chunks[0].start_frame
                target_end = source_chunks[-1].end_frame
                suspect_asr_ranges.append(
                    (target_start / cfg.samplerate, target_end / cfg.samplerate)
                )
                diagnostics.write_event(
                    "asr_suspect",
                    issue=issue,
                    words=len(timed_words),
                    target_start=round(target_start / cfg.samplerate, 3),
                    target_end=round(target_end / cfg.samplerate, 3),
                )
                publish_result(
                    "status",
                    t("status.asr_suspect"),
                    "",
                    None,
                )
            return timed_words, result.audio, result.audio_start

        def extract_sentence_audio(words):
            if not words:
                return np.empty(0, dtype=np.float32)
            start = int(words[0].start * cfg.samplerate)
            end = int(words[-1].end * cfg.samplerate)
            chunks = [
                chunk
                for chunk in utterance_chunks + list(recent_chunks)
                if chunk.end_frame > start and chunk.start_frame < end
            ]
            return extract_audio_range(chunks, start, end)

        def schedule_correction(index, words):
            if (
                cfg.mode == "complete"
                and index is not None
                and should_schedule_correction(words)
            ):
                with records_lock:
                    prompt = correction_prompt(
                        records, index, limit=cfg.history_context_window,
                        metadata=record_metadata,
                    )
                    expected_original = records[index][0]
                    expected_version = record_metadata[index]["version"]
                correction_candidates.append(
                    (
                        index,
                        int(words[0].start * cfg.samplerate),
                        int(words[-1].end * cfg.samplerate),
                        prompt,
                        [word.probability for word in words if word.probability is not None],
                        expected_original,
                        expected_version,
                    )
                )

        def enqueue_ready_corrections(force=False):
            if cfg.mode != "complete" or not recent_chunks:
                return
            available_end = recent_chunks[-1].end_frame
            for candidate in list(correction_candidates):
                (
                    index, target_start, target_end, prompt, original_confidences,
                    expected_original, expected_version,
                ) = candidate
                required_end = target_end + int(
                    cfg.correction_context_after_seconds * cfg.samplerate
                )
                if not force and available_end < required_end:
                    continue
                context_start = max(
                    0,
                    target_start - int(cfg.correction_context_before_seconds * cfg.samplerate),
                )
                context_end = min(
                    available_end,
                    required_end if not force else available_end,
                )
                context_chunks = [
                    chunk for chunk in recent_chunks
                    if chunk.end_frame > context_start and chunk.start_frame < context_end
                ]
                if not context_chunks:
                    continue
                actual_start = max(context_start, context_chunks[0].start_frame)
                actual_end = min(context_end, context_chunks[-1].end_frame)
                context_audio = extract_audio_range(context_chunks, actual_start, actual_end)
                if len(context_audio) == 0:
                    continue
                with records_lock:
                    if index in correction_pending:
                        correction_candidates.remove(candidate)
                        continue
                    correction_pending.add(index)
                try:
                    correction_queue.put_nowait(
                        (
                            index,
                            context_audio,
                            actual_start,
                            target_start,
                            target_end,
                            prompt,
                            original_confidences,
                            expected_original,
                            expected_version,
                        )
                    )
                except queue.Full:
                    with records_lock:
                        correction_pending.discard(index)
                    publish_result("status", t("status.correction_queue_full"), "", None)
                    continue
                correction_candidates.remove(candidate)

        try:
            while (
                not capture_done_event.is_set()
                or not audio_queue.empty()
            ):
                try:
                    chunk = audio_queue.get(timeout=0.3)
                except queue.Empty:
                    if capture_done_event.is_set():
                        break
                    if (
                        waiting_for_continuation
                        and waiting_since is not None
                        and time.monotonic() - waiting_since >= cfg.vad_pause_seconds * 3
                    ):
                        tail_audio = [chunk for chunk in utterance_chunks if len(chunk.data)]
                        tail_start = tail_audio[0].start_frame if tail_audio else None
                        tail_end = tail_audio[-1].end_frame if tail_audio else None
                        timed_words, _, _ = transcribe_utterance(utterance_chunks, 1)
                        if tail_start is not None and tail_end is not None:
                            timed_words = words_in_time_range(
                                timed_words,
                                tail_start / cfg.samplerate,
                                tail_end / cfg.samplerate,
                            )
                        completed, tail = split_timed_sentences(timed_words)
                        for sentence_words in completed:
                            if is_recordable_sentence(words_to_text(sentence_words)):
                                record = add_history_sentence(
                                    sentence_words, extract_sentence_audio(sentence_words)
                                )
                                schedule_correction(record, sentence_words)
                        if tail and should_finalize_after_waiting_timeout(tail):
                            record = add_history_sentence(tail, extract_sentence_audio(tail))
                            schedule_correction(record, tail)
                        if timed_words:
                            committed_until_frame = max(
                                committed_until_frame,
                                int(timed_words[-1].end * cfg.samplerate),
                            )
                        committed_until_frame = max(
                            committed_until_frame,
                            tail_end if tail_end is not None else committed_until_frame,
                        )
                        utterance_chunks = []
                        silence_consecutive_chunks = 0
                        waiting_for_continuation = False
                        waiting_since = None
                        last_forced_transcription_end = None
                        publish_result("english", "", "", None)
                        stable_preview.reset()
                        stable_preview_text = ""
                        publish_live_reset()
                        if time.monotonic() - last_history_flush >= cfg.history_flush_seconds:
                            save_history_file()
                            write_timestamped_history()
                            last_history_flush = time.monotonic()
                    continue
                audio_queue.task_done()
                if not isinstance(chunk, AudioChunk):
                    continue

                with glossary_snapshot_lock:
                    terminology.poll_files()
                    glossary_snapshot = terminology.apply_pending()
                    glossary_snapshot_state["snapshot"] = glossary_snapshot
                    if terminology.last_reload_error or terminology.last_update_error:
                        diagnostics.write_event(
                            "glossary_update_rejected",
                            error=repr(terminology.last_reload_error or terminology.last_update_error),
                            glossary_version=glossary_snapshot.version,
                        )
                        terminology.last_reload_error = None
                        terminology.last_update_error = None

                if time.monotonic() - last_history_flush >= cfg.history_flush_seconds:
                    save_history_file()
                    write_timestamped_history()
                    last_history_flush = time.monotonic()

                if chunk.gap_before or (
                    last_chunk_end_frame is not None
                    and chunk.start_frame > last_chunk_end_frame
                ):
                    audio_gap_count += 1
                    diagnostics.write_event(
                        "audio_gap",
                        gap_count=audio_gap_count,
                        frame=chunk.start_frame,
                        seconds=round(chunk.start_frame / cfg.samplerate, 3),
                    )
                    utterance_chunks = []
                    silence_consecutive_chunks = 0
                    waiting_for_continuation = False
                    waiting_since = None
                    last_preview_text = ""
                    stable_preview.reset()
                    stable_preview_text = ""
                    committed_until_frame = chunk.start_frame
                    last_chunk_end_frame = chunk.end_frame
                    publish_live_reset()
                    publish_result("status", t("status.audio_gap", count=audio_gap_count), "", None)
                with records_lock:
                    storage_full = len(records) >= cfg.max_stored_records
                if storage_full:
                    last_chunk_end_frame = chunk.end_frame
                    continue
                last_chunk_end_frame = chunk.end_frame

                recent_chunks.append(chunk)
                recent_start = chunk.end_frame - int(
                    cfg.correction_recent_audio_seconds * cfg.samplerate
                )
                while recent_chunks and recent_chunks[0].end_frame < recent_start:
                    recent_chunks.popleft()
                enqueue_ready_corrections()

                subchunk_frames = max(1, int(cfg.vad_subchunk_seconds * cfg.samplerate))
                is_voice = contains_voice(
                    chunk.data,
                    cfg.voice_energy_threshold * cfg.vad_subchunk_threshold_ratio,
                    subchunk_frames,
                    speech_detector,
                )

                with records_lock:
                    recent_context = meeting_context.asr_prompt_context()
                recent_context = glossary_snapshot.normalize_source(recent_context)
                matched_terms = glossary_snapshot.asr_prompt(
                    recent_context,
                    max_terms=cfg.asr_glossary_prompt_terms,
                    include_always=False,
                )

                backlog_seconds = audio_queue.qsize() * cfg.chunk_seconds
                diagnostics.update_quality_max(
                    "max_audio_backlog_seconds", backlog_seconds
                )
                if (
                    backlog_seconds >= cfg.audio_backlog_warning_seconds
                    and time.monotonic() - last_backlog_warning >= 5.0
                ):
                    last_backlog_warning = time.monotonic()
                    publish_result(
                        "status",
                        t("status.audio_backlog", seconds=backlog_seconds),
                        "",
                        None,
                    )
                    diagnostics.write_event(
                        "audio_backlog",
                        seconds=round(backlog_seconds, 2),
                        mode=cfg.mode,
                        pending_chunks=audio_queue.qsize(),
                    )
                current_beam = 1 if (
                    cfg.mode == "low_latency"
                    or audio_queue.qsize() > 1
                    or backlog_seconds >= cfg.audio_backlog_warning_seconds
                ) else cfg.asr_beam_size

                if is_voice:
                    silence_consecutive_chunks = 0
                    if waiting_for_continuation:
                        waiting_for_continuation = False
                    utterance_chunks.append(chunk)
                    current_duration = sum(len(c.data) for c in utterance_chunks) / cfg.samplerate

                    # Preview less often in complete mode to reduce repeated decoding.
                    now = time.monotonic()
                    preview_interval = (
                        0.8 if cfg.mode == "low_latency"
                        else cfg.asr_preview_interval_seconds
                    )
                    if (
                        now - last_preview_time >= preview_interval
                        and current_duration >= 0.7
                        and audio_queue.qsize() <= cfg.preview_max_pending_audio_chunks
                        and backlog_seconds < cfg.audio_backlog_warning_seconds
                    ):
                        last_preview_time = now
                        p_start = utterance_chunks[0].start_frame
                        p_end = chunk.end_frame
                        preview_audio = extract_audio_range(utterance_chunks, p_start, p_end)
                        if len(preview_audio) > 0:
                            try:
                                with asr_lock:
                                    p_words = asr.transcribe(
                                        highpass(preview_audio, cfg.samplerate),
                                        initial_prompt=build_asr_prompt(recent_context) or None,
                                        hotwords=matched_terms or None,
                                        beam_size=1,
                                        word_timestamps=False,
                                        repetition_penalty=cfg.asr_repetition_penalty,
                                        max_initial_timestamp=cfg.asr_max_initial_timestamp,
                                    )
                            except Exception as error:
                                publish_result("status", t("status.preview_asr_failed", error=error), "", None)
                                p_words = []
                            p_text = clean_repetitions(words_to_text(p_words))
                            preview_issue = asr_output_issue(
                                p_words, len(preview_audio) / cfg.samplerate
                            )
                            if preview_issue:
                                diagnostics.write_event(
                                    "asr_suspect",
                                    stage="preview",
                                    issue=preview_issue,
                                    words=len(p_words),
                                    audio_seconds=round(
                                        len(preview_audio) / cfg.samplerate, 3
                                    ),
                                )
                                p_text = ""
                            if p_words and not has_valid_time_span(
                                p_words,
                                max_word_seconds=cfg.max_asr_word_duration_seconds,
                                max_sentence_seconds=cfg.max_asr_sentence_duration_seconds,
                            ):
                                p_text = ""
                            if p_text and is_asr_anomaly(p_text):
                                publish_result("status", t("status.preview_loop_suppressed"), "", None)
                                p_text = ""
                            if p_text:
                                if p_text != last_preview_text:
                                    last_preview_text = p_text
                                    publish_result("english", p_text, "", None)
                                stable_delta = stable_preview.update(p_text)
                                if stable_delta:
                                    stable_preview_text = (
                                        stable_preview_text + " " + stable_delta
                                    ).strip()
                                    with live_state_lock:
                                        revision = publish_live_translation(
                                            stable_preview_text,
                                            cfg.realtime_translation_min_words,
                                            live_state,
                                            live_translation_queue,
                                        )

                    # Bound continuous speech and split at punctuation or phrase boundaries.
                    force_interval = int(2.0 * cfg.samplerate)
                    force_due = (
                        last_forced_transcription_end is None
                        or chunk.end_frame - last_forced_transcription_end >= force_interval
                    )
                    if current_duration >= cfg.max_utterance_seconds and force_due:
                        last_forced_transcription_end = chunk.end_frame
                        timed_words, full_audio, _ = transcribe_utterance(
                            utterance_chunks, current_beam
                        )
                        if timed_words:
                            completed, remaining = split_timed_sentences(timed_words)
                            groups = merge_incomplete_sentences(
                                completed + ([remaining] if remaining else [])
                            )
                            pending_words = groups[-1] if remaining and groups else []
                            finalized_groups = groups[:-1] if remaining else groups
                            if finalized_groups:
                                for s_words in finalized_groups:
                                    s_text = finalize_history_sentence(s_words)
                                    if s_text:
                                        s_audio = extract_sentence_audio(s_words)
                                        cand = add_history_sentence(s_words, s_audio)
                                        schedule_correction(cand, s_words)
                                committed_until_frame = max(
                                    committed_until_frame,
                                    int(finalized_groups[-1][-1].end * cfg.samplerate),
                                )
                                if pending_words:
                                    rem_start = int(pending_words[0].start * cfg.samplerate)
                                    utterance_chunks = [c for c in utterance_chunks if c.end_frame > rem_start]
                                    publish_result("english", clean_repetitions(words_to_text(pending_words)), "", None)
                                    stable_preview.reset()
                                    stable_preview_text = ""
                                    publish_live_reset()
                                else:
                                    utterance_chunks = []
                                    last_forced_transcription_end = None
                                    publish_result("english", "", "", None)
                                    stable_preview.reset()
                                    stable_preview_text = ""
                                    publish_live_reset()
                            else:
                                if current_duration >= cfg.max_utterance_seconds * 1.5:
                                    cand = add_history_sentence(
                                        timed_words, extract_sentence_audio(timed_words)
                                    )
                                    schedule_correction(cand, timed_words)
                                    utterance_chunks = []
                                    last_forced_transcription_end = None
                                    publish_result("english", "", "", None)
                                    stable_preview.reset()
                                    stable_preview_text = ""
                                    publish_live_reset()
                                else:
                                    publish_result("status", t("status.waiting_context"), "", None)
                        last_preview_text = ""

                else:
                    with glossary_snapshot_lock:
                        terminology.poll_files()
                        updated_snapshot = terminology.apply_pending()
                        glossary_snapshot_state["snapshot"] = updated_snapshot
                        if terminology.last_reload_error or terminology.last_update_error:
                            diagnostics.write_event(
                                "glossary_update_rejected",
                                error=repr(terminology.last_reload_error or terminology.last_update_error),
                                glossary_version=updated_snapshot.version,
                            )
                            terminology.last_reload_error = None
                            terminology.last_update_error = None
                    glossary_snapshot = updated_snapshot
                    # Use silence intervals to detect natural utterance boundaries.
                    if utterance_chunks:
                        # Retain the silence following speech as a tail pad. Quiet
                        # final words can otherwise be classified as silence.
                        if not waiting_for_continuation and utterance_chunks[-1] is not chunk:
                            utterance_chunks.append(chunk)
                        silence_consecutive_chunks += 1
                        silence_duration = silence_consecutive_chunks * (len(chunk.data) / cfg.samplerate)
                        current_duration = sum(len(c.data) for c in utterance_chunks) / cfg.samplerate

                        # Finalize after sufficient silence when the utterance is long enough.
                        if silence_duration >= cfg.vad_pause_seconds and current_duration >= 0.35:
                            timed_words, full_audio, _ = transcribe_utterance(
                                utterance_chunks, current_beam
                            )
                            if timed_words:
                                completed, remaining = split_timed_sentences(timed_words)
                                groups = merge_incomplete_sentences(
                                    completed + ([remaining] if remaining else [])
                                )
                                finalized_any = False
                                pending_words = groups[-1] if remaining and groups else []
                                if pending_words and finalize_pending_group(
                                    pending_words, waiting_for_continuation
                                ):
                                    finalized_groups = groups
                                    pending_words = []
                                else:
                                    finalized_groups = groups[:-1] if pending_words else groups
                                for sentence_words in finalized_groups:
                                    finalized = finalize_history_sentence(sentence_words)
                                    if finalized:
                                        cand = add_history_sentence(
                                            sentence_words, extract_sentence_audio(sentence_words)
                                        )
                                        schedule_correction(cand, sentence_words)
                                        finalized_any = True
                                if finalized_any and finalized_groups:
                                    committed_until_frame = max(
                                        committed_until_frame,
                                        int(finalized_groups[-1][-1].end * cfg.samplerate),
                                    )
                                    if not pending_words:
                                        waiting_since = None
                                if waiting_for_continuation and not pending_words:
                                    utterance_chunks = []
                                    silence_consecutive_chunks = 0
                                    waiting_for_continuation = False
                                    waiting_since = None
                                    last_preview_text = ""
                                    publish_result("english", "", "", None)
                                    stable_preview.reset()
                                    stable_preview_text = ""
                                    publish_live_reset()
                                    continue
                                if waiting_for_continuation and pending_words:
                                    if should_hold_after_waiting_silence(
                                        pending_words,
                                        silence_duration,
                                        cfg.vad_pause_seconds * 2,
                                    ):
                                        silence_consecutive_chunks = 0
                                        publish_result("status", t("status.waiting_speech"), "", None)
                                        continue
                                    utterance_chunks = trim_trailing_silence_chunks(
                                        utterance_chunks, silence_consecutive_chunks
                                    )
                                    silence_consecutive_chunks = 0
                                    publish_result("status", t("status.waiting_continuation"), "", None)
                                    continue
                                if pending_words and finalized_groups:
                                    remaining_start = int(pending_words[0].start * cfg.samplerate)
                                    utterance_chunks = [
                                        item for item in utterance_chunks
                                        if item.end_frame > remaining_start
                                    ]
                                    silence_consecutive_chunks = 0
                                    waiting_for_continuation = True
                                    waiting_since = time.monotonic()
                                    last_forced_transcription_end = None
                                elif finalized_any:
                                    utterance_chunks = []
                                    silence_consecutive_chunks = 0
                                    last_preview_text = ""
                                    waiting_for_continuation = False
                                    waiting_since = None
                                    publish_result("english", "", "", None)
                                    stable_preview.reset()
                                    stable_preview_text = ""
                                    publish_live_reset()
                                else:
                                    utterance_chunks = trim_trailing_silence_chunks(
                                        utterance_chunks, silence_consecutive_chunks
                                    )
                                    silence_consecutive_chunks = 0
                                    waiting_for_continuation = True
                                    waiting_since = time.monotonic()
                                    publish_result("status", t("status.waiting_context"), "", None)
                            else:
                                utterance_chunks = trim_trailing_silence_chunks(
                                    utterance_chunks, silence_consecutive_chunks
                                )
                                silence_consecutive_chunks = 0
                                waiting_for_continuation = True
                                waiting_since = time.monotonic()
                                publish_result("status", t("status.no_clear_speech"), "", None)

        except Exception as error:
            try:
                report_thread_error(t("worker.asr"), error)
            except Exception:
                pass
            capture_stop_event.set()
        finally:
            try:
                if utterance_chunks:
                    timed_words, full_audio, _ = transcribe_utterance(utterance_chunks, 1)
                    if full_audio is not None and len(full_audio) >= int(0.3 * cfg.samplerate):
                        timed_words = filter_uncommitted_words(
                            timed_words, committed_until_frame / cfg.samplerate
                        )
                        completed, tail = split_timed_sentences(timed_words)
                        for sentence_words in completed:
                            if is_recordable_sentence(words_to_text(sentence_words)):
                                record = add_history_sentence(
                                    sentence_words, extract_sentence_audio(sentence_words)
                                )
                                schedule_correction(record, sentence_words)
                        if tail and should_finalize_after_waiting_timeout(tail):
                            record = add_history_sentence(tail, extract_sentence_audio(tail))
                            schedule_correction(record, tail)
                        if timed_words:
                            committed_until_frame = max(
                                committed_until_frame,
                                int(timed_words[-1].end * cfg.samplerate),
                            )
                enqueue_ready_corrections(force=True)
            except Exception as error:
                try:
                    report_thread_error(t("worker.asr"), error)
                except Exception:
                    pass
            finally:
                asr_done_event.set()
                worker_done_event.set()

    def correction_worker():
        try:
            while not asr_done_event.is_set() or not correction_queue.empty():
                try:
                    (
                        index,
                        context_audio,
                        actual_start,
                        target_start,
                        target_end,
                        prompt,
                        original_confidences,
                        expected_original,
                        expected_version,
                ) = correction_queue.get(timeout=0.2)
                except queue.Empty:
                    if capture_done_event.is_set():
                        break
                    continue
                correction_queue.task_done()
                try:
                    with records_lock:
                        if not correction_snapshot_is_current(
                            records[index] if index < len(records) else None,
                            record_metadata[index] if index < len(record_metadata) else None,
                            expected_original,
                            expected_version,
                        ):
                            continue
                    # Yield to live audio when the capture queue is backed up.
                    wait_started = time.monotonic()
                    while (
                        not stop_event.is_set()
                        and audio_queue.qsize() > 0
                        and time.monotonic() - wait_started < cfg.correction_max_audio_wait_seconds
                    ):
                        time.sleep(0.05)

                    with glossary_snapshot_lock:
                        glossary_snapshot = glossary_snapshot_state["snapshot"]
                    asr_prompt = glossary_snapshot.asr_prompt(
                        prompt,
                        max_terms=cfg.asr_glossary_prompt_terms,
                        include_always=False,
                    )
                    correction_started = time.monotonic()
                    with asr_lock:
                        corrected_words = asr.transcribe(
                            highpass(context_audio, cfg.samplerate),
                            initial_prompt=build_asr_prompt(prompt) or None,
                            hotwords=asr_prompt or None,
                            condition_on_previous_text=cfg.asr_condition_on_previous_text,
                            beam_size=cfg.asr_correction_beam_size,
                            repetition_penalty=cfg.asr_repetition_penalty,
                            max_initial_timestamp=cfg.asr_max_initial_timestamp,
                        )
                    diagnostics.write_event(
                        "asr_correction",
                        glossary_version=getattr(glossary_snapshot, "version", 0),
                        index=index,
                        seconds=round(time.monotonic() - correction_started, 3),
                        words=len(corrected_words),
                    )
                    corrected_words = [
                        TimedWord(
                            word.start + actual_start / cfg.samplerate,
                            word.end + actual_start / cfg.samplerate,
                            word.text,
                            word.probability,
                        )
                        for word in corrected_words
                    ]
                    if target_start is not None and target_end is not None and target_end > target_start:
                        range_words = select_timed_words_for_correction(
                            corrected_words,
                            target_start / cfg.samplerate,
                            target_end / cfg.samplerate,
                            max_word_seconds=cfg.max_asr_word_duration_seconds,
                        )
                        corrected = words_to_text(range_words)
                    else:
                        corrected = words_to_text(corrected_words)

                    corrected = glossary_snapshot.normalize_source(
                        clean_repetitions(corrected)
                    )
                    corrected_target_words = (
                        range_words
                        if target_start is not None and target_end is not None and target_end > target_start
                        else corrected_words
                    )
                    corrected_confidences = [
                        word.probability for word in corrected_target_words
                        if word.probability is not None
                    ]
                    original = expected_original
                    correction_rejection_reason = None
                    if (
                        target_start is not None
                        and target_end is not None
                        and not correction_matches_target_range(
                            corrected_target_words,
                            target_start / cfg.samplerate,
                            target_end / cfg.samplerate,
                        )
                    ):
                        correction_rejection_reason = "correction_outside_target_range"
                    elif not correction_is_bounded(original, corrected):
                        correction_rejection_reason = "correction_output_expanded"
                    elif not is_recordable_sentence(corrected):
                        correction_rejection_reason = "correction_not_recordable"
                    elif not correction_is_usable(original, corrected):
                        correction_rejection_reason = "correction_not_usable"
                    elif not correction_is_confident(
                        original,
                        corrected,
                        original_confidences,
                        corrected_confidences,
                    ):
                        correction_rejection_reason = "correction_confidence_not_improved"
                    elif is_asr_anomaly(corrected):
                        correction_rejection_reason = "correction_asr_anomaly"
                    if (
                        correction_rejection_reason is None
                    ):
                        changed = False
                        with records_lock:
                            if (
                                index < len(records)
                                and correction_snapshot_is_current(
                                    records[index], record_metadata[index],
                                    expected_original, expected_version,
                                )
                                and record_metadata[index]["version"] == expected_version
                                and corrected != records[index][0]
                            ):
                                record_metadata[index]["previous_original"] = original
                                record_metadata[index]["previous_translation"] = records[index][1]
                                record_metadata[index]["corrected"] = True
                                records[index][0] = corrected
                                record_metadata[index]["version"] += 1
                                record_metadata[index].pop("asr_anomaly", None)
                                updated_metadata = dict(record_metadata[index])
                                updated_metadata = clear_review_reason(
                                    updated_metadata, "legacy_review"
                                )
                                for review_reason in (
                                    "asr_implausible_word_rate",
                                    "asr_anomaly",
                                    "asr_short_fragment",
                                ):
                                    updated_metadata = clear_review_reason(
                                        updated_metadata, review_reason
                                    )
                                updated_metadata.pop("asr_suspect", None)
                                record_metadata[index].clear()
                                record_metadata[index].update(updated_metadata)
                                metadata = updated_metadata
                                changed = True
                        if changed:
                            with records_lock:
                                for context_index, sentences in enumerate(record_contexts):
                                    record_contexts[context_index] = [
                                        corrected if sentence == original else sentence
                                        for sentence in sentences
                                    ]
                                if index in context_seeded_records:
                                    meeting_context.replace_recent(original, corrected)
                                else:
                                    meeting_context.add(corrected)
                                    context_seeded_records.add(index)
                            publish_result(
                                "history_original_update",
                                index,
                                corrected,
                                metadata,
                                version=metadata["version"],
                            )
                            if not enqueue_translation(index):
                                publish_result("status", t("status.corrected_translation_queue_full"), "", None)
                    elif correction_rejection_reason:
                        diagnostics.write_event(
                            "asr_correction_rejected",
                            index=index,
                            reason=correction_rejection_reason,
                            original_words=len(re.findall(r"[A-Za-z0-9']+", original)),
                            candidate_words=len(re.findall(r"[A-Za-z0-9']+", corrected)),
                            original=original,
                            candidate=corrected,
                            target_start=(
                                round(target_start / cfg.samplerate, 3)
                                if target_start is not None else None
                            ),
                            target_end=(
                                round(target_end / cfg.samplerate, 3)
                                if target_end is not None else None
                            ),
                        )
                except Exception as error:
                    diagnostics.write_event(
                        "asr_error", stage="correction", index=index, error=repr(error)
                    )
                    publish_result(
                        "status",
                        t("status.correction_failed", index=index + 1, error=error),
                        "",
                        None,
                    )
                finally:
                    with records_lock:
                        correction_pending.discard(index)
        except Exception as error:
            try:
                report_thread_error(t("worker.correction"), error)
            except Exception:
                pass
        finally:
            correction_done_event.set()

    def translation_worker():
        try:
            while (
                not translation_failed_event.is_set()
                and (
                    not asr_done_event.is_set()
                    or not correction_done_event.is_set()
                    or not history_translation_queue.empty()
                    or not live_translation_queue.empty()
                )
            ):
                try:
                    live_job = live_translation_queue.get_nowait()
                except queue.Empty:
                    live_job = None
                if live_job is not None:
                    live_job = take_latest(live_job, live_translation_queue)
                    revision, live_text = live_job
                    live_translation_queue.task_done()
                    with live_state_lock:
                        if not is_live_translation_current(
                            live_state, revision, live_text
                        ):
                            continue
                    try:
                        live_translation_started = time.monotonic()
                        translated_live = translator.translate(
                            live_text, quality=cfg.live_translation_quality
                        )
                        diagnostics.write_event(
                            "live_translation",
                            revision=revision,
                            seconds=round(time.monotonic() - live_translation_started, 3),
                            words=len(live_text.split()),
                        )
                        with live_state_lock:
                            is_current = is_live_translation_current(
                                live_state, revision, live_text
                            )
                        if is_current:
                            publish_result(
                                "live_translation", revision, translated_live, live_text
                            )
                    except Exception as error:
                        diagnostics.write_event(
                            "translation_error",
                            stage="live",
                            revision=revision,
                            error=repr(error),
                        )
                        publish_result("status", t("status.live_translation_failed", error=error), "", None)
                    continue
                try:
                    index = history_translation_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                history_translation_queue.task_done()

                with records_lock:
                    if index >= len(records):
                        continue
                    original = records[index][0]
                    metadata = record_metadata[index].copy()
                    context_sentences = list(record_contexts[index])
                if not original.strip():
                    continue

                try:
                    history_translation_started = time.monotonic()
                    same_record_window = False
                    with records_lock:
                        if (
                            index < len(records)
                            and record_metadata[index]["version"] == metadata["version"]
                        ):
                            same_record_window = (
                                record_metadata[index]["start"] == metadata["start"]
                                and record_metadata[index]["end"] == metadata["end"]
                            )
                    if (
                        cfg.translation_context_enabled
                        and context_sentences
                        and len(original.split()) >= 4
                        and same_record_window
                    ):
                        try:
                            translated_parts = translator.translate_context(
                                context_sentences[-2:] + [original]
                            )
                            translated = translated_parts[-1]
                        except Exception as error:
                            diagnostics.increment_quality(
                                "translation_context_fallbacks"
                            )
                            diagnostics.write_event(
                                "translation_context_fallback",
                                index=index,
                                error=repr(error),
                            )
                            translated = translator.translate(
                                original, quality=cfg.history_translation_quality
                            )
                    else:
                        translated = translator.translate(
                            original, quality=cfg.history_translation_quality
                        )

                    issues = translator.quality_issues(original, translated)
                    if issues:
                        diagnostics.increment_quality("translation_quality_retries")
                        publish_result(
                            "status",
                            t("status.translation_warning", index=index + 1, issues=", ".join(issues)),
                            "",
                            None,
                        )
                        if cfg.translation_retry_count:
                            try:
                                translated = translator.translate(original, quality=True)
                            except Exception as error:
                                issues.append("translation_retry_failed")
                                diagnostics.increment_quality("translation_retry_failures")
                                publish_result(
                                    "status",
                                    t("status.translation_retry_failed", index=index + 1, error=error),
                                    "",
                                    None,
                                )

                    remaining_issues = translator.quality_issues(original, translated)
                    if "translation_retry_failed" in issues:
                        remaining_issues.append("translation_retry_failed")
                    if "translation_error" in issues:
                        remaining_issues.append("translation_error")
                    if remaining_issues:
                        diagnostics.increment_quality("translation_quality_unresolved")
                    with records_lock:
                        current_for_quality = (
                            index < len(records)
                            and records[index][0] == original
                            and record_metadata[index]["version"] == metadata["version"]
                        )
                        if current_for_quality:
                            updated_metadata = apply_translation_quality_status(
                                record_metadata[index], remaining_issues
                            )
                            record_metadata[index].clear()
                            record_metadata[index].update(updated_metadata)
                            metadata = updated_metadata
                    diagnostics.write_event(
                        "history_translation",
                        index=index,
                        seconds=round(time.monotonic() - history_translation_started, 3),
                        issues=remaining_issues,
                    )
                    if remaining_issues or is_review_candidate(original):
                        with records_lock:
                            if index < len(record_metadata) and is_review_candidate(original):
                                updated_metadata = set_review_reason(
                                    record_metadata[index], "asr_short_fragment"
                                )
                                record_metadata[index].clear()
                                record_metadata[index].update(updated_metadata)
                                metadata = updated_metadata
                        publish_result("history_review", index, "", None)
                    if remaining_issues:
                        publish_result(
                            "status",
                            t("status.translation_unresolved", index=index + 1, issues=", ".join(remaining_issues)),
                            "",
                            None,
                        )

                    with records_lock:
                        current = (
                            index < len(records)
                            and records[index][0] == original
                            and record_metadata[index]["version"] == metadata["version"]
                        )
                        if current:
                            records[index][1] = translated
                            revision_metadata = record_metadata[index].copy()
                        else:
                            revision_metadata = None
                    if current:
                        if revision_metadata and revision_metadata.get("corrected"):
                            append_revision_log({
                                "index": index,
                                "version": metadata["version"],
                                "start": metadata.get("start"),
                                "end": metadata.get("end"),
                                "previous_en": revision_metadata.get("previous_original", ""),
                                "previous_zh": revision_metadata.get("previous_translation", ""),
                                "current_en": original,
                                "current_zh": translated,
                            })
                        if translator.last_latency_seconds > cfg.translation_timeout_seconds:
                            publish_result(
                                "status",
                                t("status.translation_over_budget", index=index + 1, seconds=translator.last_latency_seconds),
                                "",
                                None,
                            )
                        publish_result(
                            "history_update",
                            index,
                            translated,
                            metadata["version"],
                            version=metadata["version"],
                        )
                except Exception as error:
                    with records_lock:
                        if index < len(records):
                            updated_metadata = set_review_reason(
                                record_metadata[index], "translation_error"
                            )
                            record_metadata[index].clear()
                            record_metadata[index].update(updated_metadata)
                    publish_result("history_review", index, "", None)
                    diagnostics.write_event(
                        "translation_error",
                        stage="history",
                        index=index,
                        error=repr(error),
                    )
                    publish_result(
                        "status",
                        t("status.translation_failed", index=index + 1, error=error),
                        "",
                        None,
                    )
        except Exception as error:
            try:
                report_thread_error(t("worker.translation"), error)
            except Exception:
                pass

    worker_runtime = WorkerRuntime(report_thread_error)
    capture_thread = worker_runtime.start("audio-capture", capture_worker)
    worker_thread = worker_runtime.start("asr-worker", worker)
    correction_thread = worker_runtime.start("correction-worker", correction_worker)
    translation_thread = worker_runtime.start("translation-worker", translation_worker)
    capture_started_at = time.monotonic()
    audio_started_at = capture_started_at

    def refresh():
        try:
            processed = 0
            while processed < 100:
                try:
                    event, first, second, fourth, event_version = result_queue.get_nowait()
                except queue.Empty:
                    break
                result_queue.task_done()
                processed += 1
                apply_result_event(
                    overlay, cfg, (event, first, second, fourth, event_version),
                    live_state, live_state_lock,
                )

            while True:
                try:
                    name, error = error_queue.get_nowait()
                except queue.Empty:
                    break
                overlay.set_status(t("status.worker_failed", worker=name, error=error))
        except Exception as error:
            report_thread_error(t("worker.ui_refresh"), error)
        finally:
            if not overlay.closed:
                try:
                    overlay.refresh_after_id = overlay.root.after(100, refresh)
                except Exception as error:
                    report_thread_error(t("worker.ui_refresh"), error)

    overlay.refresh_after_id = overlay.root.after(100, refresh)
    try:
        overlay.run()
    finally:
        stop_event.set()
        capture_stop_event.set()
        capture_thread.join(timeout=cfg.shutdown_join_timeout_seconds)
        capture_ended_at = time.monotonic()
        if getattr(overlay, "refresh_after_id", None):
            try:
                overlay.root.after_cancel(overlay.refresh_after_id)
            except Exception:
                pass
        unfinished_workers = worker_runtime.join_all(
            timeout=cfg.shutdown_join_timeout_seconds
        )
        if unfinished_workers:
            report_thread_error(
                "shutdown",
                RuntimeError("Workers did not stop in time: " + ", ".join(unfinished_workers)),
            )
        audio_recorder.close()
        saved_records, _ = snapshot_records()
        write_timestamped_history()
        save_history_file()
        print(t("status.session_saved", count=len(saved_records), path=session_dir))
        diagnostics.write_event(
            "session_finished",
            records=len(saved_records),
            audio_gaps=audio_gap_count,
            audio_backlog_seconds=round(audio_queue.qsize() * cfg.chunk_seconds, 3),
            audio_capture_chunks=capture_stats["chunks"],
            audio_capture_frames=capture_stats["frames"],
            audio_capture_seconds=round(capture_stats["frames"] / cfg.samplerate, 3),
            audio_capture_elapsed_seconds=round(
                capture_ended_at - capture_started_at, 3
            ),
            audio_session_elapsed_seconds=round(
                capture_ended_at - audio_started_at, 3
            ),
            audio_device_reconnects=capture_stats["reconnects"],
            audio_capture_discontinuity_warnings=capture_stats[
                "runtime_discontinuities"
            ],
            audio_estimated_missing_frames=capture_stats["estimated_missing_frames"],
            audio_estimated_missing_seconds=round(
                capture_stats["estimated_missing_frames"] / cfg.samplerate, 3
            ),
            audio_frames_written=audio_recorder.frames_written,
            audio_timeline_frames=capture_stats["next_frame"],
            audio_timeline_seconds=round(
                capture_stats["next_frame"] / cfg.samplerate, 3
            ),
            audio_silence_padded_frames=max(
                0, audio_recorder.frames_written - capture_stats["frames"]
            ),
            audio_seconds_written=round(
                audio_recorder.frames_written / cfg.samplerate, 3
            ),
            quality_stats=diagnostics.quality_snapshot(),
            transcript_path=os.path.join(session_dir, cfg.history_file),
            audio_path=os.path.join(session_dir, cfg.audio_filename),
        )


if __name__ == "__main__":
    main()
