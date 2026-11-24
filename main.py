import os
import datetime
import json
import queue
import re
import threading
import time
import warnings
from collections import deque

import numpy as np

from asr import ASR, build_asr_prompt
from audio_preprocess import highpass
from audio_capture import (
    AudioWarningRecorder,
    AudioChunk,
    audio_chunks_duration,
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
    segment_should_advance_committed_until,
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
    PendingHistoryTranslations,
    is_live_translation_current,
    publish_live_translation,
    reset_live_translation,
)
from worker_runtime import WorkerRuntime
from session_history import append_revision, atomic_write_history
from speech_quality import (
    analyze_audio_window,
    analyze_timed_utterance,
    classify_segmentation_boundaries,
)
from ui_events import apply_result_event, drain_worker_errors, enqueue_result_event
from asr_pipeline import transcribe_audio_window
from review_extension import load_review_extension
from pipeline_metrics import PipelineMetrics, record_capture_time_for_event


def main():
    cfg = Config()
    review_backend, selected_ai_cli = load_review_extension(cfg)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    queue_size = (
        cfg.max_audio_queue_chunks
        if cfg.mode == "complete"
        else cfg.low_latency_queue_chunks
    )
    audio_queue = queue.Queue(maxsize=queue_size)
    history_translations = PendingHistoryTranslations()
    translation_ready = threading.Event()
    live_translation_queue = queue.Queue(maxsize=1)
    # The configured limit is a soft limit: transient UI updates may be shed,
    # but committed transcript/history events must never be dropped.
    result_queue = queue.Queue()
    correction_queue = queue.Queue(maxsize=20)
    stop_event = threading.Event()
    capture_stop_event = threading.Event()
    capture_done_event = threading.Event()
    asr_done_event = threading.Event()
    correction_done_event = threading.Event()
    translation_failed_event = threading.Event()
    copilot_done_event = threading.Event()
    glossary_snapshot_lock = threading.Lock()
    glossary_snapshot_state = {"snapshot": None}
    records = []
    record_metadata = []
    record_contexts = []
    copilot_state = {
        "round": 0,
        "next_review_at": time.monotonic() + cfg.realtime_copilot_start_delay_seconds,
        "dirty_record_ids": set(),

        "context_summary": "",
        "term_candidates": [],
        "retry_counts": {},
    } if review_backend else None
    records_lock = threading.Lock()
    result_queue_lock = threading.Lock()
    ui_shed_state = {"last_report": time.monotonic(), "counts": {}}
    ui_backlog_state = {"last_report": 0.0}
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
        "dropped_chunks": 0,
        "dropped_frames": 0,
        "stale_chunks": 0,
        "stale_frames": 0,
        "runtime_underflows": 0,
        "runtime_overflows": 0,
        "timeline_gap_events": 0,
        "timeline_gap_frames": 0,
    }
    pipeline_metrics = PipelineMetrics()
    capture_timestamps = deque(maxlen=4096)
    capture_timestamps_lock = threading.Lock()
    preview_sequence = 0
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
        realtime_copilot_enabled=review_backend is not None,
        realtime_ai_cli=selected_ai_cli["provider"] if selected_ai_cli else None,
        realtime_copilot_model=cfg.realtime_copilot_model or "CLI configured default",
        realtime_copilot_deadline_seconds=cfg.realtime_copilot_deadline_seconds,
        realtime_copilot_interval_seconds=cfg.realtime_copilot_interval_seconds,
        realtime_copilot_cumulative_rounds=cfg.realtime_copilot_cumulative_rounds,

        loopback_device=device.name,
    )

    def record_runtime_warning(
        category, message, filename, lineno, observed_range=None, count=1
    ):
        lower_message = message.lower()
        if "discontinuity" in lower_message:
            capture_stats["runtime_discontinuities"] += count
            diagnostics.increment_quality("capture_runtime_discontinuities", count)
            diagnostics.write_event(
                "audio_capture_discontinuity_warning",
                count=count,
                category=category,
                message=message,
                source=os.path.basename(filename),
                line=lineno,
                **(observed_range or {}),
            )
        elif "underflow" in lower_message or "overflow" in lower_message:
            capture_stats[
                "runtime_underflows" if "underflow" in lower_message else "runtime_overflows"
            ] += count
            diagnostics.increment_quality(
                "capture_runtime_underflows" if "underflow" in lower_message
                else "capture_runtime_overflows",
                count,
            )
            diagnostics.write_event(
                "audio_runtime_warning",
                category=category,
                message=message,
                source=os.path.basename(filename),
                line=lineno,
                count=count,
                **(observed_range or {}),
            )
        diagnostics.record_warning(category, message, filename, lineno, count)

    warning_recorder = AudioWarningRecorder(record_runtime_warning)
    previous_warning_hook = warnings.showwarning

    def capture_warning_hook(message, category, filename, lineno, file=None, line=None):
        warning_recorder(message, category, filename, lineno, file=file, line=line)
        previous_warning_hook(message, category, filename, lineno, file=file, line=line)

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
        warning_recorder.flush()
        warnings.showwarning = previous_warning_hook
        raise
    overlay.set_status(
        t("status.mode_complete")
        if cfg.mode == "complete"
        else t("status.mode_low_latency")
    )
    warnings.showwarning = capture_warning_hook

    def report_thread_error(name, error):
        try:
            diagnostics.write_event("worker_error", worker=name, error=repr(error))
        except Exception:
            pass
        try:
            error_queue.put_nowait((name, error))
        except Exception:
            pass
        if name in ("audio-capture", t("worker.audio_capture")):
            capture_done_event.set()
        elif name in ("asr-worker", t("worker.asr")):
            capture_stop_event.set()
        elif name in ("correction-worker", t("worker.correction")):
            correction_done_event.set()
        elif name in ("translation-worker", t("worker.translation")):
            translation_failed_event.set()

    def publish_result(event, first, second="", fourth=None, version=None):
        queued_at = time.monotonic()
        if event == "history_add":
            pipeline_metrics.remember_record(
                first, version, (fourth or {}).get("captured_at_monotonic"),
                published_at=queued_at,
            )
        elif event == "history_original_update":
            pipeline_metrics.remember_record(
                first, version,
                record_capture_time_for_event(event, fourth, captured_at_for_audio_time),
                published_at=queued_at,
            )
        elif event == "history_update":
            pipeline_metrics.translation_published(
                "committed_chinese", (first, version), queued_at
            )
        elif event == "english" and isinstance(fourth, dict):
            pipeline_metrics.remember_preview(
                fourth["pipeline_id"], fourth["captured_at_monotonic"]
            )
        elif event == "live_translation":
            pipeline_metrics.translation_published("live_chinese", first, queued_at)
        elif event == "history_review" and version is not None:
            # Review-status events do not establish a new record version.
            pass
        item = (event, first, second, fourth, version)
        with result_queue_lock:
            queue_depth = result_queue.qsize()
            accepted = enqueue_result_event(
                result_queue, item, cfg.max_result_queue_items
            )
            if not accepted:
                if event == "english" and isinstance(fourth, dict):
                    pipeline_metrics.discard_preview(fourth["pipeline_id"])
                counts = ui_shed_state["counts"]
                counts[event] = counts.get(event, 0) + 1
                now = time.monotonic()
                if now - ui_shed_state["last_report"] >= 5.0:
                    diagnostics.write_event(
                        "ui_events_shed_summary",
                        event_counts=dict(counts),
                        queue_depth=queue_depth,
                        soft_limit=cfg.max_result_queue_items,
                    )
                    ui_shed_state["counts"] = {}
                    ui_shed_state["last_report"] = now
            elif (
                cfg.max_result_queue_items > 0
                and queue_depth < cfg.max_result_queue_items
                and result_queue.qsize() >= cfg.max_result_queue_items
            ):
                diagnostics.write_event(
                    "ui_queue_soft_limit_reached",
                    queue_depth=result_queue.qsize(),
                    soft_limit=cfg.max_result_queue_items,
                )
            diagnostics.update_quality_max("max_result_queue_items", result_queue.qsize())

    def publish_live_reset():
        with live_state_lock:
            reset_live_translation(live_state, live_translation_queue)
        pipeline_metrics.clear_live()
        publish_result("live_reset", "", "", None)

    def enqueue_translation(index):
        if translation_failed_event.is_set():
            return False
        history_translations.add(index)
        translation_ready.set()
        return True

    def capture_worker():
        try:
            capture_loop(
                device, cfg.samplerate, chunk_frames, audio_queue, capture_stop_event, cfg,
                audio_sink=audio_recorder,
                diagnostic_callback=diagnostics.write_event,
                capture_stats=capture_stats,
                gap_callback=record_audio_queue_drop,
                drop_callback=record_dropped_audio_chunk,
                captured_chunk_callback=remember_captured_chunk,
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

    def record_audio_queue_drop(frame):
        diagnostics.write_event(
            "audio_queue_drop", frame=frame,
            dropped_chunks=capture_stats["dropped_chunks"],
        )
        try:
            print(t("log.audio_chunk_dropped", frame=frame))
        except Exception:
            pass

    def record_dropped_audio_chunk(chunk):
        diagnostics.increment_quality("audio_queue_dropped_chunks")
        diagnostics.increment_quality("audio_queue_dropped_frames", len(chunk.data))
        diagnostics.write_event(
            "audio_queue_chunk_dropped",
            start_frame=chunk.start_frame,
            end_frame=chunk.end_frame,
            audio_seconds=round(len(chunk.data) / cfg.samplerate, 3),
        )

    def remember_captured_chunk(chunk):
        with capture_timestamps_lock:
            capture_timestamps.append((chunk.start_frame, chunk.end_frame, chunk.captured_at))

    def captured_at_for_audio_time(seconds):
        target_frame = int(max(0.0, seconds) * cfg.samplerate)
        with capture_timestamps_lock:
            for start_frame, end_frame, captured_at in reversed(capture_timestamps):
                if start_frame <= target_frame <= end_frame:
                    return captured_at - (end_frame - target_frame) / cfg.samplerate
        return None

    def add_history_sentence(words, audio, window_id=None, segment_id=None):
        if not has_valid_time_span(
            words,
            max_word_seconds=cfg.max_asr_word_duration_seconds,
            max_sentence_seconds=cfg.max_asr_sentence_duration_seconds,
        ):
            diagnostics.write_event(
                "transcript_segment_dropped",
                reason="invalid_or_implausible_timestamps",
                window_id=window_id,
                segment_id=segment_id,
                word_count=len(words),
                start=round(words[0].start, 3) if words else None,
                end=round(words[-1].end, 3) if words else None,
                text=words_to_text(words),
            )
            diagnostics.record_quality_issues(
                "asr_quality",
                ["invalid_or_implausible_timestamps"],
                start=round(words[0].start, 3) if words else None,
                end=round(words[-1].end, 3) if words else None,
                text=words_to_text(words),
                window_id=window_id,
                segment_id=segment_id,
            )
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
        raw_text = words_to_text(words)
        if previous and has_time_overlap(previous_end, current_start):
            words = remove_sentence_overlap_words(previous, words, min_overlap=1)
        recognized_text = words_to_text(words)
        quality = analyze_timed_utterance(
            words,
            audio_duration=len(audio) / cfg.samplerate if audio is not None else None,
        )
        if quality["issues"]:
            diagnostics.record_quality_issues(
                "asr_quality",
                quality["issues"],
                start=round(current_start, 3),
                end=round(current_end, 3),
                text=recognized_text,
                metrics=quality,
            )
        raw_fillers = quality["filler_count"]
        cleaned_text = clean_repetitions(recognized_text)
        if raw_fillers and cleaned_text != recognized_text:
            diagnostics.write_event(
                "transcript_cleanup_candidate",
                start=round(current_start, 3),
                end=round(current_end, 3),
                original=recognized_text,
                cleaned=cleaned_text,
                filler_count=raw_fillers,
                cleanup_applied=True,
            )
        asr_anomaly = is_asr_anomaly(recognized_text)
        with glossary_snapshot_lock:
            glossary_snapshot = glossary_snapshot_state["snapshot"]
        original = glossary_snapshot.normalize_source(
            recognized_text if asr_anomaly else cleaned_text
        )
        if not asr_anomaly and not is_recordable_sentence(
            original
        ):
            diagnostics.record_quality_issues(
                "asr_discarded_text",
                ["possible_unrecordable_fragment"],
                start=round(current_start, 3),
                end=round(current_end, 3),
                text=recognized_text,
                reason="not_recordable_sentence",
                window_id=window_id,
                segment_id=segment_id,
            )
            return None
        with records_lock:
            if len(records) >= cfg.max_stored_records:
                diagnostics.write_event(
                    "history_record_dropped",
                    reason="max_stored_records_reached",
                    limit=cfg.max_stored_records,
                    start=round(current_start, 3),
                    end=round(current_end, 3),
                    text=recognized_text,
                    window_id=window_id,
                    segment_id=segment_id,
                )
                return None
        with records_lock:
            if records:
                last_orig = re.sub(r"[^\w']+", " ", records[-1][0]).strip().lower()
                curr_orig = re.sub(r"[^\w']+", " ", original).strip().lower()
                if not asr_anomaly and last_orig and last_orig == curr_orig:
                    diagnostics.write_event(
                        "history_record_dropped",
                        reason="duplicate_adjacent_record",
                        start=round(current_start, 3),
                        end=round(current_end, 3),
                        text=original,
                        window_id=window_id,
                        segment_id=segment_id,
                    )
                    return None
            records.append([original, ""])
            start = words[0].start if words else 0.0
            end = words[-1].end if words else start
            metadata = {
                "start": start,
                "end": end,
                "version": 1,
                "captured_at_monotonic": captured_at_for_audio_time(end),
            }
            if window_id is not None:
                metadata["window_id"] = window_id
            if segment_id is not None:
                metadata["segment_id"] = segment_id
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
            diagnostics.write_event(
                "transcript_segment",
                record_index=index,
                window_id=window_id,
                segment_id=segment_id,
                start=round(start, 3),
                end=round(end, 3),
                word_count=len(words),
                raw_asr_text=recognized_text,
                cleaned_text=cleaned_text,
                normalized_text=original,
                cleanup_changed=cleaned_text != recognized_text,
                glossary_version=getattr(glossary_snapshot, "version", 0),
                quality_issues=quality["issues"],
                review_reasons=metadata.get("review_reasons", []),
                duplicate_overlap_removed=raw_text != recognized_text,
                captured_at_monotonic=metadata["captured_at_monotonic"],
            )
            diagnostics.write_event(
                "transcript_review_decision",
                record_index=index,
                window_id=window_id,
                segment_id=segment_id,
                review_required=bool(metadata.get("review")),
                review_reasons=metadata.get("review_reasons", []),
                asr_anomaly=asr_anomaly,
                asr_suspect=suspect_asr,
                quality_issues=quality["issues"],
            )
        publish_result("history_add", index, original, metadata, metadata["version"])
        enqueue_copilot_review(index)
        if not enqueue_translation(index):
            publish_result("status", t("status.translation_queue_full"), "", None)
        return index

    def enqueue_copilot_review(index):
        if review_backend is None or stop_event.is_set():
            return
        with records_lock:
            copilot_state["dirty_record_ids"].add(index)

    def submit_copilot_review():
        now = time.monotonic()
        with records_lock:
            if not copilot_state["dirty_record_ids"] or now < copilot_state["next_review_at"]:
                return None
            copilot_state["next_review_at"] = review_backend.next_review_deadline(
                now, cfg.realtime_copilot_interval_seconds
            )

            dirty_ids = sorted(copilot_state["dirty_record_ids"])
            latest_index = len(records) - 1
            full_history, new_records = review_backend.build_bounded_review_window(
                records,
                record_metadata,
                dirty_ids,
                max_records=cfg.realtime_copilot_context_max_records,
                max_targets=cfg.realtime_copilot_max_target_records,
            )
            if not new_records:
                return None
            next_round = copilot_state["round"] + 1
            target_ids = {item["record_id"] for item in new_records}
            context_records = [
                item for item in full_history if item["record_id"] not in target_ids
            ]
            target_records = new_records
            request_payload = {
                "mode": "selected_history_revision",
                "round": next_round,
                "context_window": context_records,
                "history_window": target_records,
                "target_record_ids": [item["record_id"] for item in target_records],
                "new_record_ids": [item["record_id"] for item in new_records],
                "context_summary": copilot_state["context_summary"],
                "term_candidates": copilot_state["term_candidates"],
            }

            payload_chars = len(json.dumps(request_payload, ensure_ascii=False))
            if payload_chars > cfg.realtime_copilot_max_input_chars:
                diagnostics.write_event(
                    "ai_review_context_compressed",
                    round=next_round,
                    original_payload_characters=payload_chars,
                    max_input_characters=cfg.realtime_copilot_max_input_chars,
                    retained_history_records=len(target_records),
                )
                request_payload = {
                    "mode": "compact_full_history_revision",
                    "round": next_round,
                    "context_summary": (
                        copilot_state["context_summary"]
                        + "\n"
                        + "\n".join(
                            item["english"] + " | " + item["chinese"]
                            for item in full_history
                            if item["record_id"] not in {
                                record["record_id"] for record in new_records
                            }
                        )
                    )[-cfg.realtime_copilot_context_chars:],
                    "term_candidates": copilot_state["term_candidates"],
                    "context_window": [],
                    "history_window": new_records,
                    "target_record_ids": [item["record_id"] for item in new_records],
                    "new_record_ids": [item["record_id"] for item in new_records],
                }
                target_records = new_records

                payload_chars = len(json.dumps(request_payload, ensure_ascii=False))
            if payload_chars > cfg.realtime_copilot_max_input_chars:
                diagnostics.write_event(
                    "ai_review_deferred", index=latest_index,
                    reason="request_context_budget_exceeded",
                    round=next_round,
                    payload_characters=payload_chars,
                    max_input_characters=cfg.realtime_copilot_max_input_chars,
                )
                copilot_state["context_summary"] = "\n".join(
                    item["english"] + " | " + item["chinese"]
                    for item in full_history
                )[-cfg.realtime_copilot_context_chars:]
                copilot_state["next_review_at"] = review_backend.next_review_deadline(
                    now, cfg.realtime_copilot_interval_seconds
                )
                return None

            record_ids = [item["record_id"] for item in target_records]
            token = f"{session_id}:{latest_index}:{time.monotonic_ns()}"
            versions = {item: record_metadata[item].get("version", 1) for item in record_ids}
            for item in record_ids:
                record_metadata[item]["copilot_review_submitted"] = True
                record_metadata[item]["copilot_review_token"] = token
            return {
                "record_ids": record_ids,
                "new_record_ids": [item["record_id"] for item in new_records],
                "versions": versions,
                "history": target_records,
                "request_payload": request_payload,
                "round": next_round,
                "retry_attempt": max(
                    (
                        copilot_state["retry_counts"].get(item["record_id"], 0)
                        for item in new_records
                    ),
                    default=0,
                ),
                "latest_index": latest_index,
                "token": token,
                "submitted_at": time.monotonic(),
                "context_summary": copilot_state["context_summary"],
                "term_candidates": list(copilot_state["term_candidates"]),
            }

    def release_copilot_window(record_ids, token, fallback_translation=False):
        with records_lock:
            valid_ids = [
                item for item in record_ids
                if item < len(record_metadata)
                and record_metadata[item].get("copilot_review_token") == token
            ]
            for item in valid_ids:
                record_metadata[item].pop("copilot_review_submitted", None)
                record_metadata[item].pop("copilot_review_token", None)
            needs_fallback = [
                item for item in valid_ids
                if item < len(records) and not records[item][1]
            ] if fallback_translation else []
        for item in needs_fallback:
            enqueue_translation(item)

    def copilot_review_worker():
        try:
            while not stop_event.is_set():
                task = submit_copilot_review()
                if task is None:
                    stop_event.wait(0.1)
                    continue
                record_ids = task["record_ids"]
                latest_index = task["latest_index"]
                token = task["token"]
                if stop_event.is_set():
                    diagnostics.write_event(
                        "copilot_review_skipped", record_ids=record_ids,
                        reason="session_closing",
                    )
                    release_copilot_window(record_ids, token, fallback_translation=True)
                    break

                copilot_request = {}
                try:
                    current_record_ids = set()
                    with records_lock:
                        current_record_ids = review_backend.current_review_record_ids(
                            records, record_metadata, task["history"], record_ids,
                            task["versions"], token,
                        )
                    if not current_record_ids:
                        diagnostics.write_event(
                            "copilot_review_skipped", record_ids=record_ids,
                            reason="history_changed_before_review",
                        )
                        release_copilot_window(record_ids, token, fallback_translation=True)
                        with records_lock:
                            copilot_state["dirty_record_ids"].update(
                                task["new_record_ids"]
                            )
                        continue
                    review_target_records = [
                        item for item in task["history"]
                        if item["record_id"] in current_record_ids
                    ]
                    context_window = [
                        item for item in task["request_payload"].get("context_window", [])
                        if item["record_id"] not in current_record_ids
                    ]
                    review_request_payload = dict(task["request_payload"])
                    review_request_payload["context_window"] = context_window
                    review_request_payload["history_window"] = review_target_records
                    review_request_payload["target_record_ids"] = [
                        item["record_id"] for item in review_target_records
                    ]
                    result = review_backend.review_and_translate(
                        review_target_records,
                        request_payload=review_request_payload,
                        model=cfg.realtime_copilot_model,
                        deadline_seconds=cfg.realtime_copilot_deadline_seconds,
                        submitted_at=task["submitted_at"],
                        cancel_event=stop_event,
                        cli=selected_ai_cli,
                        on_request=lambda request: (
                            copilot_request.update(request),
                            diagnostics.write_event(
                                "copilot_request_started",
                                request_id=request["title"],
                                provider=request["provider"],
                                cli_name=request["cli"],
                                model=request["model"],
                                prompt=request["prompt"],
                                agent_definition=request["agent_definition"],
                                input=request["payload"],
                                record_ids=record_ids,
                                input_characters=sum(
                                    len(item["english"]) + len(item["chinese"])
                                    for item in task["history"]
                                ),
                                github_credits=None,
                                github_credits_status="not_exposed_by_cli",
                            ),
                        ),
                        on_response=lambda response: diagnostics.write_event(
                            "copilot_response_received",
                            request_id=copilot_request.get("title"),
                            provider=copilot_request.get("provider"),
                            model=copilot_request.get("model"),
                            provider_output=response["stdout"],
                            provider_stderr=response["stderr"],
                        ),
                    )
                    updated_items = []
                    accepted_record_ids = set()
                    validated_result = review_backend.validate_review_response(
                        result["raw_response"], review_request_payload,
                        review_target_records, round_number=task["round"],
                        term_candidates=task["term_candidates"],
                        cumulative_rounds=cfg.realtime_copilot_cumulative_rounds,
                    )
                    for item in validated_result["items"]:
                        item_english = item["english"]
                        item_chinese = translator.normalize_output(
                            translator._apply_terms(item_english, item["chinese"])
                        )
                        quality_issues = translator.quality_issues(item_english, item_chinese)
                        if quality_issues:
                            diagnostics.write_event(
                                "copilot_record_rejected",
                                record_id=item["record_id"],
                                reason="translation_quality",
                                issues=quality_issues,
                            )
                            continue
                        updated_items.append({
                            "record_id": item["record_id"],
                            "english": item_english,
                            "chinese": item_chinese,
                        })
                        accepted_record_ids.add(item["record_id"])
                    if stop_event.is_set():
                        raise RuntimeError("Meeting session closed before Copilot candidate was applied")
                    old_records = {}
                    updates = []
                    changed_ids = set()
                    window_content_changed = False
                    with records_lock:
                        applied_record_ids = set()
                        still_current_ids = review_backend.current_review_record_ids(
                            records, record_metadata, task["history"], record_ids,
                            task["versions"], token,
                        )
                        for item in updated_items:
                            if item["record_id"] not in still_current_ids:
                                diagnostics.write_event(
                                    "copilot_record_rejected",
                                    record_id=item["record_id"],
                                    reason="stale_record_version",
                                )
                                continue
                            record_index = item["record_id"]
                            old_en, old_zh = records[record_index]
                            changed = item["english"] != old_en
                            translated_changed = item["chinese"] != old_zh
                            old_records[record_index] = (old_en, old_zh)
                            metadata = record_metadata[record_index]
                            metadata["copilot_completed"] = True
                            metadata["copilot_latency_seconds"] = result["latency_seconds"]
                            if changed or translated_changed:
                                if changed:
                                    metadata["previous_original"] = old_en
                                    metadata["previous_translation"] = old_zh
                                    metadata["corrected"] = True
                                metadata["copilot_corrected"] = changed
                                metadata["version"] = task["versions"][record_index] + 1
                                records[record_index] = [item["english"], item["chinese"]]
                                window_content_changed = True
                                updated_metadata = metadata.copy()
                                record_metadata[record_index] = updated_metadata
                                updates.append({
                                    **item,
                                    "version": updated_metadata["version"],
                                    "metadata": updated_metadata,
                                    "english_changed": changed,
                                    "translation_changed": translated_changed,
                                })
                                if changed:
                                    changed_ids.add(record_index)
                            else:
                                record_metadata[record_index] = metadata.copy()
                            record_metadata[record_index]["translation_version"] = (
                                record_metadata[record_index].get("version", 1)
                            )
                            applied_record_ids.add(record_index)
                        english_replacements = {
                            old_records[item][0]: records[item][0]
                            for item in changed_ids
                        }
                        for context_index, sentences in enumerate(record_contexts):
                            record_contexts[context_index] = [
                                english_replacements.get(sentence, sentence)
                                for sentence in sentences
                            ]
                        for record_index in changed_ids:
                            previous, current = old_records[record_index][0], records[record_index][0]
                            if record_index in context_seeded_records:
                                meeting_context.replace_recent(previous, current)
                    if updates and window_content_changed:
                        publish_result(
                            "history_window_update",
                            updates, None, None,
                        )
                    for record_id in accepted_record_ids:
                        if record_id not in {item["record_id"] for item in updates}:
                            enqueue_translation(record_id)
                    if updates:
                        for item in updates:
                            previous_en, previous_zh = old_records[item["record_id"]]
                            append_revision_log({
                                "index": item["record_id"],
                                "version": item["version"],
                                "start": item["metadata"].get("start"),
                                "end": item["metadata"].get("end"),
                                "previous_en": previous_en if item["english_changed"] else "",
                                "previous_zh": previous_zh,
                                "current_en": item["english"],
                                "current_zh": item["chinese"],
                                "reason": "ai_cli_history_window",
                                "latency_seconds": result["latency_seconds"],
                                "audio_verified": False,
                            })
                    diagnostics.write_event(
                        "copilot_review_applied", latest_index=latest_index,
                        record_ids=record_ids, updated_record_ids=[item["record_id"] for item in updates],
                        latency_seconds=result["latency_seconds"],
                        round=task["round"],
                        history_record_count=len(task["history"]),
                        audio_verified=False,
                    )
                    with records_lock:
                        copilot_state["round"] = task["round"]
                        copilot_state["dirty_record_ids"].difference_update(
                            applied_record_ids
                        )
                        for item in applied_record_ids:
                            copilot_state["retry_counts"].pop(item, None)
                        copilot_state["context_summary"] = validated_result[
                            "context_summary"
                        ]
                        if task["round"] >= cfg.realtime_copilot_cumulative_rounds:
                            copilot_state["term_candidates"] = validated_result[
                                "term_candidates"
                            ]
                    save_history_file()
                    write_timestamped_history()
                    diagnostics.write_event(
                        "ai_review_context_updated",
                        round=task["round"],
                        context_summary=validated_result["context_summary"],
                        term_candidates=validated_result["term_candidates"],
                        term_candidates_used_asr_hints=False,
                    )
                    candidates_path = os.path.join(session_dir, "ai_term_candidates.json")
                    try:
                        with open(candidates_path, "w", encoding="utf-8") as output:
                            json.dump(
                                copilot_state["term_candidates"], output,
                                ensure_ascii=False, indent=2,
                            )
                            output.write("\n")
                    except OSError as error:
                        diagnostics.write_event(
                            "ai_term_candidates_save_failed",
                            path=candidates_path,
                            error=repr(error),
                        )
                    diagnostics.write_event(
                        "copilot_request_completed",
                        request_id=copilot_request.get("title"),
                        provider=copilot_request.get("provider"),
                        model=copilot_request.get("model"),
                        output=result.get("raw_response"),
                        provider_usage=result.get("provider_usage", {}),
                        latency_seconds=result["latency_seconds"],
                        outcome="applied",
                        github_credits=None,
                        github_credits_status="not_exposed_by_cli",
                    )
                    release_copilot_window(record_ids, token)
                    with records_lock:
                        copilot_state["dirty_record_ids"].difference_update(accepted_record_ids)
                        for item in accepted_record_ids:
                            copilot_state["retry_counts"].pop(item, None)
                except Exception as error:
                    if copilot_request:
                        diagnostics.write_event(
                            "copilot_request_completed",
                            request_id=copilot_request.get("title"),
                            provider=copilot_request.get("provider"),
                            model=copilot_request.get("model"),
                            outcome="failed_or_discarded",
                            error=repr(error),
                            latency_seconds=round(time.monotonic() - task["submitted_at"], 3),
                            github_credits=None,
                            github_credits_status="not_exposed_by_cli",
                        )
                    with records_lock:
                        retryable_ids = []
                        for item in task["new_record_ids"]:
                            attempts = copilot_state["retry_counts"].get(item, 0)
                            if review_backend.should_retry_copilot_error(
                                error, attempts, cfg.realtime_copilot_max_retries
                            ):
                                copilot_state["retry_counts"][item] = attempts + 1
                                retryable_ids.append(item)
                            else:
                                copilot_state["dirty_record_ids"].discard(item)

                    diagnostics.write_event(
                        "ai_review_retry_decision",
                        round=task["round"],
                        retry_attempt=task["retry_attempt"],
                        retry_record_ids=retryable_ids,
                        dropped_record_ids=[
                            item for item in task["new_record_ids"]
                            if item not in retryable_ids
                        ],
                        error=repr(error),
                    )
                    diagnostics.write_event(
                        "copilot_review_skipped", record_ids=record_ids,
                        reason=("deadline_exceeded" if isinstance(error, TimeoutError)
                                else "provider_or_validation_error"),
                        error=repr(error),
                        elapsed_seconds=round(time.monotonic() - task["submitted_at"], 3),
                    )
                    release_copilot_window(record_ids, token, fallback_translation=True)
        except Exception as error:
            try:
                report_thread_error(t("worker.copilot"), error)
            except Exception:
                pass
        finally:
            copilot_done_event.set()

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
        nonlocal audio_gap_count, last_history_flush, last_backlog_warning, preview_sequence
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
        window_sequence = 0

        def transcribe_utterance(source_chunks, beam_size):
            """Transcribe with nearby audio so quiet boundary words survive VAD."""
            nonlocal window_sequence
            if not source_chunks:
                return [], None, None, None
            window_sequence += 1
            window_id = f"{session_id}:{window_sequence:05d}"
            with glossary_snapshot_lock:
                glossary_snapshot = glossary_snapshot_state["snapshot"]
            with records_lock:
                recent_context = meeting_context.asr_prompt_context()
            recent_context = glossary_snapshot.normalize_source(recent_context)
            matching_terms = glossary_snapshot.asr_prompt(
                recent_context,
                max_terms=cfg.asr_glossary_prompt_terms,
                include_always=True,
            )
            if review_backend:
                with records_lock:
                    candidate_terms = copilot_state["term_candidates"]
                candidate_terms = [
                    item for item in candidate_terms
                    if item.get("confidence", 0) >= 0.9
                    and item.get("evidence", "").lower() in recent_context.lower()
                ]
                matching_terms = ", ".join(dict.fromkeys(
                    [term for term in matching_terms.split(", ") if term]
                    + review_backend.matching_candidate_terms(recent_context, candidate_terms)
                ))
            asr_started_at = time.monotonic()
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
            diagnostics.write_event(
                "asr_window_input",
                window_id=window_id,
                source_chunk_count=len(source_chunks),
                recent_chunk_count=len(recent_chunks),
                committed_until_seconds=round(
                    committed_until_frame / cfg.samplerate, 3
                ),
                queue_wait_seconds=round(max(
                    0.0, asr_started_at - source_chunks[0].captured_at
                ), 3) if source_chunks[0].captured_at else None,
                target_start=round(source_chunks[0].start_frame / cfg.samplerate, 3),
                target_end=round(source_chunks[-1].end_frame / cfg.samplerate, 3),
                beam_size=beam_size,
                glossary_version=getattr(glossary_snapshot, "version", 0),
                glossary_prompt_term_count=len(
                    [term for term in matching_terms.split(", ") if term]
                ),
                recent_context_chars=len(recent_context),
                audio_boundary_padding_seconds=cfg.audio_boundary_padding_seconds,
            )
            if result.primary_error:
                diagnostics.write_event(
                    "asr_error",
                    stage="primary",
                    beam_size=beam_size,
                    error=repr(result.primary_error),
                    window_id=window_id,
                )
                if result.retry_error:
                    diagnostics.write_event(
                        "asr_error", stage="retry", beam_size=1,
                        error=repr(result.retry_error),
                        window_id=window_id,
                    )
                    publish_result(
                        "status",
                        t("status.asr_retry_failed", error=result.retry_error),
                        "", None,
                    )
                    return [], result.audio, result.audio_start, window_id
                publish_result(
                    "status", t("status.asr_retry", error=result.primary_error), "", None
                )
            timed_words = result.words
            window_quality = analyze_timed_utterance(
                timed_words, audio_duration=result.audio_seconds
            )
            window_start = round(
                source_chunks[0].start_frame / cfg.samplerate, 3
            )
            window_end = round(source_chunks[-1].end_frame / cfg.samplerate, 3)
            window_text = words_to_text(timed_words)
            window_issues = list(window_quality["issues"])
            if not timed_words and result.audio_seconds >= 0.5:
                window_issues.append("no_asr_text_for_audio_window")
            if window_issues:
                diagnostics.record_quality_issues(
                    "asr_window_quality",
                    window_issues,
                    start=window_start,
                    end=window_end,
                    text=window_text,
                    metrics=window_quality,
                    window_id=window_id,
                )
            if not timed_words and result.audio_seconds >= 0.5:
                diagnostics.write_event(
                    "asr_window_no_text",
                    window_id=window_id,
                    start=window_start,
                    end=window_end,
                    audio_seconds=round(result.audio_seconds, 3),
                    audio_signal=analyze_audio_window(result.audio),
                    beam_size=beam_size,
                    retry_attempted=result.primary_error is not None,
                    retry_succeeded=result.primary_error is not None
                    and result.retry_error is None,
                )
            diagnostics.write_event(
                "asr_window",
                window_id=window_id,
                start=window_start,
                end=window_end,
                audio_seconds=round(result.audio_seconds, 3),
                source_chunk_count=len(source_chunks),
                target_start=round(
                    source_chunks[0].start_frame / cfg.samplerate, 3
                ),
                target_end=round(
                    source_chunks[-1].end_frame / cfg.samplerate, 3
                ),
                audio_signal=analyze_audio_window(result.audio),
                word_count=len(timed_words),
                words_per_second=round(
                    len(timed_words) / max(result.audio_seconds, 0.001), 3
                ),
                glossary_version=result.glossary_version,
                queue_wait_seconds=round(max(
                    0.0, asr_started_at - source_chunks[0].captured_at
                ), 3) if source_chunks[0].captured_at else None,
            )
            diagnostics.write_event(
                "asr_result",
                window_id=window_id,
                glossary_version=result.glossary_version,
                words=len(timed_words),
                beam_size=beam_size,
                seconds=round(result.latency_seconds, 3),
                audio_seconds=round(result.audio_seconds, 3),
                queue_wait_seconds=round(max(
                    0.0, time.monotonic() - source_chunks[0].captured_at
                ), 3) if source_chunks[0].captured_at else None,
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
                    window_id=window_id,
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
            diagnostics.write_event(
                "asr_word_timestamps",
                window_id=window_id,
                words=[
                    {
                        "text": word.text,
                        "start": round(word.start, 3),
                        "end": round(word.end, 3),
                        "probability": (
                            round(word.probability, 3)
                            if word.probability is not None else None
                        ),
                    }
                    for word in timed_words
                ],
            )
            return timed_words, result.audio, result.audio_start, window_id

        def log_segmentation(window_id, words, completed, remaining, trigger):
            groups = completed + ([remaining] if remaining else [])
            diagnostics.write_event(
                "asr_segmentation",
                window_id=window_id,
                trigger=trigger,
                input_word_count=len(words),
                completed_count=len(completed),
                remaining_word_count=len(remaining or []),
                segments=[
                    {
                        "candidate_segment_id": f"{window_id}:{index:02d}",
                        "start": round(segment[0].start, 3),
                        "end": round(segment[-1].end, 3),
                        "word_count": len(segment),
                        "text": words_to_text(segment),
                        "finalized": index <= len(completed),
                    }
                    for index, segment in enumerate(groups, 1)
                    if segment
                ],
            )

        def log_segment_grouping(
            window_id, groups, finalized_groups, pending_words, trigger
        ):
            diagnostics.write_event(
                "transcript_segment_grouping",
                window_id=window_id,
                trigger=trigger,
                input_group_count=len(groups),
                finalized_count=len(finalized_groups),
                pending_word_count=len(pending_words or []),
                groups=[
                    {
                        "segment_id": f"{window_id}:{index:02d}",
                        "start": round(group[0].start, 3),
                        "end": round(group[-1].end, 3),
                        "word_count": len(group),
                        "text": words_to_text(group),
                        "finalized": index <= len(finalized_groups),
                    }
                    for index, group in enumerate(groups, 1)
                    if group
                ],
            )

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
                        timed_words, _, _, window_id = transcribe_utterance(utterance_chunks, 1)
                        if tail_start is not None and tail_end is not None:
                            timed_words = words_in_time_range(
                                timed_words,
                                tail_start / cfg.samplerate,
                                tail_end / cfg.samplerate,
                            )
                        completed, tail = split_timed_sentences(
                            timed_words, terminal_is_boundary=False
                        )
                        log_segmentation(
                            window_id, timed_words, completed, tail, "waiting_timeout"
                        )
                        for segment_number, sentence_words in enumerate(completed, 1):
                            if is_recordable_sentence(words_to_text(sentence_words)):
                                record = add_history_sentence(
                                    sentence_words, extract_sentence_audio(sentence_words),
                                    window_id, f"{window_id}:{segment_number:02d}",
                                )
                                schedule_correction(record, sentence_words)
                        if tail and should_finalize_after_waiting_timeout(tail):
                            tail_segment = len(completed) + 1
                            record = add_history_sentence(
                                tail, extract_sentence_audio(tail), window_id,
                                f"{window_id}:{tail_segment:02d}",
                            )
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

                chunk_age = max(0.0, time.monotonic() - chunk.captured_at)
                if chunk_age >= cfg.audio_backlog_warning_seconds:
                    capture_stats["stale_chunks"] += 1
                    capture_stats["stale_frames"] += len(chunk.data)
                    diagnostics.increment_quality("audio_stale_chunks")
                    diagnostics.increment_quality("audio_stale_chunk_frames", len(chunk.data))
                    diagnostics.write_event(
                        "audio_chunk_stale",
                        start_frame=chunk.start_frame,
                        end_frame=chunk.end_frame,
                        age_seconds=round(chunk_age, 3),
                        mode=cfg.mode,
                    )

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
                    gap_frames = max(
                        0,
                        chunk.start_frame - last_chunk_end_frame
                        if last_chunk_end_frame is not None else 0,
                    )
                    audio_gap_count += 1
                    diagnostics.increment_quality("audio_timeline_gap_events")
                    diagnostics.increment_quality("audio_timeline_gap_frames", gap_frames)
                    diagnostics.increment_quality("asr_skipped_audio_frames", gap_frames)
                    capture_stats["asr_gap_events"] = capture_stats.get("asr_gap_events", 0) + 1
                    capture_stats["asr_gap_frames"] = capture_stats.get("asr_gap_frames", 0) + gap_frames
                    with capture_timestamps_lock:
                        while (
                            capture_timestamps
                            and capture_timestamps[-1][0] >= chunk.start_frame
                        ):
                            capture_timestamps.pop()
                    diagnostics.write_event(
                        "audio_gap",
                        gap_count=audio_gap_count,
                        gap_frames=gap_frames,
                        gap_seconds=round(gap_frames / cfg.samplerate, 3),
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
                    include_always=True,
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
                    current_duration = audio_chunks_duration(
                        utterance_chunks, cfg.samplerate
                    )

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
                        preview_captured_at = captured_at_for_audio_time(
                            p_end / cfg.samplerate
                        )
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
                                diagnostics.write_event(
                                    "asr_error",
                                    stage="preview",
                                    start=round(p_start / cfg.samplerate, 3),
                                    end=round(p_end / cfg.samplerate, 3),
                                    error=repr(error),
                                )
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
                                    preview_sequence += 1
                                    preview_id = preview_sequence
                                    publish_result(
                                        "english", p_text, "",
                                        {
                                            "pipeline_id": preview_id,
                                            "captured_at_monotonic": preview_captured_at,
                                        },
                                    )
                                else:
                                    preview_id = None
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
                                        if revision is not None:
                                            pipeline_metrics.remember_live(
                                                revision, preview_captured_at
                                            )
                                            translation_ready.set()

                    # Bound continuous speech and split at punctuation or phrase boundaries.
                    force_interval = int(2.0 * cfg.samplerate)
                    force_due = (
                        last_forced_transcription_end is None
                        or chunk.end_frame - last_forced_transcription_end >= force_interval
                    )
                    if current_duration >= cfg.max_utterance_seconds and force_due:
                        last_forced_transcription_end = chunk.end_frame
                        timed_words, full_audio, _, window_id = transcribe_utterance(
                            utterance_chunks, current_beam
                        )
                        if timed_words:
                            completed, remaining = split_timed_sentences(
                                timed_words, terminal_is_boundary=False
                            )
                            log_segmentation(
                                window_id, timed_words, completed, remaining,
                                "forced_max_utterance",
                            )
                            for boundary in classify_segmentation_boundaries(completed):
                                diagnostics.record_quality_issues(
                                    "segmentation_quality",
                                    ["possible_pause_boundary_fragment"],
                                    start=boundary["left_end"],
                                    end=boundary["right_start"],
                                    text=boundary["left_text"] + " | " + boundary["right_text"],
                                    metrics=boundary,
                                )
                            groups = merge_incomplete_sentences(
                                completed + ([remaining] if remaining else [])
                            )
                            pending_words = groups[-1] if remaining and groups else []
                            finalized_groups = groups[:-1] if remaining else groups
                            log_segment_grouping(
                                window_id, groups, finalized_groups, pending_words,
                                "forced_max_utterance",
                            )
                            if finalized_groups:
                                terminal_segments = []
                                for segment_number, s_words in enumerate(finalized_groups, 1):
                                    s_text = finalize_history_sentence(s_words)
                                    if s_text:
                                        s_audio = extract_sentence_audio(s_words)
                                        cand = add_history_sentence(
                                            s_words, s_audio, window_id,
                                            f"{window_id}:{segment_number:02d}",
                                        )
                                        schedule_correction(cand, s_words)
                                        disposition = "saved"
                                    else:
                                        disposition = "rejected"
                                        diagnostics.write_event(
                                            "transcript_segment_rejected",
                                            reason="not_recordable_sentence",
                                            window_id=window_id,
                                            segment_id=f"{window_id}:{segment_number:02d}",
                                            start=round(s_words[0].start, 3),
                                            end=round(s_words[-1].end, 3),
                                            text=words_to_text(s_words),
                                        )
                                    terminal_segments.append(disposition)
                                if terminal_segments and all(
                                    segment_should_advance_committed_until(item)
                                    for item in terminal_segments
                                ):
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
                                        timed_words, extract_sentence_audio(timed_words),
                                        window_id, f"{window_id}:01",
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
                        current_duration = audio_chunks_duration(
                            utterance_chunks, cfg.samplerate
                        )

                        # Finalize after sufficient silence when the utterance is long enough.
                        if silence_duration >= cfg.vad_pause_seconds and current_duration >= 0.35:
                            timed_words, full_audio, _, window_id = transcribe_utterance(
                                utterance_chunks, current_beam
                            )
                            if timed_words:
                                completed, remaining = split_timed_sentences(timed_words)
                                log_segmentation(
                                    window_id, timed_words, completed, remaining,
                                    "silence_boundary",
                                )
                                for boundary in classify_segmentation_boundaries(completed):
                                    diagnostics.record_quality_issues(
                                        "segmentation_quality",
                                        ["possible_pause_boundary_fragment"],
                                        start=boundary["left_end"],
                                        end=boundary["right_start"],
                                        text=boundary["left_text"] + " | " + boundary["right_text"],
                                        metrics=boundary,
                                    )
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
                                log_segment_grouping(
                                    window_id, groups, finalized_groups, pending_words,
                                    "silence_boundary",
                                )
                                terminal_segments = []
                                for segment_number, sentence_words in enumerate(finalized_groups, 1):
                                    finalized = finalize_history_sentence(sentence_words)
                                    if finalized:
                                        cand = add_history_sentence(
                                            sentence_words, extract_sentence_audio(sentence_words),
                                            window_id,
                                            f"{window_id}:{segment_number:02d}",
                                        )
                                        schedule_correction(cand, sentence_words)
                                        finalized_any = True
                                        disposition = "saved"
                                    else:
                                        disposition = "rejected"
                                        diagnostics.write_event(
                                            "transcript_segment_rejected",
                                            reason="not_recordable_sentence",
                                            window_id=window_id,
                                            segment_id=f"{window_id}:{segment_number:02d}",
                                            start=round(sentence_words[0].start, 3),
                                            end=round(sentence_words[-1].end, 3),
                                            text=words_to_text(sentence_words),
                                        )
                                    terminal_segments.append(disposition)
                                if terminal_segments and all(
                                    segment_should_advance_committed_until(item)
                                    for item in terminal_segments
                                ):
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
                    timed_words, full_audio, _, window_id = transcribe_utterance(utterance_chunks, 1)
                    if full_audio is not None and len(full_audio) >= int(0.3 * cfg.samplerate):
                        timed_words = filter_uncommitted_words(
                            timed_words, committed_until_frame / cfg.samplerate
                        )
                        completed, tail = split_timed_sentences(
                            timed_words, terminal_is_boundary=False
                        )
                        log_segmentation(window_id, timed_words, completed, tail, "shutdown_flush")
                        for segment_number, sentence_words in enumerate(completed, 1):
                            if is_recordable_sentence(words_to_text(sentence_words)):
                                record = add_history_sentence(
                                    sentence_words, extract_sentence_audio(sentence_words),
                                    window_id, f"{window_id}:{segment_number:02d}",
                                )
                                schedule_correction(record, sentence_words)
                        if tail and should_finalize_after_waiting_timeout(tail):
                            tail_segment = len(completed) + 1
                            record = add_history_sentence(
                                tail, extract_sentence_audio(tail), window_id,
                                f"{window_id}:{tail_segment:02d}",
                            )
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
                        "asr_correction_candidate",
                        glossary_version=getattr(glossary_snapshot, "version", 0),
                        index=index,
                        record_index=index,
                        record_version=expected_version,
                        target_start=round(target_start / cfg.samplerate, 3),
                        target_end=round(target_end / cfg.samplerate, 3),
                        context_audio_start=round(actual_start / cfg.samplerate, 3),
                        context_audio_seconds=round(
                            len(context_audio) / cfg.samplerate, 3
                        ),
                        beam_size=cfg.asr_correction_beam_size,
                        seconds=round(time.monotonic() - correction_started, 3),
                        words=len(corrected_words),
                        original=expected_original,
                        candidate=words_to_text(corrected_words),
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
                    diagnostics.write_event(
                        "asr_correction_decision",
                        index=index,
                        record_index=index,
                        record_version=expected_version,
                        start=round(target_start / cfg.samplerate, 3)
                        if target_start is not None else None,
                        end=round(target_end / cfg.samplerate, 3)
                        if target_end is not None else None,
                        original=original,
                        candidate=corrected,
                        accepted=correction_rejection_reason is None,
                        rejection_reason=correction_rejection_reason,
                    )
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
                                record_metadata[index].pop("copilot_completed", None)
                                record_metadata[index].pop("copilot_review_submitted", None)
                                record_metadata[index].pop("copilot_corrected", None)
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
                            diagnostics.write_event(
                                "asr_revision",
                                index=index,
                                start=round(target_start / cfg.samplerate, 3)
                                if target_start is not None else None,
                                end=round(target_end / cfg.samplerate, 3)
                                if target_end is not None else None,
                                previous=original,
                                current=corrected,
                                reason="contextual_redecode_confidence_gate",
                                correction_applied=True,
                            )
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
                            enqueue_copilot_review(index)
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
                or not copilot_done_event.is_set()
                or history_translations
                or not live_translation_queue.empty()
                )
            ):
                translation_ready.clear()
                history_job = history_translations.take()
                live_job = None
                if history_job is None:
                    with live_state_lock:
                        try:
                            live_job = live_translation_queue.get_nowait()
                        except queue.Empty:
                            pass
                if live_job is not None:
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
                if history_job is None:
                    translation_ready.wait(timeout=0.2)
                    continue
                index, enqueued_at = history_job
                queue_wait_seconds = round(time.monotonic() - enqueued_at, 3)

                with records_lock:
                    if index >= len(records):
                        diagnostics.write_event(
                            "history_translation_skipped",
                            index=index,
                            reason="record_missing",
                        )
                        continue
                    original = records[index][0]
                    existing_translation = records[index][1]
                    metadata = record_metadata[index].copy()
                    context_sentences = list(record_contexts[index])
                if not original.strip():
                    diagnostics.write_event(
                        "history_translation_skipped",
                        index=index,
                        reason="empty_source",
                    )
                    continue
                if (
                    existing_translation
                    and metadata.get("translation_version") == metadata.get("version")
                ):
                    diagnostics.write_event(
                        "history_translation_skipped", index=index,
                        reason="translation_already_current",
                    )
                    continue
                translation_start = metadata.get("start", 0.0)
                translation_end = metadata.get("end", 0.0)
                translation_version = metadata.get("version")
                window_id = metadata.get("window_id")
                segment_id = metadata.get("segment_id")
                original_record_metadata = metadata.copy()
                glossary_snapshot = terminology.snapshot()

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
                    translation_mode = (
                        "context"
                        if (
                            cfg.translation_context_enabled
                            and context_sentences
                            and len(original.split()) >= 4
                            and same_record_window
                        ) else "sentence"
                    )
                    diagnostics.write_event(
                        "translation_started",
                        index=index,
                        record_index=index,
                        window_id=window_id,
                        segment_id=segment_id,
                        start=round(translation_start, 3),
                        end=round(translation_end, 3),
                        source_word_count=len(original.split()),
                        source_char_count=len(original),
                        context_sentence_count=len(context_sentences),
                        translation_mode=translation_mode,
                        model_backend=("ctranslate2" if translator.ct2_model else "pytorch"),
                        model_device=translator.device,
                        glossary_version=getattr(glossary_snapshot, "version", 0),
                        glossary_terms=len(getattr(glossary_snapshot, "entries", [])),
                        record_version=translation_version,
                        pending_records=len(history_translations),
                        queue_wait_seconds=round(queue_wait_seconds, 3),
                    )
                    diagnostics.update_quality_max(
                        "max_pending_history_translations", len(history_translations)
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
                                record_index=index,
                                window_id=window_id,
                                segment_id=segment_id,
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
                        diagnostics.write_event(
                            "translation_quality_retry",
                            index=index,
                            record_index=index,
                            window_id=window_id,
                            segment_id=segment_id,
                            english=original,
                            initial_chinese=translated,
                            initial_issues=list(issues),
                            retry_started=bool(cfg.translation_retry_count),
                        )
                        publish_result(
                            "status",
                            t("status.translation_warning", index=index + 1, issues=", ".join(issues)),
                            "",
                            None,
                        )
                        if cfg.translation_retry_count:
                            try:
                                translated = translator.translate(original, quality=True)
                                diagnostics.write_event(
                                    "translation_quality_retry_result",
                                    index=index,
                                    record_index=index,
                                    window_id=window_id,
                                    segment_id=segment_id,
                                    english=original,
                                    chinese=translated,
                                    issues=translator.quality_issues(original, translated),
                                )
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
                            copilot_completed = record_metadata[index].get("copilot_completed")
                            if not copilot_completed:
                                updated_metadata = apply_translation_quality_status(
                                    record_metadata[index], remaining_issues
                                )
                                record_metadata[index].clear()
                                record_metadata[index].update(updated_metadata)
                                metadata = updated_metadata
                    translation_metadata_changed = any(
                        original_record_metadata.get(key) != metadata.get(key)
                        for key in (
                            "review", "review_reasons", "translation_review",
                            "translation_issues",
                        )
                    )
                    diagnostics.write_event(
                        "history_translation",
                        index=index,
                        record_index=index,
                        window_id=window_id,
                        segment_id=segment_id,
                        start=round(translation_start, 3),
                        end=round(translation_end, 3),
                        record_version=translation_version,
                        translation_mode=translation_mode,
                        model_backend=("ctranslate2" if translator.ct2_model else "pytorch"),
                        source_word_count=len(original.split()),
                        translated_char_count=len(translated),
                        quality_metadata_updated=translation_metadata_changed,
                        seconds=round(time.monotonic() - history_translation_started, 3),
                        issues=remaining_issues,
                        english=original,
                        chinese=translated,
                        review_required=bool(remaining_issues),
                    )
                    if remaining_issues:
                        diagnostics.record_quality_issues(
                            "translation_quality",
                            remaining_issues,
                            start=round(translation_start, 3),
                            end=round(translation_end, 3),
                            english=original,
                            chinese=translated,
                            record_index=index,
                            window_id=window_id,
                            segment_id=segment_id,
                        )
                    with records_lock:
                        current = (
                            index < len(records)
                            and records[index][0] == original
                            and record_metadata[index]["version"] == metadata["version"]
                            and not record_metadata[index].get("copilot_completed")
                        )
                        if current:
                            records[index][1] = translated
                            record_metadata[index]["translation_version"] = metadata["version"]
                            revision_metadata = record_metadata[index].copy()
                        else:
                            revision_metadata = None
                    if current:
                        if remaining_issues:
                            publish_result("history_review", index, "", None, version=translation_version)
                            publish_result(
                                "status",
                                t("status.translation_unresolved", index=index + 1, issues=", ".join(remaining_issues)),
                            )
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
                        current = (
                            record_metadata[index]["version"] == translation_version
                            and not record_metadata[index].get("copilot_completed")
                        )
                        if current:
                            updated_metadata = set_review_reason(
                                record_metadata[index], "translation_error"
                            )
                            record_metadata[index].clear()
                            record_metadata[index].update(updated_metadata)
                    if current:
                        publish_result("history_review", index, "", None, version=translation_version)
                    diagnostics.write_event(
                        "translation_error",
                        stage="history",
                        index=index,
                        english=original,
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
    if review_backend:
        worker_runtime.start(t("worker.copilot"), copilot_review_worker)
    else:
        copilot_done_event.set()
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
                presentation = apply_result_event(
                    overlay, cfg, (event, first, second, fourth, event_version),
                    live_state, live_state_lock,
                )
                if presentation:
                    sample = (
                        pipeline_metrics.presented_english(
                            presentation["kind"], presentation["key"],
                            presentation["applied_at"],
                        )
                        if presentation["kind"] == "preview_english"
                        else pipeline_metrics.presented_committed_english(
                            presentation["key"], presentation["applied_at"]
                        )
                        if presentation["kind"] == "committed_english"
                        else pipeline_metrics.presented(
                            presentation["kind"], presentation["key"],
                            presentation["applied_at"],
                        )
                    )
                    if sample:
                        diagnostics.write_event(
                            "subtitle_presented", **sample,
                            record_id=(presentation["key"][0]
                                       if isinstance(presentation["key"], tuple) else None),
                            record_version=(presentation["key"][1]
                                            if isinstance(presentation["key"], tuple) else None),
                            revision=(presentation["key"]
                                      if isinstance(presentation["key"], int) else None),
                        )
                        if presentation["kind"] == "committed_chinese":
                            pipeline_metrics.clear_record(presentation["key"])

            drain_worker_errors(error_queue, overlay)
            now = time.monotonic()
            if (
                result_queue.qsize() > cfg.max_result_queue_items * 0.8
                and now - ui_backlog_state["last_report"] >= 5.0
            ):
                diagnostics.write_event(
                    "ui_queue_backlog",
                    queue_depth=result_queue.qsize(),
                    soft_limit=cfg.max_result_queue_items,
                    events_processed=processed,
                    refresh_interval_ms=100,
                )
                ui_backlog_state["last_report"] = now
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
        warnings.showwarning = previous_warning_hook
        warning_recorder.flush()
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
            audio_capture_underflow_warnings=capture_stats["runtime_underflows"],
            audio_capture_overflow_warnings=capture_stats["runtime_overflows"],
            audio_capture_timeline_gap_events=capture_stats["timeline_gap_events"],
            audio_capture_timeline_gap_frames=capture_stats["timeline_gap_frames"],
            asr_detected_gap_events=capture_stats.get("asr_gap_events", 0),
            asr_detected_gap_frames=capture_stats.get("asr_gap_frames", 0),
            low_latency_queue_dropped_chunks=capture_stats["dropped_chunks"],
            low_latency_queue_dropped_frames=capture_stats["dropped_frames"],
            stale_audio_chunks=capture_stats["stale_chunks"],
            stale_audio_frames=capture_stats["stale_frames"],
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
            runtime_warning_counts=warning_recorder.summary_counts(),
            warning_counts=diagnostics.warning_counts(),
            ui_result_queue_remaining=result_queue.qsize(),
            pending_history_translations=len(history_translations),
            pipeline_latency=PipelineMetrics.percentile_summary(
                pipeline_metrics.summary()
            ),
            workers_still_running=unfinished_workers,
            transcript_path=os.path.join(session_dir, cfg.history_file),
            audio_path=os.path.join(session_dir, cfg.audio_filename),
        )


if __name__ == "__main__":
    main()
