import os
import datetime
import json
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
    start_capture_warnings,
    stop_capture_warnings,
)
from config import Config
from diagnostics import SessionDiagnostics
from default_terminology import DEFAULT_TERMINOLOGY
from model_validation import missing_model_files
from overlay import Overlay
from text_processing import (
    TimedWord,
    clean_repetitions,
    split_timed_sentences,
    words_to_text,
    correction_is_usable,
    correction_is_confident,
    merge_incomplete_sentences,
    is_recordable_sentence,
    is_review_candidate,
    is_likely_incomplete,
    remove_sentence_overlap_words,
    is_asr_anomaly,
    has_valid_time_span,
    words_in_time_range,
    select_timed_words_for_target,
    finalize_pending_group,
    trim_trailing_silence_chunks,
    filter_uncommitted_words,
    select_timed_words_for_correction,
    should_schedule_correction,
    should_hold_after_waiting_silence,
    should_finalize_after_waiting_timeout,
)
from translator import Translator
from terminology import Terminology
from meeting_context import MeetingContext
from session_archive import AudioRecorder, create_session_directory
from stable_text import StableText
from speech_detection import SileroSpeechDetector, contains_voice
from translation_queue import publish_live_translation, take_latest


def correction_prompt(records, index, limit=3):
    indexes = range(max(0, index - limit), index)
    return " ".join(records[item_index][0] for item_index in indexes)[-360:]


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
    capture_done_event = threading.Event()
    asr_done_event = threading.Event()
    worker_done_event = threading.Event()
    correction_done_event = threading.Event()
    records = []
    record_metadata = []
    record_contexts = []
    records_lock = threading.Lock()
    correction_pending = set()
    last_history_flush = time.monotonic()
    last_backlog_warning = 0.0
    history_save_lock = threading.Lock()
    revision_log_lock = threading.Lock()
    live_state_lock = threading.Lock()
    live_state = {"revision": 0}

    error_queue = queue.Queue()
    audio_gap_count = 0
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
        raise FileNotFoundError("模型文件缺失: " + ", ".join(missing))
    asr = ASR(
        cfg.whisper_model, cfg.whisper_device, cfg.whisper_compute_type,
        os.path.join(root_dir, "models", "whisper_base_en"),
        cpu_threads=cfg.whisper_cpu_threads,
    )
    terminology_path = os.path.join(root_dir, cfg.terminology_file)
    if os.path.isfile(terminology_path):
        terminology = Terminology.from_file(terminology_path)
    else:
        terminology = Terminology(DEFAULT_TERMINOLOGY)
        print("[main] 未找到 terminology.json，使用通用内置术语表。")
    meeting_context = MeetingContext(
        terminology,
        max_sentences=cfg.context_max_sentences,
        max_chars=cfg.context_max_chars,
    )
    translator = Translator(
        os.path.join(root_dir, cfg.translate_model_dir),
        cfg.translate_device,
        terminology,
        intra_threads=cfg.translate_intra_threads,
        inter_threads=cfg.translate_inter_threads,
    )
    try:
        speech_detector = SileroSpeechDetector(cfg.samplerate)
        print("[vad] Silero VAD 已启用；仅用于能量门控未检测到语音的音频块")
    except Exception as error:
        speech_detector = None
        print(f"[vad] Silero VAD 不可用，使用原能量门控: {error}")
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_dir = create_session_directory(
        os.path.join(root_dir, "sessions"),
        session_id,
        prefix=cfg.session_directory_prefix,
    )
    print(f"[main] 本次会议文件夹: {session_dir}")
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

    def record_capture_warning(category, message, filename, lineno, count):
        if "discontinuity" in message.lower():
            capture_stats["runtime_discontinuities"] += count
            diagnostics.write_event(
                "audio_capture_discontinuity_warning",
                count=count,
                category=category,
                message=message,
            )
        diagnostics.record_warning(category, message, filename, lineno, count)

    warning_recorder = start_capture_warnings(
        record_capture_warning
    )
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
        "完整模式：优先保证内容完整"
        if cfg.mode == "complete"
        else "低延迟模式：优先追上当前语音"
    )

    def report_thread_error(name, error):
        diagnostics.write_event("worker_error", worker=name, error=repr(error))
        error_queue.put((name, error))
        if name in {"音频采集", "语音识别"}:
            stop_event.set()

    def publish_result(event, first, second="", fourth=None, version=None):
        item = (event, first, second, fourth, version)
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

    def enqueue_translation(index):
        while True:
            try:
                history_translation_queue.put(index, timeout=0.2)
                return True
            except queue.Full:
                if stop_event.is_set() and worker_done_event.is_set():
                    return False
                continue

    def capture_worker():
        try:
            capture_loop(
                device, cfg.samplerate, chunk_frames, audio_queue, stop_event, cfg,
                audio_sink=audio_recorder,
                diagnostic_callback=diagnostics.write_event,
                capture_stats=capture_stats,
                audio_timeline_sink=audio_recorder,
                consumer_done_event=asr_done_event,
            )
        except Exception as error:
            report_thread_error("音频采集", error)
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
        if previous and previous_end is not None and current_start - previous_end <= 2.0:
            words = remove_sentence_overlap_words(previous, words)
        recognized_text = words_to_text(words)
        asr_anomaly = is_asr_anomaly(recognized_text)
        original = terminology.normalize_source(
            recognized_text if asr_anomaly else clean_repetitions(recognized_text)
        )
        if not asr_anomaly and not is_recordable_sentence(
            original, allow_single_word=bool(terminology.matching_terms(original))
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
            if asr_anomaly or is_review_candidate(original):
                metadata["review"] = True
            record_metadata.append(metadata)
            record_contexts.append(meeting_context.recent_sentences(limit=3))
            index = len(records) - 1
            if not asr_anomaly:
                meeting_context.add(original)
        publish_result("history_add", index, original, metadata, metadata["version"])
        if not enqueue_translation(index):
            publish_result("status", "翻译队列未能接收任务，英文记录已保留", "", None)
        return index

    def save_history_file():
        with history_save_lock:
            with records_lock:
                saved_records = [tuple(record) for record in records]
                saved_metadata = [metadata.copy() for metadata in record_metadata]
            history_path = os.path.join(session_dir, cfg.history_file)
            content = "".join(
                f"[{Overlay._format_timestamp(saved_metadata[index]['start'])}-"
                f"{Overlay._format_timestamp(saved_metadata[index]['end'])}] "
                f"{'[需要复核] ' if saved_metadata[index].get('review') else ''}"
                f"{'[已修订] ' if saved_metadata[index].get('corrected') else ''}"
                f"[EN] {original}\n[ZH] {translated}\n"
                f"{'[EN - 修订前] ' + saved_metadata[index]['previous_original'] + chr(10) if saved_metadata[index].get('previous_original') else ''}"
                f"{'[ZH - 修订前] ' + saved_metadata[index]['previous_translation'] + chr(10) if saved_metadata[index].get('previous_translation') else ''}\n"
                for index, (original, translated) in enumerate(saved_records)
            )
            temporary_path = history_path + ".tmp"
            with open(temporary_path, "w", encoding="utf-8") as output:
                output.write(content)
            os.replace(temporary_path, history_path)

    def write_timestamped_history():
            with records_lock:
                saved_records = [tuple(record) for record in records]
                saved_metadata = [metadata.copy() for metadata in record_metadata]
            timestamped_file = f"meeting_transcript_{session_id}.txt"
            timestamped_path = os.path.join(session_dir, timestamped_file)
            content_lines = []
            for index, (original, translated) in enumerate(saved_records):
                metadata = saved_metadata[index]
                timestamps = (
                    f"[{Overlay._format_timestamp(metadata['start'])}-"
                    f"{Overlay._format_timestamp(metadata['end'])}] "
                )
                review = "[需要复核] " if metadata.get("review") else ""
                corrected = "[已修订] " if metadata.get("corrected") else ""
                content_lines.append(
                    f"{timestamps}{review}{corrected}[EN] {original}\n[ZH] {translated}\n"
                )
                if metadata.get("previous_original"):
                    content_lines.append(f"[EN - 修订前] {metadata['previous_original']}\n")
                if metadata.get("previous_translation"):
                    content_lines.append(f"[ZH - 修订前] {metadata['previous_translation']}\n")
                content_lines.append("\n")
            temporary_path = timestamped_path + ".tmp"
            with open(temporary_path, "w", encoding="utf-8") as output:
                output.write("".join(content_lines))
            os.replace(temporary_path, timestamped_path)

    def append_revision_log(entry):
        revision_path = os.path.join(session_dir, "meeting_revisions.jsonl")
        with revision_log_lock:
            with open(revision_path, "a", encoding="utf-8") as output:
                output.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def finalize_history_sentence(words):
        text = terminology.normalize_source(clean_repetitions(words_to_text(words)))
        if not is_recordable_sentence(
            text, allow_single_word=bool(terminology.matching_terms(text))
        ):
            return None
        return text

    def worker():
        nonlocal audio_gap_count, last_history_flush, last_backlog_warning
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
            padding = int(cfg.audio_boundary_padding_seconds * cfg.samplerate)
            target_start = source_chunks[0].start_frame
            target_end = source_chunks[-1].end_frame
            if committed_until_frame:
                target_start = max(target_start, committed_until_frame)
            available_chunks = list(recent_chunks)
            known_starts = {chunk.start_frame for chunk in available_chunks}
            available_chunks.extend(
                chunk for chunk in source_chunks
                if chunk.start_frame not in known_starts
            )
            available_chunks.sort(key=lambda chunk: chunk.start_frame)
            audio_start = max(0, target_start - padding)
            audio_end = target_end + padding
            audio_start = max(audio_start, available_chunks[0].start_frame)
            audio_end = min(audio_end, available_chunks[-1].end_frame)
            audio = extract_audio_range(available_chunks, audio_start, audio_end)
            if len(audio) == 0:
                return [], np.empty(0, dtype=np.float32), None
            with records_lock:
                recent_context = meeting_context.asr_prompt_context()
            recent_context = terminology.normalize_source(recent_context)
            matching_terms = terminology.asr_prompt(
                recent_context, max_terms=cfg.asr_glossary_prompt_terms
            )
            audio = highpass(audio, cfg.samplerate)
            try:
                with asr_lock:
                    words = asr.transcribe(
                        audio,
                initial_prompt=matching_terms or None,
                        hotwords=matching_terms or None,
                        beam_size=beam_size,
                        repetition_penalty=cfg.asr_repetition_penalty,
                        max_initial_timestamp=cfg.asr_max_initial_timestamp,
                    )
            except Exception as error:
                diagnostics.write_event(
                    "asr_error", stage="primary", beam_size=beam_size, error=repr(error)
                )
                publish_result("status", f"ASR 首次推理失败，尝试轻量重试: {error}", "", None)
                try:
                    with asr_lock:
                        words = asr.transcribe(
                            audio,
                            initial_prompt=matching_terms or None,
                            hotwords=matching_terms or None,
                            beam_size=1,
                            repetition_penalty=cfg.asr_repetition_penalty,
                            max_initial_timestamp=cfg.asr_max_initial_timestamp,
                        )
                except Exception as retry_error:
                    diagnostics.write_event(
                        "asr_error", stage="retry", beam_size=1, error=repr(retry_error)
                    )
                    publish_result("status", f"ASR 重试失败，该音频片段未能识别: {retry_error}", "", None)
                    return [], audio, audio_start
            diagnostics.write_event(
                "asr_result",
                words=len(words),
                beam_size=beam_size,
                seconds=round(asr.last_latency_seconds, 3),
                audio_seconds=round(len(audio) / cfg.samplerate, 3),
            )
            timed_words = [
                TimedWord(
                    (audio_start / cfg.samplerate) + word.start,
                    (audio_start / cfg.samplerate) + word.end,
                    word.text,
                    word.probability,
                )
                for word in words
            ]
            if asr.last_latency_seconds > cfg.asr_timeout_seconds and beam_size > 1:
                publish_result(
                    "status",
                    f"ASR 片段处理耗时 {asr.last_latency_seconds:.1f}s，超过延迟预算",
                    "",
                    None,
                )
            # Padding is only recognition context. Do not let speech from the
            # previous utterance leak into the current transcript.
            timed_words = select_timed_words_for_target(
                timed_words,
                target_start / cfg.samplerate,
                target_end / cfg.samplerate,
                max_word_seconds=cfg.max_asr_word_duration_seconds,
            )
            return timed_words, audio, audio_start

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
                        records, index, limit=cfg.history_context_window
                    )
                correction_candidates.append(
                    (
                        index,
                        int(words[0].start * cfg.samplerate),
                        int(words[-1].end * cfg.samplerate),
                        prompt,
                        [word.probability for word in words if word.probability is not None],
                    )
                )

        def enqueue_ready_corrections(force=False):
            if cfg.mode != "complete" or not recent_chunks:
                return
            available_end = recent_chunks[-1].end_frame
            for candidate in list(correction_candidates):
                index, target_start, target_end, prompt, original_confidences = candidate
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
                        )
                    )
                except queue.Full:
                    with records_lock:
                        correction_pending.discard(index)
                    publish_result("status", "纠错队列已满，暂缓本句延迟纠错", "", None)
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
                        publish_result("live_reset", "", "", None)
                        if time.monotonic() - last_history_flush >= cfg.history_flush_seconds:
                            save_history_file()
                            write_timestamped_history()
                            last_history_flush = time.monotonic()
                    continue
                audio_queue.task_done()
                if not isinstance(chunk, AudioChunk):
                    continue

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
                    publish_result("live_reset", "", "", None)
                    publish_result("status", f"音频流出现第 {audio_gap_count} 个间隙", "", None)
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
                recent_context = terminology.normalize_source(recent_context)
                matched_terms = terminology.asr_prompt(
                    recent_context, max_terms=cfg.asr_glossary_prompt_terms
                )

                backlog_seconds = audio_queue.qsize() * cfg.chunk_seconds
                if (
                    backlog_seconds >= cfg.audio_backlog_warning_seconds
                    and time.monotonic() - last_backlog_warning >= 5.0
                ):
                    last_backlog_warning = time.monotonic()
                    publish_result(
                        "status",
                        f"音频积压约 {backlog_seconds:.1f} 秒，正在追赶实时内容",
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
                                            initial_prompt=matched_terms or None,
                                        hotwords=matched_terms or None,
                                        beam_size=1,
                                        word_timestamps=False,
                                        repetition_penalty=cfg.asr_repetition_penalty,
                                        max_initial_timestamp=cfg.asr_max_initial_timestamp,
                                    )
                            except Exception as error:
                                publish_result("status", f"实时预览识别失败，正式识别继续: {error}", "", None)
                                p_words = []
                            p_text = clean_repetitions(words_to_text(p_words))
                            if p_words and not has_valid_time_span(
                                p_words,
                                max_word_seconds=cfg.max_asr_word_duration_seconds,
                                max_sentence_seconds=cfg.max_asr_sentence_duration_seconds,
                            ):
                                p_text = ""
                            if p_text and is_asr_anomaly(p_text):
                                publish_result("status", "实时识别检测到重复循环，已抑制异常预览", "", None)
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

                    # 超长语音防溢出保护：若说话人连续说话超8.5秒未停顿，在标点/意群处拆分
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
                                    publish_result("live_reset", "", "", None)
                                else:
                                    utterance_chunks = []
                                    last_forced_transcription_end = None
                                    publish_result("english", "", "", None)
                                    stable_preview.reset()
                                    stable_preview_text = ""
                                    publish_result("live_reset", "", "", None)
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
                                    publish_result("live_reset", "", "", None)
                                else:
                                    publish_result("status", "检测到语音片段，等待更多上下文", "", None)
                        last_preview_text = ""

                else:
                    # 静音切片：通过气口检测自然结句
                    if utterance_chunks:
                        # Retain the silence following speech as a tail pad. Quiet
                        # final words can otherwise be classified as silence.
                        if not waiting_for_continuation and utterance_chunks[-1] is not chunk:
                            utterance_chunks.append(chunk)
                        silence_consecutive_chunks += 1
                        silence_duration = silence_consecutive_chunks * (len(chunk.data) / cfg.samplerate)
                        current_duration = sum(len(c.data) for c in utterance_chunks) / cfg.samplerate

                        # 气口自然停顿达标（>=0.8秒）且语音有效（>=0.35秒）
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
                                    publish_result("live_reset", "", "", None)
                                    continue
                                if waiting_for_continuation and pending_words:
                                    if should_hold_after_waiting_silence(
                                        pending_words,
                                        silence_duration,
                                        cfg.vad_pause_seconds * 2,
                                    ):
                                        silence_consecutive_chunks = 0
                                        publish_result("status", "检测到未完成词，延长等待语音", "", None)
                                        continue
                                    utterance_chunks = trim_trailing_silence_chunks(
                                        utterance_chunks, silence_consecutive_chunks
                                    )
                                    silence_consecutive_chunks = 0
                                    publish_result("status", "检测到未完成片段，等待后续语音", "", None)
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
                                    publish_result("live_reset", "", "", None)
                                else:
                                    utterance_chunks = trim_trailing_silence_chunks(
                                        utterance_chunks, silence_consecutive_chunks
                                    )
                                    silence_consecutive_chunks = 0
                                    waiting_for_continuation = True
                                    waiting_since = time.monotonic()
                                    publish_result("status", "检测到语音片段，等待更多上下文", "", None)
                            else:
                                utterance_chunks = trim_trailing_silence_chunks(
                                    utterance_chunks, silence_consecutive_chunks
                                )
                                silence_consecutive_chunks = 0
                                waiting_for_continuation = True
                                waiting_since = time.monotonic()
                                publish_result("status", "暂未识别到清晰语音，等待后续内容", "", None)

        except Exception as error:
            report_thread_error("语音识别", error)
        finally:
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
            asr_done_event.set()
            if stop_event.is_set():
                capture_done_event.set()
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
                ) = correction_queue.get(timeout=0.2)
                except queue.Empty:
                    if capture_done_event.is_set():
                        break
                    continue
                correction_queue.task_done()
                try:
                    # 优先保证实时识别流畅，若有实时音频等待则先让步
                    wait_started = time.monotonic()
                    while (
                        not stop_event.is_set()
                        and audio_queue.qsize() > 0
                        and time.monotonic() - wait_started < cfg.correction_max_audio_wait_seconds
                    ):
                        time.sleep(0.05)

                    asr_prompt = terminology.asr_prompt(
                        prompt, max_terms=cfg.asr_glossary_prompt_terms
                    )
                    correction_started = time.monotonic()
                    with asr_lock:
                        corrected_words = asr.transcribe(
                            highpass(context_audio, cfg.samplerate),
                            initial_prompt=build_asr_prompt(prompt) or asr_prompt or None,
                            hotwords=asr_prompt or None,
                            condition_on_previous_text=cfg.asr_condition_on_previous_text,
                            beam_size=cfg.asr_correction_beam_size,
                            repetition_penalty=cfg.asr_repetition_penalty,
                            max_initial_timestamp=cfg.asr_max_initial_timestamp,
                        )
                    diagnostics.write_event(
                        "asr_correction",
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

                    corrected = terminology.normalize_source(clean_repetitions(corrected))
                    corrected_target_words = (
                        range_words
                        if target_start is not None and target_end is not None and target_end > target_start
                        else corrected_words
                    )
                    corrected_confidences = [
                        word.probability for word in corrected_target_words
                        if word.probability is not None
                    ]
                    with records_lock:
                        original = records[index][0] if index < len(records) else ""
                        expected_version = (
                            record_metadata[index]["version"]
                            if index < len(record_metadata) else None
                        )
                    if (
                        correction_is_usable(original, corrected)
                        and correction_is_confident(
                            original,
                            corrected,
                            original_confidences,
                            corrected_confidences,
                        )
                        and not is_asr_anomaly(corrected)
                    ):
                        changed = False
                        with records_lock:
                            if (
                                index < len(records)
                                and record_metadata[index]["version"] == expected_version
                                and corrected != records[index][0]
                            ):
                                record_metadata[index]["previous_original"] = original
                                record_metadata[index]["previous_translation"] = records[index][1]
                                record_metadata[index]["corrected"] = True
                                records[index][0] = corrected
                                record_metadata[index]["version"] += 1
                                record_metadata[index].pop("asr_anomaly", None)
                                metadata = record_metadata[index].copy()
                                changed = True
                        if changed:
                            with records_lock:
                                for context_index, sentences in enumerate(record_contexts):
                                    record_contexts[context_index] = [
                                        corrected if sentence == original else sentence
                                        for sentence in sentences
                                    ]
                                meeting_context.replace_recent(original, corrected)
                            publish_result(
                                "history_original_update",
                                index,
                                corrected,
                                metadata,
                                version=metadata["version"],
                            )
                            if not enqueue_translation(index):
                                publish_result("status", "修正句翻译任务未能入队", "", None)
                except Exception as error:
                    diagnostics.write_event(
                        "asr_error", stage="correction", index=index, error=repr(error)
                    )
                    publish_result(
                        "status", f"第 {index + 1} 句延迟纠错失败，保留原转写: {error}", "", None
                    )
                finally:
                    with records_lock:
                        correction_pending.discard(index)
        except Exception as error:
            report_thread_error("语音纠错", error)
        finally:
            correction_done_event.set()

    def translation_worker():
        try:
            while (
                not asr_done_event.is_set()
                or not correction_done_event.is_set()
                or not history_translation_queue.empty()
                or not live_translation_queue.empty()
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
                        if (
                            live_state.get("revision") != revision
                            or live_state.get("text") != live_text
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
                            is_current = (
                                live_state.get("revision") == revision
                                and live_state.get("text") == live_text
                            )
                        if is_current:
                            publish_result("live_translation", revision, translated_live, None)
                    except Exception as error:
                        diagnostics.write_event(
                            "translation_error",
                            stage="live",
                            revision=revision,
                            error=repr(error),
                        )
                        publish_result("status", f"实时中文翻译失败: {error}", "", None)
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
                        except Exception:
                            translated = translator.translate(
                                original, quality=cfg.history_translation_quality
                            )
                    else:
                        translated = translator.translate(
                            original, quality=cfg.history_translation_quality
                        )

                    issues = translator.quality_issues(original, translated)
                    if issues:
                        publish_result(
                            "status",
                            f"第 {index + 1} 句翻译质量检查提示: {', '.join(issues)}",
                            "",
                            None,
                        )
                        if cfg.translation_retry_count:
                            try:
                                translated = translator.translate(original, quality=True)
                            except Exception as error:
                                publish_result(
                                    "status",
                                    f"第 {index + 1} 句增强翻译重试失败: {error}",
                                    "",
                                    None,
                                )

                    remaining_issues = translator.quality_issues(original, translated)
                    diagnostics.write_event(
                        "history_translation",
                        index=index,
                        seconds=round(time.monotonic() - history_translation_started, 3),
                        issues=remaining_issues,
                    )
                    if remaining_issues or is_review_candidate(original):
                        with records_lock:
                            if index < len(record_metadata):
                                record_metadata[index]["review"] = True
                                metadata["review"] = True
                        publish_result("history_review", index, "", None)

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
                                f"第 {index + 1} 句翻译耗时 {translator.last_latency_seconds:.1f}s，已输出当前结果",
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
                    diagnostics.write_event(
                        "translation_error",
                        stage="history",
                        index=index,
                        error=repr(error),
                    )
                    publish_result(
                        "status",
                        f"第 {index + 1} 句翻译失败，英文记录已保留: {error}",
                        "",
                        None,
                    )
        except Exception as error:
            report_thread_error("翻译工作线程", error)

    capture_thread = threading.Thread(target=capture_worker, name="audio-capture", daemon=True)
    worker_thread = threading.Thread(target=worker, name="asr-worker", daemon=True)
    correction_thread = threading.Thread(target=correction_worker, name="correction-worker", daemon=True)
    translation_thread = threading.Thread(target=translation_worker, name="translation-worker", daemon=True)
    capture_thread.start()
    worker_thread.start()
    correction_thread.start()
    translation_thread.start()
    capture_started_at = time.monotonic()
    audio_started_at = capture_started_at

    def refresh():
        processed = 0
        while processed < 100:
            try:
                event, first, second, fourth, event_version = result_queue.get_nowait()
            except queue.Empty:
                break
            result_queue.task_done()
            processed += 1
            if event == "english":
                overlay.append_english(first)
                overlay.update_current(first, "")
            elif event == "history_add":
                visible_index = first - overlay.history_offset
                if len(overlay.history) >= cfg.max_overlay_records:
                    overlay.append_history(first, second, "", metadata=fourth)
                elif visible_index < len(overlay.history):
                    overlay.update_history_original(
                        visible_index, second, metadata=fourth, version=event_version
                    )
                elif visible_index == len(overlay.history):
                    overlay.append_history(first, second, "", metadata=fourth)
                if fourth and fourth.get("review"):
                    overlay.set_status("本句可能是识别片段，请复核")
                else:
                    overlay.set_status("正在翻译与识别...")
            elif event == "history_update":
                visible_index = first - overlay.history_offset
                if visible_index >= 0:
                    overlay.update_history(visible_index, second, version=fourth)
                overlay.set_status(f"已翻译第 {first + 1} 句")
            elif event == "history_original_update":
                visible_index = first - overlay.history_offset
                if visible_index >= 0:
                    overlay.update_history_original(
                        visible_index, second, metadata=fourth, version=event_version
                    )
                overlay.set_status(f"已二次修正第 {first + 1} 句转写")
            elif event == "live_translation":
                with live_state_lock:
                    is_current = live_state.get("revision") == first
                if is_current:
                    overlay.update_current(overlay.current_english, second)
            elif event == "live_reset":
                with live_state_lock:
                    live_state["revision"] += 1
                    live_state["text"] = ""
                overlay.update_current("", "")
            elif event == "history_review":
                visible_index = first - overlay.history_offset
                if visible_index >= 0:
                    overlay.mark_history_for_review(visible_index)
                overlay.set_status(f"第 {first + 1} 句可能需要复核")
            elif event == "status":
                overlay.set_status(first)

        try:
            name, error = error_queue.get_nowait()
        except queue.Empty:
            name = None
        if name:
            overlay.set_status(f"{name}线程异常: {error}")
        if not overlay.closed:
            overlay.refresh_after_id = overlay.root.after(100, refresh)

    overlay.refresh_after_id = overlay.root.after(100, refresh)
    try:
        overlay.run()
    finally:
        stop_event.set()
        capture_thread.join()
        capture_ended_at = time.monotonic()
        if getattr(overlay, "refresh_after_id", None):
            try:
                overlay.root.after_cancel(overlay.refresh_after_id)
            except Exception:
                pass
        worker_thread.join()
        correction_thread.join()
        translation_thread.join()
        audio_recorder.close()
        with records_lock:
            saved_records = [tuple(record) for record in records]
            saved_metadata = [metadata.copy() for metadata in record_metadata]

        now_str = session_id
        timestamped_file = f"meeting_transcript_{now_str}.txt"
        timestamped_path = os.path.join(session_dir, timestamped_file)
        content_lines = []
        for index, (original, translated) in enumerate(saved_records):
            metadata = saved_metadata[index]
            timestamps = (
                f"[{Overlay._format_timestamp(metadata['start'])}-"
                f"{Overlay._format_timestamp(metadata['end'])}] "
            )
            review = "[需要复核] " if metadata.get("review") else ""
            corrected = "[已修订] " if metadata.get("corrected") else ""
            content_lines.append(f"{timestamps}{review}{corrected}[EN] {original}\n[ZH] {translated}\n")
            if metadata.get("previous_original"):
                content_lines.append(f"[EN - 修订前] {metadata['previous_original']}\n")
            if metadata.get("previous_translation"):
                content_lines.append(f"[ZH - 修订前] {metadata['previous_translation']}\n")
            content_lines.append("\n")
        full_content = "".join(content_lines)

        with open(timestamped_path, "w", encoding="utf-8") as output:
            output.write(full_content)
        save_history_file()
        stop_capture_warnings(warning_recorder)
        print(
            f"[main] 已保存 {len(saved_records)} 条记录、音频和文本到 {session_dir}。"
        )
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
            transcript_path=os.path.join(session_dir, cfg.history_file),
            audio_path=os.path.join(session_dir, cfg.audio_filename),
        )
        diagnostics.flush_warning_summary()


if __name__ == "__main__":
    main()
