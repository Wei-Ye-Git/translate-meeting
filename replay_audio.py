import argparse
import json
import os
import sys
import time
import wave

import numpy as np

from asr import ASR
from audio_preprocess import highpass
from config import Config
from terminology import Terminology
from text_processing import (
    TimedWord,
    clean_repetitions,
    is_asr_anomaly,
    is_recordable_sentence,
    merge_incomplete_sentences,
    remove_chunk_overlap_words,
    split_timed_sentences,
    words_to_text,
)
from translator import Translator


def iter_wav_chunks(path, chunk_seconds=8.5, overlap_seconds=0.35):
    """Stream mono 16 kHz PCM WAV windows without loading the whole meeting."""
    with wave.open(path, "rb") as source:
        if (
            source.getnchannels() != 1
            or source.getframerate() != 16000
            or source.getsampwidth() != 2
            or source.getcomptype() != "NONE"
        ):
            raise ValueError("回放 WAV 必须是 16 kHz、单声道、16-bit PCM")
        chunk_frames = max(1, int(chunk_seconds * source.getframerate()))
        overlap_frames = min(
            max(0, int(overlap_seconds * source.getframerate())),
            chunk_frames - 1,
        )
        step_frames = max(1, chunk_frames - overlap_frames)
        buffer = np.empty(0, dtype=np.float32)
        next_start = 0
        while True:
            raw = source.readframes(step_frames)
            if not raw:
                break
            new_audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            buffer = np.concatenate((buffer, new_audio)) if len(buffer) else new_audio
            if len(buffer) >= chunk_frames:
                yield next_start / source.getframerate(), buffer[:chunk_frames].copy()
                buffer = buffer[step_frames:]
                next_start += step_frames
        if len(buffer):
            yield next_start / source.getframerate(), buffer


def write_jsonl_exclusive(path, records):
    with open(path, "x", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_replay_hotwords(terminology, context):
    return terminology.asr_prompt(
        " ".join(context[-4:]), include_always=False
    ) or None


def replay(path, output_path, start_seconds=0.0, duration_seconds=None, beam_size=1):
    root = os.path.dirname(os.path.abspath(__file__))
    config = Config()
    terminology = Terminology.from_file(os.path.join(root, config.terminology_file))
    asr = ASR(
        config.whisper_model,
        config.whisper_device,
        config.whisper_compute_type,
        os.path.join(root, "models", "whisper_base_en"),
        cpu_threads=config.whisper_cpu_threads,
    )
    translator = Translator(
        os.path.join(root, config.translate_model_dir),
        config.translate_device,
        terminology,
        intra_threads=config.translate_intra_threads,
        inter_threads=config.translate_inter_threads,
    )
    end_seconds = start_seconds + duration_seconds if duration_seconds else None
    records = []
    previous_words = []
    pending_words = []
    translation_context = []
    started = time.monotonic()
    processed_audio_seconds = 0.0
    pending_asr_seconds = 0.0

    for window_start, audio in iter_wav_chunks(path):
        window_end = window_start + len(audio) / 16000
        if window_end <= start_seconds:
            continue
        if end_seconds is not None and window_start >= end_seconds:
            break
        if window_start < start_seconds:
            trim = int((start_seconds - window_start) * 16000)
            audio = audio[trim:]
            window_start = start_seconds
        if end_seconds is not None and window_start + len(audio) / 16000 > end_seconds:
            audio = audio[:int((end_seconds - window_start) * 16000)]
        if not len(audio):
            continue
        processed_audio_seconds += len(audio) / 16000

        asr_started = time.monotonic()
        words = asr.transcribe(
            highpass(audio, 16000),
            hotwords=build_replay_hotwords(terminology, translation_context),
            beam_size=beam_size,
        )
        asr_seconds = time.monotonic() - asr_started
        pending_asr_seconds += asr_seconds
        timed_words = [
            TimedWord(window_start + word.start, window_start + word.end, word.text)
            for word in words
        ]
        if previous_words and timed_words:
            timed_words = remove_chunk_overlap_words(previous_words[-24:], timed_words)
        if timed_words:
            previous_words.extend(timed_words)
            previous_words = previous_words[-24:]
            pending_words.extend(timed_words)
        completed, remainder = split_timed_sentences(pending_words)
        groups = merge_incomplete_sentences(completed + ([remainder] if remainder else []))
        pending_words = groups[-1] if remainder and groups else []
        finalized_groups = groups[:-1] if remainder else groups
        for group_index, group in enumerate(finalized_groups):
            english = terminology.normalize_source(clean_repetitions(words_to_text(group)))
            if not english or is_asr_anomaly(english) or not is_recordable_sentence(english):
                continue
            translation_started = time.monotonic()
            chinese = translator.translate_with_context(
                english, translation_context[-2:], quality=False
            )
            translation_seconds = time.monotonic() - translation_started
            translation_context.append(english)
            records.append({
                "event": "sentence",
                "start": round(group[0].start, 3),
                "end": round(group[-1].end, 3),
                "en": english,
                "zh": chinese,
                "asr_seconds": round(pending_asr_seconds if group_index == 0 else 0, 3),
                "translation_seconds": round(translation_seconds, 3),
            })
        if finalized_groups:
            pending_asr_seconds = 0.0

    if pending_words:
        english = terminology.normalize_source(clean_repetitions(words_to_text(pending_words)))
        if english and not is_asr_anomaly(english) and is_recordable_sentence(english):
            translation_started = time.monotonic()
            chinese = translator.translate_with_context(
                english, translation_context[-2:], quality=False
            )
            translation_context.append(english)
            records.append({
                "event": "sentence",
                "start": round(pending_words[0].start, 3),
                "end": round(pending_words[-1].end, 3),
                "en": english,
                "zh": chinese,
                "asr_seconds": round(pending_asr_seconds, 3),
                "translation_seconds": round(time.monotonic() - translation_started, 3),
            })

    records.insert(0, {
        "event": "replay_summary",
        "input": os.path.abspath(path),
        "start_seconds": start_seconds,
        "audio_seconds": round(processed_audio_seconds, 3),
        "processing_seconds": round(time.monotonic() - started, 3),
        "sentences": len(records),
        "beam_size": beam_size,
    })
    write_jsonl_exclusive(output_path, records)
    return len(records) - 1


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="用现有本地模型离线复测会议 WAV 片段")
    parser.add_argument("wav", help="16 kHz mono 16-bit PCM WAV")
    parser.add_argument("--output", required=True, help="新 JSONL 输出路径（不得已存在）")
    parser.add_argument("--start", type=float, default=0.0, help="开始秒数")
    parser.add_argument("--duration", type=float, help="复测秒数；省略则处理至文件结束")
    parser.add_argument("--beam-size", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    count = replay(args.wav, args.output, args.start, args.duration, args.beam_size)
    print(f"已回放 {count} 条字幕；结果写入 {args.output}")


if __name__ == "__main__":
    main()
