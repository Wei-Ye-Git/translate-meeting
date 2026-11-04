import queue
import threading
import time
import warnings
from dataclasses import dataclass

import numpy as np
import soundcard as sc


def _safe_print(message):
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="backslashreplace").decode("ascii"))


class AudioWarningRecorder:
    """Count repeated capture warnings without flooding the console log."""

    def __init__(self, warning_callback):
        self.warning_callback = warning_callback
        self.original_showwarning = warnings.showwarning
        self.original_filters = list(warnings.filters)
        self.counts = {}
        self.details = {}
        self._lock = threading.Lock()

    def __call__(self, message, category, filename, lineno, file=None, line=None):
        key = (category.__name__, str(message), filename, lineno)
        with self._lock:
            self.counts[key] = self.counts.get(key, 0) + 1
            self.details[key] = (category.__name__, str(message), filename, lineno)

    def flush(self):
        with self._lock:
            items = [(self.details[key], count) for key, count in self.counts.items()]
            self.counts.clear()
            self.details.clear()
        for (category, message, filename, lineno), count in items:
            self.warning_callback(category, message, filename, lineno, count)


def start_capture_warnings(warning_callback):
    recorder = AudioWarningRecorder(warning_callback)
    warnings.showwarning = recorder
    warnings.simplefilter("always")
    return recorder


def stop_capture_warnings(recorder):
    recorder.flush()
    warnings.showwarning = recorder.original_showwarning
    warnings.filters[:] = recorder.original_filters


def enqueue_audio_chunk(
    out_queue,
    chunk,
    config,
    stop_event=None,
    gap_callback=None,
    consumer_done_event=None,
):
    if config.mode == "complete":
        if isinstance(out_queue, list):
            out_queue.append(chunk)
        else:
            while True:
                try:
                    out_queue.put(chunk, timeout=0.1)
                    break
                except queue.Full:
                    if (
                        stop_event is not None
                        and stop_event.is_set()
                        and consumer_done_event is not None
                        and consumer_done_event.is_set()
                    ):
                        return False
        return False
    if isinstance(out_queue, list):
        if len(out_queue) >= config.low_latency_queue_chunks:
            dropped = out_queue.pop(0)
            chunk.gap_before = True
            if out_queue:
                out_queue[0].gap_before = True
            out_queue.append(chunk)
            if gap_callback is not None:
                gap_callback(dropped.end_frame)
            return True
        out_queue.append(chunk)
        return False
    try:
        out_queue.put_nowait(chunk)
        return False
    except queue.Full:
        dropped = None
        try:
            dropped = out_queue.get_nowait()
        except queue.Empty:
            pass
        chunk.gap_before = True
        with out_queue.mutex:
            if out_queue.queue:
                out_queue.queue[0].gap_before = True
        out_queue.put_nowait(chunk)
        if gap_callback is not None:
            gap_callback(dropped.end_frame if dropped is not None else chunk.start_frame)
        return True


def queued_audio_seconds(out_queue, chunk_seconds):
    size = out_queue.qsize() if hasattr(out_queue, "qsize") else len(out_queue)
    return size * chunk_seconds


@dataclass
class AudioChunk:
    start_frame: int
    data: np.ndarray
    gap_before: bool = False

    @property
    def end_frame(self):
        return self.start_frame + len(self.data)


_VIRTUAL_KEYWORDS = (
    "todesk", "virtual", "stereo mix", "what u hear", "voicemeeter",
    "vb-audio", "remote audio", "cable", "blackhole",
)


def list_loopback_devices():
    try:
        return [m for m in sc.all_microphones(include_loopback=True) if m.isloopback]
    except Exception:
        return []


def _is_virtual(name: str) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in _VIRTUAL_KEYWORDS)


def pick_loopback_device(name: str = ""):
    loops = list_loopback_devices()
    if not loops:
        raise RuntimeError("未找到扬声器回环设备，请在 Windows 中启用扬声器回环设备。")
    if name:
        for device in loops:
            if name.lower() in device.name.lower():
                return device
        print(f"[audio] 未找到配置的设备 '{name}'，改用自动选择。")
    try:
        speaker_name = sc.default_speaker().name
        for device in loops:
            if speaker_name in device.name:
                return device
    except Exception:
        pass
    real = [device for device in loops if not _is_virtual(device.name)]
    selected = real[0] if real else loops[0]
    print(f"[audio] 使用设备: {selected.name}")
    return selected


def capture_loop(device, samplerate, chunk_frames, out_queue: queue.Queue,
                 stop_event: threading.Event, config, audio_sink=None,
                 gap_callback=None, diagnostic_callback=None, capture_stats=None,
                 audio_timeline_sink=None, consumer_done_event=None):
    next_frame = 0
    current_device = device
    last_audio_time = time.monotonic()
    while not stop_event.is_set():
        try:
            with current_device.recorder(samplerate=samplerate) as input_recorder:
                while not stop_event.is_set():
                    data = input_recorder.record(numframes=chunk_frames)
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    audio = data.astype(np.float32)
                    last_audio_time = time.monotonic()
                    if audio_sink is not None:
                        audio_sink.write(audio)
                    if capture_stats is not None:
                        capture_stats["chunks"] += 1
                        capture_stats["frames"] += len(audio)
                        capture_stats["next_frame"] = next_frame + len(audio)
                    was_dropped = enqueue_audio_chunk(
                        out_queue,
                        AudioChunk(next_frame, audio),
                        config,
                        stop_event,
                        gap_callback=gap_callback or (lambda frame: print(
                            f"[audio] 丢弃旧音频块，缺口从帧 {frame} 开始"
                        )),
                        consumer_done_event=consumer_done_event,
                    )
                    next_frame += len(audio)
                    if was_dropped:
                        continue
        except Exception as error:
            if stop_event.is_set():
                break
            _safe_print(f"[audio] 录音异常: {error}，尝试自动重新连接扬声器设备...")
            if diagnostic_callback is not None:
                diagnostic_callback("audio_device_reconnect", error=repr(error))
            time.sleep(1.0)
            try:
                current_device = pick_loopback_device(config.loopback_device_name)
            except Exception:
                pass
            missing_frames = max(
                0, int((time.monotonic() - last_audio_time) * samplerate)
            )
            next_frame += missing_frames
            if capture_stats is not None:
                capture_stats["reconnects"] += 1
                capture_stats["estimated_missing_frames"] += missing_frames
            if missing_frames and audio_sink is not None:
                write_silence = getattr(audio_sink, "write_silence", None)
                if write_silence is not None:
                    write_silence(missing_frames)
            if missing_frames and audio_timeline_sink is not None:
                timeline_sink = audio_timeline_sink
                if timeline_sink is not audio_sink:
                    write_silence = getattr(timeline_sink, "write_silence", None)
                    if write_silence is not None:
                        write_silence(missing_frames)
            if missing_frames and diagnostic_callback is not None:
                diagnostic_callback(
                    "audio_timeline_gap",
                    estimated_missing_frames=missing_frames,
                    estimated_missing_seconds=round(missing_frames / samplerate, 3),
                    next_frame=next_frame,
                )
            last_audio_time = time.monotonic()


def extract_audio_range(chunks, start_frame, end_frame):
    """Return the samples belonging to an absolute frame range."""
    if end_frame <= start_frame:
        return np.empty(0, dtype=np.float32)
    parts = []
    expected_frame = start_frame
    saw_overlap = False
    for chunk in sorted(chunks, key=lambda item: item.start_frame):
        if expected_frame >= end_frame:
            break
        if chunk.end_frame <= start_frame or chunk.start_frame >= end_frame:
            continue
        if chunk.start_frame > expected_frame:
            gap_end = min(chunk.start_frame, end_frame)
            parts.append(np.zeros(gap_end - expected_frame, dtype=np.float32))
            expected_frame = gap_end
        overlap_start = max(start_frame, chunk.start_frame)
        overlap_end = min(end_frame, chunk.end_frame)
        if overlap_start >= overlap_end:
            continue
        start = overlap_start - chunk.start_frame
        end = overlap_end - chunk.start_frame
        parts.append(chunk.data[start:end])
        saw_overlap = True
        expected_frame = max(expected_frame, overlap_end)
    if not saw_overlap:
        return np.empty(0, dtype=np.float32)
    if expected_frame < end_frame:
        parts.append(np.zeros(end_frame - expected_frame, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32, copy=False)
