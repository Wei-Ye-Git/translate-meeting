"""Session audio and transcript persistence tests."""

import os
import tempfile
import threading
import time
import unittest
import wave

import numpy as np

from audio_capture import capture_loop
from config import Config
from session_archive import AudioRecorder, create_session_directory


class SessionArchiveTests(unittest.TestCase):
    def test_creates_directory_named_with_session_start_time(self):
        with tempfile.TemporaryDirectory() as root:
            path = create_session_directory(root, "20261010_093015")

            self.assertEqual(os.path.basename(path), "meeting_20261010_093015")
            self.assertTrue(os.path.isdir(path))

    def test_writes_audio_chunks_as_mono_pcm16_wav(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "meeting.wav")
            recorder = AudioRecorder(path, samplerate=16000)
            recorder.write(np.array([0.0, 0.5, -0.5], dtype=np.float32))
            recorder.close()

            with wave.open(path, "rb") as source:
                self.assertEqual(source.getnchannels(), 1)
                self.assertEqual(source.getframerate(), 16000)
                self.assertEqual(source.getsampwidth(), 2)
                self.assertEqual(source.getnframes(), 3)
                samples = np.frombuffer(source.readframes(3), dtype="<i2")

            self.assertEqual(samples.tolist(), [0, 16383, -16383])

    def test_writes_silence_and_counts_timeline_frames(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "meeting.wav")
            recorder = AudioRecorder(path, samplerate=16000)
            recorder.write(np.array([0.5], dtype=np.float32))
            recorder.write_silence(3)
            recorder.close()

            with wave.open(path, "rb") as source:
                samples = np.frombuffer(source.readframes(4), dtype="<i2")

        self.assertEqual(recorder.frames_written, 4)
        self.assertEqual(samples.tolist(), [16383, 0, 0, 0])

    def test_keeps_full_captured_audio_when_low_latency_asr_drops_queue_chunks(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "meeting.wav")
            audio_writer = AudioRecorder(path, samplerate=16000)
            stop_event = threading.Event()
            chunks = [
                np.full(4, 0.1, dtype=np.float32),
                np.full(4, 0.2, dtype=np.float32),
                np.full(4, 0.3, dtype=np.float32),
            ]

            class FakeRecorder:
                def __init__(self):
                    self.index = 0

                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    return False

                def record(self, numframes):
                    audio = chunks[self.index]
                    self.index += 1
                    if self.index > len(chunks):
                        raise RuntimeError("finished")
                    if self.index == len(chunks):
                        stop_event.set()
                    return audio

            class FakeDevice:
                def recorder(self, samplerate):
                    return FakeRecorder()

            class RecorderProxy:
                def write(self, audio):
                    audio_writer.write(audio)

                def close(self):
                    audio_writer.close()

                def write_silence(self, frames):
                    audio_writer.write_silence(frames)

            import queue
            audio_queue = queue.Queue(maxsize=1)

            capture_loop(
                FakeDevice(), 16000, 4, audio_queue, stop_event,
                Config(mode="low_latency"), audio_sink=RecorderProxy(),
                gap_callback=lambda frame: None,
            )
            audio_writer.close()

            with wave.open(path, "rb") as source:
                self.assertEqual(source.getnframes(), 12)

    def test_reconnect_pads_wav_and_reports_estimated_timeline_gap(self):
        import queue
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "meeting.wav")
            recorder = AudioRecorder(path, samplerate=16000)
            stop_event = threading.Event()
            events = []

            class FirstRecorder:
                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    return False

                def record(self, numframes):
                    if not hasattr(self, "returned_audio"):
                        self.returned_audio = True
                        return np.ones(8, dtype=np.float32)
                    time.sleep(0.01)
                    raise RuntimeError("device disconnected")

            class SecondRecorder:
                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    return False

                def record(self, numframes):
                    stop_event.set()
                    return np.full(8, 0.25, dtype=np.float32)

            class Device:
                def __init__(self, input_recorder):
                    self.input_recorder = input_recorder

                def recorder(self, samplerate):
                    return self.input_recorder

            stats = {
                "chunks": 0,
                "frames": 0,
                "reconnects": 0,
                "estimated_missing_frames": 0,
                "next_frame": 0,
            }
            audio_queue = queue.Queue()

            with patch("audio_capture.pick_loopback_device", return_value=Device(SecondRecorder())):
                capture_loop(
                    Device(FirstRecorder()),
                    16000,
                    8,
                    audio_queue,
                    stop_event,
                    Config(mode="low_latency"),
                    audio_sink=recorder,
                    audio_timeline_sink=recorder,
                    diagnostic_callback=lambda event, **fields: events.append((event, fields)),
                    capture_stats=stats,
                )
            recorder.close()

            with wave.open(path, "rb") as source:
                written_frames = source.getnframes()
                samples = np.frombuffer(source.readframes(written_frames), dtype="<i2")

        self.assertEqual(stats["chunks"], 2)
        self.assertEqual(stats["frames"], 16)
        self.assertEqual(stats["reconnects"], 1)
        self.assertGreater(stats["estimated_missing_frames"], 0)
        self.assertEqual(stats["next_frame"], written_frames)
        self.assertEqual(written_frames, recorder.frames_written)
        self.assertEqual(samples[8], 0)
        self.assertTrue(any(event == "audio_timeline_gap" for event, _ in events))


if __name__ == "__main__":
    unittest.main()
