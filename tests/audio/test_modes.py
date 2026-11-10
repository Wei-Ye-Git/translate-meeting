"""Audio queue and operating mode tests."""

import unittest
import queue
import threading

import numpy as np

from audio_capture import AudioChunk, enqueue_audio_chunk, queued_audio_seconds
from config import Config


class ModeTests(unittest.TestCase):
    def test_complete_mode_keeps_all_audio_chunks(self):
        config = Config(mode="complete")
        chunks = []
        for number in range(3):
            enqueue_audio_chunk(
                chunks,
                AudioChunk(number * 4, np.full(4, number, dtype=np.float32)),
                config,
            )

        self.assertEqual([chunk.start_frame for chunk in chunks], [0, 4, 8])

    def test_low_latency_mode_discards_oldest_chunk_when_full(self):
        config = Config(mode="low_latency", low_latency_queue_chunks=2)
        chunks = []
        for number in range(3):
            enqueue_audio_chunk(
                chunks,
                AudioChunk(number * 4, np.full(4, number, dtype=np.float32)),
                config,
            )

        self.assertEqual([chunk.start_frame for chunk in chunks], [4, 8])

    def test_config_can_be_selected_by_environment(self):
        import os

        previous = os.environ.get("MEETING_MODE")
        try:
            os.environ["MEETING_MODE"] = "low_latency"
            self.assertEqual(Config().mode, "low_latency")
        finally:
            if previous is None:
                os.environ.pop("MEETING_MODE", None)
            else:
                os.environ["MEETING_MODE"] = previous

    def test_complete_mode_uses_boundary_padding_and_lower_voice_gate(self):
        config = Config(mode="complete")
        self.assertEqual(config.audio_boundary_padding_seconds, 0.35)
        self.assertLess(config.voice_energy_threshold, 0.002)
        self.assertEqual(config.whisper_cpu_threads, 6)
        self.assertEqual(config.vad_subchunk_seconds, 0.1)
        self.assertGreater(config.max_audio_queue_chunks, 0)
        self.assertGreaterEqual(config.max_audio_queue_chunks * config.chunk_seconds, 300)

    def test_complete_mode_reduces_repeat_preview_and_keeps_light_correction(self):
        complete = Config(mode="complete")
        low_latency = Config(mode="low_latency")

        self.assertEqual(complete.asr_preview_interval_seconds, 1.5)
        self.assertEqual(complete.asr_correction_beam_size, 2)
        self.assertEqual(low_latency.asr_correction_beam_size, 1)
        self.assertEqual(low_latency.asr_preview_interval_seconds, 0.8)

    def test_backlog_controls_are_bounded_and_context_is_lightweight(self):
        config = Config()

        self.assertEqual(config.context_max_sentences, 4)
        self.assertLessEqual(config.context_max_chars, 512)
        self.assertGreater(config.audio_backlog_warning_seconds, 0)
        self.assertEqual(config.translation_retry_count, 1)
        self.assertLessEqual(config.asr_glossary_prompt_terms, 30)
        self.assertGreater(config.asr_timeout_seconds, 0)
        self.assertGreater(config.translation_timeout_seconds, 0)
        self.assertLessEqual(config.max_overlay_records, 10000)
        self.assertGreaterEqual(config.max_stored_records, config.max_overlay_records)

    def test_audio_backlog_is_reported_as_queued_seconds(self):
        config = Config(mode="low_latency", low_latency_queue_chunks=2)
        chunks = []
        enqueue_audio_chunk(chunks, AudioChunk(0, np.zeros(8000)), config)
        enqueue_audio_chunk(chunks, AudioChunk(8000, np.zeros(8000)), config)

        self.assertEqual(queued_audio_seconds(chunks, 0.5), 1.0)

    def test_low_latency_drop_emits_explicit_gap_marker(self):
        config = Config(mode="low_latency", low_latency_queue_chunks=1)
        audio_queue = queue.Queue(maxsize=1)
        audio_queue.put(AudioChunk(0, np.ones(8, dtype=np.float32)))
        markers = []

        dropped = enqueue_audio_chunk(
            audio_queue,
            AudioChunk(8, np.ones(8, dtype=np.float32)),
            config,
            gap_callback=markers.append,
        )

        self.assertTrue(dropped)
        self.assertEqual(markers, [8])
        retained = audio_queue.get_nowait()
        self.assertEqual(retained.start_frame, 8)
        self.assertTrue(retained.gap_before)

    def test_gap_marker_is_attached_to_the_first_retained_chunk(self):
        config = Config(mode="low_latency", low_latency_queue_chunks=2)
        audio_queue = queue.Queue(maxsize=2)
        audio_queue.put(AudioChunk(0, np.ones(8, dtype=np.float32)))
        audio_queue.put(AudioChunk(8, np.ones(8, dtype=np.float32)))

        enqueue_audio_chunk(
            audio_queue, AudioChunk(16, np.ones(8, dtype=np.float32)), config
        )

        first_retained = audio_queue.get_nowait()
        self.assertEqual(first_retained.start_frame, 8)
        self.assertTrue(first_retained.gap_before)

    def test_complete_mode_stops_waiting_for_queue_space_after_consumer_finishes(self):
        config = Config(mode="complete")
        audio_queue = queue.Queue(maxsize=1)
        audio_queue.put(AudioChunk(0, np.ones(8, dtype=np.float32)))
        stop_event = threading.Event()
        stop_event.set()
        consumer_done_event = threading.Event()
        consumer_done_event.set()

        dropped = enqueue_audio_chunk(
            audio_queue,
            AudioChunk(8, np.ones(8, dtype=np.float32)),
            config,
            stop_event=stop_event,
            consumer_done_event=consumer_done_event,
        )

        self.assertFalse(dropped)
        self.assertEqual(audio_queue.qsize(), 1)


if __name__ == "__main__":
    unittest.main()
