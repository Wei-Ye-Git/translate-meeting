import os
from dataclasses import dataclass


@dataclass
class Config:
    mode: str = "complete"
    low_latency_queue_chunks: int = 1
    loopback_device_name: str = ""
    samplerate: int = 16000
    chunk_seconds: float = 0.5
    vad_pause_seconds: float = 0.8
    max_utterance_seconds: float = 7.5
    voice_energy_threshold: float = 0.0012
    vad_subchunk_seconds: float = 0.1
    vad_subchunk_threshold_ratio: float = 0.65
    audio_boundary_padding_seconds: float = 0.35
    realtime_translation_min_words: int = 2
    max_asr_word_duration_seconds: float = 5.0
    max_asr_sentence_duration_seconds: float = 15.0
    whisper_model: str = "base.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_cpu_threads: int = 6
    translate_model_dir: str = "models/opus_mt_en_zh"
    translate_device: str = "cpu"
    translate_intra_threads: int = 2
    translate_inter_threads: int = 1
    terminology_file: str = "terminology.json"
    default_terminology_file: str = "glossaries/general.json"
    overlay_alpha: float = 0.92
    font_size: int = 18
    history_file: str = "meeting_transcript.txt"
    session_directory_prefix: str = "meeting_"
    audio_filename: str = "meeting_audio.wav"
    history_flush_seconds: float = 6.0
    correction_context_before_seconds: float = 3.0
    correction_context_after_seconds: float = 3.0
    correction_recent_audio_seconds: float = 30.0
    asr_beam_size: int = 4
    asr_condition_on_previous_text: bool = False
    asr_correction_beam_size: int = 4
    asr_preview_interval_seconds: float = 1.5
    asr_repetition_penalty: float = 1.1
    asr_max_initial_timestamp: float = 0.8
    history_translation_quality: bool = True
    history_context_window: int = 4
    max_audio_queue_chunks: int = 1200
    max_result_queue_items: int = 500
    shutdown_join_timeout_seconds: float = 5.0
    preview_max_pending_audio_chunks: int = 1
    correction_max_audio_wait_seconds: float = 1.0
    context_max_sentences: int = 4
    context_max_chars: int = 480
    translation_retry_count: int = 1
    translation_context_enabled: bool = True
    audio_backlog_warning_seconds: float = 8.0
    asr_glossary_prompt_terms: int = 30
    asr_timeout_seconds: float = 30.0
    translation_timeout_seconds: float = 20.0
    live_translation_quality: bool = False
    realtime_copilot_enabled: bool = False
    realtime_copilot_model: str = ""
    realtime_ai_cli: str = "opencode"
    realtime_copilot_deadline_seconds: float = 60.0
    realtime_copilot_start_delay_seconds: float = 20.0
    realtime_copilot_interval_seconds: float = 30.0
    realtime_copilot_cumulative_rounds: int = 1
    realtime_copilot_context_chars: int = 1200
    realtime_copilot_max_retries: int = 1
    realtime_copilot_max_input_chars: int = 24000
    realtime_copilot_context_max_records: int = 30
    realtime_copilot_max_target_records: int = 15
    max_overlay_records: int = 5000
    max_stored_records: int = 20000

    def __post_init__(self):
        environment_mode = os.environ.get("MEETING_MODE")
        if environment_mode in ("complete", "low_latency"):
            self.mode = environment_mode
        copilot_enabled = os.environ.get("MEETING_COPILOT_ENABLED")
        if copilot_enabled is not None:
            self.realtime_copilot_enabled = copilot_enabled.strip().lower() in {
                "1", "true", "yes", "on"
            }
        self.realtime_copilot_model = os.environ.get(
            "MEETING_COPILOT_MODEL", self.realtime_copilot_model
        ).strip()
        self.realtime_ai_cli = os.environ.get(
            "MEETING_AI_CLI", self.realtime_ai_cli
        ).strip()
        if self.mode == "low_latency":
            self.asr_beam_size = 1
            self.asr_correction_beam_size = 1
            self.asr_preview_interval_seconds = 0.8
            self.asr_condition_on_previous_text = False

    def configure_ai_timing(self):
        try:
            self.realtime_copilot_deadline_seconds = max(
                10.0,
                float(os.environ.get(
                    "MEETING_AI_TIMEOUT_SECONDS",
                    self.realtime_copilot_deadline_seconds,
                )),
            )
            self.realtime_copilot_start_delay_seconds = max(
                0.0,
                float(os.environ.get(
                    "MEETING_AI_START_DELAY_SECONDS",
                    self.realtime_copilot_start_delay_seconds,
                )),
            )
            self.realtime_copilot_interval_seconds = max(
                1.0,
                float(os.environ.get(
                    "MEETING_AI_INTERVAL_SECONDS",
                    self.realtime_copilot_interval_seconds,
                )),
            )
        except ValueError:
            raise ValueError("Meeting AI timeout, start delay, and interval must be numeric")
