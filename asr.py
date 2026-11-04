import os
import re
import time

import numpy as np
from faster_whisper import WhisperModel

from text_processing import TimedWord

PROMPT_CHAR_BUDGET = 224


def build_asr_prompt(previous_text, hotwords=None):
    del hotwords
    text = (previous_text or "").strip()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    return text[-PROMPT_CHAR_BUDGET:]


class ASR:
    def __init__(self, model_size="base.en", device="cpu", compute_type="int8",
                 local_dir=None, cpu_threads=None):
        if not local_dir or not os.path.isfile(os.path.join(local_dir, "model.bin")):
            raise FileNotFoundError(
                f"未找到本地 Whisper 模型: {local_dir}。请先运行 setup.bat 下载模型。"
            )
        model_name = local_dir
        if cpu_threads is None:
            cpu_threads = min(8, max(2, (os.cpu_count() or 4) // 2))
        print(f"[asr] 加载 Whisper: {model_name} ({device}/{compute_type}, threads={cpu_threads})")
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        self.last_latency_seconds = 0.0

    def transcribe(
        self,
        audio_np,
        initial_prompt=None,
        hotwords=None,
        condition_on_previous_text=False,
        beam_size=1,
        word_timestamps=True,
        repetition_penalty=1.1,
        max_initial_timestamp=0.8,
    ):
        audio_np = np.asarray(audio_np, dtype=np.float32)
        if len(audio_np) == 0:
            return []
        started = time.monotonic()
        segments, _ = self.model.transcribe(
            audio_np,
            language="en",
            beam_size=beam_size,
            best_of=beam_size,
            temperature=0,
            # The outer energy gate already controls utterance boundaries. A
            # second VAD here can remove quiet first/last words.
            vad_filter=False,
            condition_on_previous_text=condition_on_previous_text,
            word_timestamps=word_timestamps,
            initial_prompt=build_asr_prompt(initial_prompt, hotwords),
            hotwords=hotwords or None,
            repetition_penalty=repetition_penalty,
            max_initial_timestamp=max_initial_timestamp,
            no_speech_threshold=0.4,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 400,
            },
        )
        words = []
        for segment in segments:
            if segment.words:
                words.extend(
                    TimedWord(
                        word.start,
                        word.end,
                        word.word.strip(),
                        getattr(word, "probability", None),
                    )
                    for word in segment.words
                    if word.word.strip()
                )
            elif segment.text.strip():
                words.append(TimedWord(segment.start, segment.end, segment.text.strip()))
        self.last_latency_seconds = time.monotonic() - started
        return words
