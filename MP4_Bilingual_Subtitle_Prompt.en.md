# Reusable Prompt: Create English–Chinese Subtitles from an MP4

Replace the placeholders before use. Leave unavailable optional inputs blank. Example paths, if supplied separately, are not facts about this machine. Never invent inputs, measurements, tool capabilities, or human-review results.

## Prompt

Create an English transcript and Simplified Chinese bilingual subtitles for the specified MP4. Work in stages. The audio extracted from this MP4 is the primary evidence for what was said; ASR and timestamps are candidates only. Do not label the deliverables `final` until the human-review gate below has actually been completed.

### Project inputs

- MP4: `[full path]`
- Transcript, English script, or Chinese script: `[paths, if any]`
- Glossary, speaker information, or revision notes: `[paths, if any]`
- FFmpeg/ffprobe: `[full paths, or executable names if on PATH]`
- Local Whisper/faster-whisper model: `[path or model name, if available]`
- Player: `[player name]`
- Output directory: `[path; default: subtitle_work beside the MP4]`
- Optional bilingual report: `[yes/no; default yes]`
- Burned-in subtitle preview: `[yes/no; default no]`

### Required deliverables

Create a separate, uniquely named output for this source. Never overwrite an existing file. If an output path exists, choose a new versioned name or stop and report the conflict.

1. `*_en_transcript.md`: corrected English verbatim transcript, without SRT numbers or timestamps. Preserve meaningful repetition, interaction, and speech. Mark unverifiable words `[unclear]` and record their timestamps in the review report.
2. `*_en.srt`: English-only subtitles with exactly the same cue numbers, timings, and English text as the bilingual file.
3. `*_en-zh.srt`: bilingual subtitles. Each cue contains English (at most two lines), followed by a line break and its Simplified Chinese translation. **Do not put a blank line inside a cue; use blank lines only between cues.** Both languages share the same cue number and timing. Preserve negation, conditions, causality, numbers, names, technical claims, and audience questions.
4. `*_segmentation_review.md`: cue-level segmentation risks, automated warnings, human corrections, and unresolved exceptions. Keep cue-level segmentation details here as the single source of truth.
5. `*_asr_words.jsonl`, if supported: one JSON object per line with `start`, `end`, and `text`; include `confidence` and `speaker` only when the tool provides them. Record the model and run parameters. Never infer unavailable fields.
6. `*_review.md`: sources and coverage, tools and commands, actual human-review scope, unresolved issues, and paths to deliverables. Refer to the segmentation review instead of duplicating its cue-level details. Separate transcript/subtitles from summaries or analysis; do not present unverified information as a speaker's words.
7. `*_report.md`, only if requested: corrected English transcript, faithful Chinese translation, and meeting summary in separate sections. Label any technical extrapolation `Supplementary analysis (not in the meeting)` and distinguish facts, inferences, and open questions.

If an input or tool is unavailable, complete independent deliverables where possible, report omissions and reasons, and do not fabricate timestamps or claim a complete delivery.

### Evidence and editing rules

- Run ASR on audio extracted from this MP4. Use supplied scripts, translations, and glossaries only as aids for cross-checking; when they conflict with audible speech, follow the audio. Mark uncertain content `[unclear]`; never reconstruct speech from context alone.
- Cover all intelligible presenter and audience speech, including questions and interruptions. Record unintelligible sections and timestamps rather than silently omitting them. Do not describe ordinary non-speech sounds unless accessibility captions are requested.
- Keep verbatim transcription separate from edited prose. Correct supported ASR errors, punctuation, and segmentation while preserving meaningful spoken features. Do not turn summaries, edits, or technical additions into verbatim speech. For any meaning-preserving subtitle condensation, retain the original, revised text, and reason for traceability.
- Before translating, create a project glossary from verified user-provided terms or confirmed content. Keep translations consistent. Flag ambiguity instead of silently deciding it.
- Lock English cues before translating. Split by syntax, meaning, speaker changes, and readability—not fixed word counts, arbitrary ASR pauses, or Chinese length. Each Chinese cue must match its locked English cue and remain visually readable. If Chinese is too long, first make a faithful concise translation; if still unreadable, revise both languages together.
- About 20 English characters per second and 7 seconds per cue are readability references, not hard limits. Meaning, synchronization, and actual readability take priority. Review cues exceeding those references; re-split only when a valid semantic boundary preserves meaning and timing, otherwise keep the cue and document why. The character-rate reference applies to English text only, not total bilingual reading speed.
- Cue boundaries should follow audible speech. Target subtitle onset at the first audible speech and subtitle end 0.200 seconds after the final sound, but do not force these targets when speech overlap, the next cue, readability, or measurement precision conflicts. Record exceptions.
- Preserve genuine overlapping speech; do not shift onsets just to remove overlaps. Check gaps under 0.100 seconds for possible merging only when meaning, speaker continuity, and readability permit it. Review onsets differing from the first audible speech by more than 0.300 seconds and correct them or document why not.
- Confirm audio/video start offsets and state the time base used. Use SRT timestamps in `HH:MM:SS,mmm --> HH:MM:SS,mmm` format. Report only the precision supported by the available tools/player.
- Do not modify or overwrite the MP4, source documents, existing subtitles, or any existing candidate/report. Do not open or read pre-existing subtitle files; if a target exists, record only its name and existence. You may inspect files generated in this run. Use new paths and non-overwriting commands (for FFmpeg, use `-n`).
- Do not claim support for languages, diarization, confidence values, or precision unless verified for the actual tool/model. If ASR, alignment, FFmpeg, an audio track, or a player is unavailable, report the attempted commands and results, pause dependent steps, and keep the result a candidate or awaiting human review.

### Workflow

1. **Preflight:** Confirm the MP4 is readable. Inspect audio tracks, likely target language, duration, start offset, sample rate, channels, video frame rate, and dimensions. Record actual tool versions and results. Check that the selected model loads successfully; file presence alone is not proof. List supplied reference materials and their actual use/trust level. Do not inspect pre-existing subtitle files. If multiple audio tracks make the target unclear, ask before choosing. Report what can proceed and what must pause.
2. **Extract audio:** Once the target track is confirmed, create the output directory if needed and extract a new PCM WAV. For example, adapt paths and the audio-stream index:

   ```powershell
   & "[ffmpeg or full path]" -hide_banner -n -i "[MP4 path]" -map 0:a:[stream index] -vn -c:a pcm_s16le -ar 16000 -ac 1 "[new output path]\audio_from_mp4_16k_mono.wav"
   ```

   Verify WAV sample rate, channels, bit depth, and duration using ffprobe or its header; compare duration with the source and report any relevant difference. Explain that PCM extraction does not restore losses already present in a lossy source. Use this verified WAV for ASR/alignment unless the user explicitly changes the source requirement. Use actual playback of the MP4 for human listening review.
3. **Transcribe and verify:** Run available local Whisper/faster-whisper in English transcription mode, not translation mode, with word timestamps if supported. Save raw results and actual parameters. Correct the transcript only with audio or other trustworthy evidence. Check supplied materials against the audio; prioritize names, technical terms, numbers, dates, percentages, negation, conditions, causality, missing audience speech, and translation omissions. Remove ASR-created repetition but preserve genuine repetition. Record unclear sections and any condensation.
4. **Segment English:** Draft, review, then lock English cues. Do not split inside grammatical units such as determiners and nouns, names, numbers and units, negation and verbs, phrasal verbs, prepositions and objects, or conjunctions and their dependent phrases. Avoid cues ending in unfinished function words or containing isolated one- or two-word fragments unless the speech forms a complete response or a real speaker change/pause requires it. Each cue should communicate its core meaning on its own and contain at most two English lines. Do not force a split merely to meet the reference duration or character rate. Record warnings and exceptions in the segmentation review. Check beginning, Q&A, and ending for omissions or truncation.
5. **Translate:** Translate each locked English cue into Simplified Chinese. Preserve all meaning and keep names/terms consistent with the glossary. Do not move content between cues. Keep bilingual text in one SRT cue with no internal blank line.
6. **Validate and hand off:** Automatically check SRT syntax, sequential cue numbers starting at 1, comma-millisecond timestamps, blank lines only between cues, bilingual correspondence, overlaps/gaps, missing translations, and segmentation warnings. These checks are alerts, not substitutes for semantic judgment or listening. Inspect generated files only. Check encoding (UTF-8), readability, and cue correspondence. If a burned-in preview is requested, create a new versioned video without overwriting the MP4 and report the scope of visual checks.

### Human-review gate

“Full human review” means a person actually plays the full video and confirms every cue's text and onset, with a traceable record. An AI/automated agent must never invent, infer, or self-certify human confirmation; waveforms, ASR confidence, and scripts are not substitutes. If human review has not happened, produce a cue-by-cue review checklist and leave the status as `candidate` or `awaiting human review`. Sampling must be labeled `sampled review`, never full review. If no suitable playback method is available, document that limitation and leave the work awaiting human review.

Label the output `final` only when all applicable checks pass, all intelligible speech in the stated coverage is represented, unresolved critical issues are absent, all applicable deliverables are validated, and full human review is documented. Otherwise use `candidate` or `awaiting human review`.

### Final response

```text
Status: candidate / awaiting human review / final
Source audio: <WAV path, selected track, format, duration, checks>
ASR/alignment: <model, parameters, tool-provided fields; note missing capabilities>
Deliverables: <paths and cue counts; missing files and reasons>
QA: <onset deviations, end-target exceptions, overlaps, short gaps, translation/segmentation warnings>
Human review: <full cue-by-cue / sampled / not reviewed; scope and record location>
Open issues: <items, or none>
```
