"""Summarize local meeting diagnostics JSONL into quality counts and latency."""

import argparse
import json
import math
import os
from collections import Counter


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 3)


def _sum_numeric_leaves(value, totals, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            _sum_numeric_leaves(item, totals, name)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        totals[prefix] = totals.get(prefix, 0) + value


def summarize(path):
    issue_counts = Counter()
    event_counts = Counter()
    asr_latencies = []
    asr_audio_seconds = 0.0
    correction_latencies = []
    translation_latencies = []
    history_queue_waits = []
    subtitle_audio_to_ui = {}
    subtitle_ui_queue = {}
    audio_queue_drop_frames = 0
    audio_queue_drop_chunks = 0
    audio_capture_gap_frames = 0
    audio_capture_gap_events = 0
    audio_asr_gap_events = 0
    asr_timeline_gap_frames = 0
    asr_timeline_gap_events = 0
    stale_audio_frames = 0
    stale_audio_chunks = 0
    audio_capture_underflows = 0
    audio_capture_overflows = 0
    audio_capture_discontinuities = 0
    audio_queue_drop_events = 0
    stale_audio_events = 0
    ui_backlog_events = 0
    copilot_latencies = []
    copilot_usage_events = []
    copilot_input_characters = 0
    copilot_started = 0
    copilot_completed = 0
    copilot_applied = 0
    copilot_failed_or_discarded = 0
    rows = []
    pipeline_counts = Counter()
    runtime_warning_counts = {}

    with open(path, encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    explicit_quality_issues = set()
    for event in rows:
        event_name = event.get("event", "unknown")
        if "issue" not in event or not event_name.endswith("quality"):
            continue
        issue_key = (event["issue"], event.get("record_index", event.get("index")))
        explicit_quality_issues.add(issue_key)

    for event in rows:
        event_name = event.get("event", "unknown")
        event_counts[event_name] += 1
        if event_name == "session_finished":
            runtime_warning_counts = event.get("runtime_warning_counts", {})
            for warning, count in event.get("warning_counts", {}).items():
                runtime_warning_counts[warning] = max(
                    runtime_warning_counts.get(warning, 0), count
                )
            audio_queue_drop_frames = int(event.get("low_latency_queue_dropped_frames", 0))
            audio_queue_drop_chunks = int(event.get("low_latency_queue_dropped_chunks", 0))
            audio_capture_gap_frames = int(event.get("audio_capture_timeline_gap_frames", 0))
            audio_capture_gap_events = int(event.get("audio_capture_timeline_gap_events", 0))
            audio_asr_gap_events = int(event.get("asr_detected_gap_events", 0))
            stale_audio_frames = int(event.get("stale_audio_frames", 0))
            stale_audio_chunks = int(event.get("stale_audio_chunks", 0))
            audio_capture_underflows = int(event.get("audio_capture_underflow_warnings", 0))
            audio_capture_overflows = int(event.get("audio_capture_overflow_warnings", 0))
            audio_capture_discontinuities = int(
                event.get("audio_capture_discontinuity_warnings", 0)
            )
            asr_timeline_gap_frames = int(
                event.get("quality_stats", {}).get("audio_timeline_gap_frames", 0)
            )
            asr_timeline_gap_events = int(
                event.get("quality_stats", {}).get("audio_timeline_gap_events", 0)
            )
            audio_queue_drop_events = int(
                event.get("quality_stats", {}).get("audio_queue_dropped_chunks", 0)
            )
            stale_audio_events = int(
                event.get("quality_stats", {}).get("audio_stale_chunks", 0)
            )
        elif event_name == "ui_queue_backlog":
            ui_backlog_events += 1
        if event_name in {
            "asr_window", "asr_window_no_text", "asr_segmentation",
            "transcript_segment", "transcript_segment_dropped",
            "transcript_segment_rejected",
            "history_record_dropped", "asr_correction_decision",
            "translation_quality_retry", "translation_quality_retry_result",
            "ui_events_shed_summary", "ui_queue_backlog", "audio_gap",
            "audio_timeline_gap", "audio_device_reconnect", "audio_queue_drop",
            "audio_queue_chunk_dropped", "audio_chunk_stale", "subtitle_presented",
        }:
            pipeline_counts[event_name] += 1
        issue = event.get("issue")
        if issue and not event_name.endswith("quality"):
            issue_counts[issue] += 1
        for item in event.get("issues", []):
            issue_key = (item, event.get("record_index", event.get("index")))
            if event_name == "history_translation" and issue_key in explicit_quality_issues:
                continue
            issue_counts[item] += 1
        if issue and event_name.endswith("quality"):
            issue_counts[issue] += 1
        if event_name == "asr_result":
            asr_latencies.append(float(event.get("seconds", 0.0)))
            asr_audio_seconds += float(event.get("audio_seconds", 0.0))
        elif event_name in {"asr_correction", "asr_correction_candidate"}:
            correction_latencies.append(float(event.get("seconds", 0.0)))
        elif event_name in {"history_translation", "live_translation"}:
            translation_latencies.append(float(event.get("seconds", 0.0)))
        elif event_name == "translation_started" and "queue_wait_seconds" in event:
            history_queue_waits.append(float(event["queue_wait_seconds"]))
        elif event_name == "subtitle_presented":
            kind = event.get("subtitle_kind", "unknown")
            subtitle_audio_to_ui.setdefault(kind, []).append(
                float(event.get("audio_to_ui_seconds", 0.0))
            )
            queue_delay = event.get("ui_queue_seconds")
            if queue_delay is not None:
                subtitle_ui_queue.setdefault(kind, []).append(float(queue_delay))
        elif event_name == "copilot_request_started":
            copilot_started += 1
            copilot_input_characters += int(event.get("input_characters", 0))
        elif event_name == "copilot_request_completed":
            copilot_completed += 1
            copilot_latencies.append(float(event.get("latency_seconds", 0.0)))
            outcome = event.get("outcome")
            if outcome == "applied":
                copilot_applied += 1
            elif outcome == "failed_or_discarded":
                copilot_failed_or_discarded += 1
            usage = event.get("provider_usage", {})
            for item in usage.get("events", []) if isinstance(usage, dict) else []:
                copilot_usage_events.append(item)

    session_elapsed = max(
        (float(event.get("elapsed_seconds", 0.0)) for event in rows), default=0.0
    )
    usage_totals = {}
    for usage in copilot_usage_events:
        _sum_numeric_leaves(usage, usage_totals)

    total_asr_seconds = sum(asr_latencies)
    report = {
        "diagnostics_path": os.path.abspath(path),
        "event_count": len(rows),
        "event_counts": dict(sorted(event_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "asr_result_count": len(asr_latencies),
        "asr_latency_p50_seconds": _percentile(asr_latencies, 0.5),
        "asr_latency_p95_seconds": _percentile(asr_latencies, 0.95),
        "asr_real_time_factor": (
            round(total_asr_seconds / asr_audio_seconds, 3)
            if asr_audio_seconds else None
        ),
        "asr_correction_count": len(correction_latencies),
        "asr_correction_latency_p50_seconds": _percentile(correction_latencies, 0.5),
        "asr_correction_latency_p95_seconds": _percentile(correction_latencies, 0.95),
        "translation_count": len(translation_latencies),
        "translation_latency_p50_seconds": _percentile(translation_latencies, 0.5),
        "translation_latency_p95_seconds": _percentile(translation_latencies, 0.95),
        "history_translation_queue_wait_p50_seconds": _percentile(history_queue_waits, 0.5),
        "history_translation_queue_wait_p95_seconds": _percentile(history_queue_waits, 0.95),
        "subtitle_audio_to_ui": {
            kind: {
                "count": len(values),
                "p50_seconds": _percentile(values, 0.5),
                "p95_seconds": _percentile(values, 0.95),
            }
            for kind, values in sorted(subtitle_audio_to_ui.items())
        },
        "subtitle_ui_queue": {
            kind: {
                "count": len(values),
                "p50_seconds": _percentile(values, 0.5),
                "p95_seconds": _percentile(values, 0.95),
            }
            for kind, values in sorted(subtitle_ui_queue.items())
        },
        "audio_queue_dropped_frames": audio_queue_drop_frames,
        "audio_queue_dropped_chunks": audio_queue_drop_chunks,
        "audio_queue_drop_events": audio_queue_drop_events,
        "audio_capture_timeline_gap_frames": audio_capture_gap_frames,
        "audio_capture_timeline_gap_events": audio_capture_gap_events,
        "asr_detected_gap_events": audio_asr_gap_events,
        "asr_timeline_gap_frames": asr_timeline_gap_frames,
        "asr_timeline_gap_events": asr_timeline_gap_events,
        "stale_audio_frames": stale_audio_frames,
        "stale_audio_chunks": stale_audio_chunks,
        "stale_audio_events": stale_audio_events,
        "ui_queue_backlog_events": ui_backlog_events,
        "audio_capture_underflow_warnings": audio_capture_underflows,
        "audio_capture_overflow_warnings": audio_capture_overflows,
        "audio_capture_discontinuity_warnings": audio_capture_discontinuities,
        "copilot_request_count": copilot_started,
        "copilot_completed_count": copilot_completed,
        "copilot_applied_count": copilot_applied,
        "copilot_failed_or_discarded_count": copilot_failed_or_discarded,
        "copilot_requests_per_session_hour": (
            round(copilot_started * 3600 / session_elapsed, 2)
            if session_elapsed > 0 else None
        ),
        "copilot_input_characters": copilot_input_characters,
        "copilot_latency_p50_seconds": _percentile(copilot_latencies, 0.5),
        "copilot_latency_p95_seconds": _percentile(copilot_latencies, 0.95),
        "copilot_provider_usage_totals": usage_totals,
        "copilot_github_credits": None,
        "copilot_github_credits_status": "not_exposed_by_opencode_cli",
        "pipeline_counts": dict(sorted(pipeline_counts.items())),
        "runtime_warning_counts": dict(sorted(runtime_warning_counts.items())),
    }
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Summarize one local meeting diagnostics.jsonl file."
    )
    parser.add_argument("diagnostics", help="Path to a session diagnostics.jsonl")
    parser.add_argument("--output", help="Optional path for a JSON summary")
    args = parser.parse_args()
    report = summarize(args.diagnostics)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "x", encoding="utf-8") as output:
            output.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
