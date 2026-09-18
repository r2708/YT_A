"""Dataset statistics (printed as a table and exported as JSON)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from video_dataset.schemas.quality import ValidationStatus


def _status_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    out = {s.value: 0 for s in ValidationStatus}
    for r in records:
        st = ((r.get("validation") or {}).get("status")) if isinstance(r, dict) else None
        if st in out:
            out[st] += 1
    return out


def compute_statistics(
    videos: list[dict[str, Any]],
    records: dict[str, list[dict[str, Any]]],
    validated_summaries: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_duration = sum(float(v.get("duration") or 0.0) for v in videos)
    qa_all = records.get("temporal_qa", []) + records.get("long_video_qa", [])
    qa_by_type: dict[str, int] = {}
    qa_by_difficulty: dict[str, int] = {}
    for q in qa_all:
        qa_by_type[str(q.get("type"))] = qa_by_type.get(str(q.get("type")), 0) + 1
        qa_by_difficulty[str(q.get("difficulty"))] = qa_by_difficulty.get(str(q.get("difficulty")), 0) + 1
    events_by_type: dict[str, int] = {}
    for e in records.get("events", []):
        events_by_type[str(e.get("event_type"))] = events_by_type.get(str(e.get("event_type")), 0) + 1
    relations = sum(len(e.get("relations") or []) for e in records.get("events", []))

    agg: dict[str, dict[str, int]] = {"scenes": {}, "events": {}, "relations": {}, "qa": {}}
    for s in validated_summaries:
        for key in agg:
            for st, n in (s.get(key) or {}).items():
                agg[key][st] = agg[key].get(st, 0) + int(n)

    stats = {
        "videos": len(videos),
        "videos_by_status": _count_by(videos, "status"),
        "total_duration_seconds": round(total_duration, 1),
        "total_duration_hours": round(total_duration / 3600.0, 2),
        "scenes": len(records.get("scenes", [])),
        "frames": len(records.get("frames", [])),
        "clips": len(records.get("clips", [])),
        "events": len(records.get("events", [])),
        "events_by_type": events_by_type,
        "temporal_relations": relations,
        "temporal_qa": len(records.get("temporal_qa", [])),
        "long_video_qa": len(records.get("long_video_qa", [])),
        "qa_by_type": qa_by_type,
        "qa_by_difficulty": qa_by_difficulty,
        "video_descriptions": len(records.get("video_descriptions", [])),
        "validation": agg,
        "exported_status_counts": {name: _status_counts(recs) for name, recs in records.items()},
    }
    if extra:
        stats.update(extra)
    return stats


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        k = str(r.get(key))
        out[k] = out.get(k, 0) + 1
    return out


def format_statistics(stats: dict[str, Any]) -> str:
    lines = ["Dataset Statistics", "------------------", ""]

    def row(label: str, value: Any) -> None:
        lines.append(f"{label + ':':<22}{value:>12,}" if isinstance(value, int) else f"{label + ':':<22}{value:>12}")

    row("Videos", stats["videos"])
    row("Total duration", f"{stats['total_duration_hours']:.2f} hours")
    lines.append("")
    row("Scenes", stats["scenes"])
    row("Frames", stats["frames"])
    row("Clips", stats["clips"])
    row("Events", stats["events"])
    lines.append("")
    row("Transcripts", stats.get("transcript_segments", 0))
    row("OCR records", stats.get("ocr_tracks", 0))
    lines.append("")
    row("Temporal relations", stats["temporal_relations"])
    lines.append("")
    row("Temporal QA", stats["temporal_qa"])
    row("Long-video QA", stats["long_video_qa"])
    lines.append("")
    qa = stats["validation"].get("qa", {})
    row("Accepted", qa.get("accepted", 0))
    row("Rejected", qa.get("rejected", 0))
    row("Needs review", qa.get("review", 0))
    row("Duplicates", qa.get("duplicate", 0))
    files = stats.get("files") or {}
    if files:
        lines.append("")
        lines.append("Files")
        for key in ("dataset_jsonl", "dataset_parquet"):
            if key in files:
                lines.append(f"  {files[key]}")
    return "\n".join(lines)
