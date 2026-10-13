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


def _field_counts(records: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        v = r.get(key) if isinstance(r, dict) else None
        k = str(v) if v is not None else "unassigned"
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def _split_videos(records: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    seen: dict[str, set[str]] = {}
    for rows in records.values():
        for r in rows:
            if isinstance(r, dict) and r.get("video_id"):
                seen.setdefault(str(r.get("split") or "unassigned"), set()).add(str(r["video_id"]))
    return {k: len(v) for k, v in sorted(seen.items())}


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
        "sequences": len(records.get("sequences", [])),
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
        "videos_by_split": _split_videos(records),
        "records_by_split": {name: _field_counts(recs, "split") for name, recs in records.items()},
        "records_by_tier": {name: _field_counts(recs, "tier") for name, recs in records.items()},
        "records_with_hard_negatives": {name: sum(1 for r in recs if isinstance(r, dict) and r.get("hard_negatives")) for name, recs in records.items() if name in ("scenes", "clips", "temporal_qa", "long_video_qa")},
        "records_in_cinematic_subset": {name: sum(1 for r in recs if isinstance(r, dict) and "cinematic" in (r.get("subsets") or [])) for name, recs in records.items()},
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


# ------------------------------------------------------------------------------------ diversity
_UNKNOWN = {None, "", "unknown", "None", "unassigned"}
_DURATION_BUCKETS = [(0, 2, "<2s"), (2, 5, "2-5s"), (5, 10, "5-10s"), (10, 30, "10-30s"), (30, 1e9, ">=30s")]
_AESTHETIC_BUCKETS = [(0, 4, "<4"), (4, 5, "4-5"), (5, 6, "5-6"), (6, 7, "6-7"), (7, 11, ">=7")]


def _bucket(value: float | None, buckets: list[tuple[float, float, str]]) -> str:
    if value is None:
        return "unknown"
    for lo, hi, name in buckets:
        if lo <= float(value) < hi:
            return name
    return "unknown"


def _dist(values: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        k = "unknown" if v in _UNKNOWN else str(v)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _underrepresented(dist: dict[str, int], floor_fraction: float = 0.5) -> list[dict[str, Any]]:
    """Known categories whose share is below ``floor_fraction`` of the uniform share (1 / n_known)."""
    known = {k: n for k, n in dist.items() if k != "unknown"}
    total = sum(known.values())
    if total == 0 or len(known) < 2:
        return []
    floor = floor_fraction / len(known)
    return [{"value": k, "count": n, "share": round(n / total, 4)} for k, n in sorted(known.items(), key=lambda kv: kv[1]) if n / total < floor]


def compute_diversity(records: dict[str, list[dict[str, Any]]], manifests: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Distributions of the dataset's measured / labelled attributes over scene records (camera,
    subjects, environment, lighting, colour, motion, duration, quality) and per-video facts
    (license, resolution, near-duplicates), with the under-represented categories of each listed."""
    scenes = [r for r in records.get("scenes", []) if isinstance(r, dict)]
    cam = [r.get("camera") or {} for r in scenes]
    env = [r.get("environment") or {} for r in scenes]
    meas = [r.get("measurements") or {} for r in scenes]
    sm = [r.get("subject_motion") or {} for r in scenes]
    dims: dict[str, dict[str, int]] = {
        "shot_size": _dist(c.get("shot_size") for c in cam),
        "camera_movement": _dist(c.get("camera_movement") for c in cam),
        "camera_movement_direction": _dist(f"{c.get('camera_movement')}_{c.get('movement_direction')}" if c.get("movement_direction") else c.get("camera_movement") for c in cam),
        "movement_speed": _dist(c.get("movement_speed") for c in cam),
        "camera_angle": _dist(c.get("camera_angle") for c in cam),
        "setting": _dist(e.get("setting") for e in env),
        "time_of_day": _dist(e.get("time_of_day") for e in env),
        "lighting_level": _dist(m.get("lighting_level") for m in meas),
        "color_temperature": _dist(m.get("color_temperature") for m in meas),
        "exposure": _dist(m.get("exposure") for m in meas),
        "subject_motion": _dist(s.get("label") for s in sm),
        "subject_direction": _dist(s.get("direction") for s in sm),
        "transition_in": _dist(r.get("transition_in") for r in scenes),
        "scene_duration": _dist(_bucket(r.get("duration"), _DURATION_BUCKETS) for r in scenes),
        "aesthetic_score": _dist(_bucket(m.get("aesthetic_score"), _AESTHETIC_BUCKETS) for m in meas),
        "provider": _dist(r.get("provider") for r in scenes),
    }
    subjects: dict[str, int] = {}
    for r in scenes:
        for o in set(str(x).lower() for x in (r.get("objects") or [])):
            subjects[o] = subjects.get(o, 0) + 1
    flags: dict[str, int] = {}
    for r in scenes:
        for f in r.get("quality_flags") or []:
            flags[str(f)] = flags.get(str(f), 0) + 1
    videos = [m for m in (manifests or []) if isinstance(m, dict) and m.get("video_id")]
    video_dims = {
        "license_status": _dist(m.get("license_status") for m in videos),
        "source_height": _dist((m.get("media") or {}).get("height") for m in videos),
        "near_duplicate": _dist("duplicate" if m.get("canonical_video_id") or m.get("near_duplicate_of") else "unique" for m in videos),
    }
    return {
        "scenes": len(scenes),
        "videos": len(videos),
        "dimensions": dims,
        "subjects": dict(sorted(subjects.items(), key=lambda kv: (-kv[1], kv[0]))[:40]),
        "distinct_subjects": len(subjects),
        "quality_flags": dict(sorted(flags.items(), key=lambda kv: -kv[1])),
        "videos_by": video_dims,
        "underrepresented": {k: v for k, v in ((k, _underrepresented(d)) for k, d in dims.items()) if v},
        "unknown_fraction": {k: round(d.get("unknown", 0) / max(1, sum(d.values())), 3) for k, d in dims.items()},
    }


def format_diversity(div: dict[str, Any], top: int = 8) -> str:
    lines = ["Dataset Diversity", "-----------------", f"scenes: {div.get('scenes', 0):,}   videos: {div.get('videos', 0):,}   distinct subjects: {div.get('distinct_subjects', 0)}", ""]
    for key, dist in (div.get("dimensions") or {}).items():
        total = sum(dist.values()) or 1
        parts = [f"{k} {n} ({n / total:.0%})" for k, n in list(dist.items())[:top]]
        lines.append(f"{key:<28} " + ", ".join(parts))
    if div.get("subjects"):
        lines.append("")
        lines.append("top subjects: " + ", ".join(f"{k} {n}" for k, n in list(div["subjects"].items())[:top * 2]))
    for key, dist in (div.get("videos_by") or {}).items():
        lines.append(f"videos by {key:<18} " + ", ".join(f"{k} {n}" for k, n in list(dist.items())[:top]))
    if div.get("quality_flags"):
        lines.append("quality flags: " + ", ".join(f"{k} {n}" for k, n in div["quality_flags"].items()))
    under = div.get("underrepresented") or {}
    lines.append("")
    if under:
        lines.append("under-represented (below half the uniform share):")
        for key, items in under.items():
            lines.append(f"  {key}: " + ", ".join(f"{i['value']} ({i['share']:.1%})" for i in items))
    else:
        lines.append("no under-represented category")
    return "\n".join(lines)


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
    splits = stats.get("videos_by_split") or {}
    if splits:
        lines.append("")
        lines.append("Videos by split")
        for name, n in splits.items():
            row(f"  {name}", n)
    tiers = (stats.get("records_by_tier") or {}).get("scenes") or {}
    qa_tiers = (stats.get("records_by_tier") or {}).get("temporal_qa") or {}
    if tiers or qa_tiers:
        lines.append("")
        lines.append("Tiers (scenes / temporal QA)")
        for name in ("gold", "silver", "bronze", "unassigned"):
            if tiers.get(name) or qa_tiers.get(name):
                row(f"  {name}", f"{tiers.get(name, 0):,} / {qa_tiers.get(name, 0):,}")
    cine = stats.get("records_in_cinematic_subset") or {}
    if cine.get("scenes"):
        lines.append("")
        row("Cinematic scenes", cine.get("scenes", 0))
    files = stats.get("files") or {}
    if files:
        lines.append("")
        lines.append("Files")
        for key in ("dataset_jsonl", "dataset_parquet"):
            if key in files:
                lines.append(f"  {files[key]}")
    return "\n".join(lines)
