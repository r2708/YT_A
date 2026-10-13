"""Structural checks. Each returns (ok, issue_message_or_None)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


def check_timespan(start: float, end: float, duration: float, allow_instant: bool = True, tol: float = 0.05) -> tuple[bool, str | None]:
    if start < -tol or end < -tol:
        return False, "negative timestamp"
    if end < start:
        return False, "end_time before start_time"
    if not allow_instant and end - start <= 0:
        return False, "zero duration"
    if start > duration + tol or end > duration + tol:
        return False, f"timestamp beyond video duration ({duration:.2f}s)"
    return True, None


def check_ids_exist(ids: Iterable[str], valid: set[str], label: str) -> tuple[bool, str | None]:
    missing = [i for i in ids if i not in valid]
    if missing:
        return False, f"unknown {label}: {', '.join(missing[:3])}{'…' if len(missing) > 3 else ''}"
    return True, None


def check_files_exist(paths: Iterable[str | Path]) -> tuple[bool, str | None]:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        return False, f"missing files: {', '.join(Path(m).name for m in missing[:3])}{'…' if len(missing) > 3 else ''}"
    return True, None


def check_non_empty(text: str | None, label: str, min_words: int = 1) -> tuple[bool, str | None]:
    if not text or len(text.split()) < min_words:
        return False, f"{label} missing or too short"
    return True, None


def check_frame_sequence(frames: list[Any], start: float, end: float, tol: float = 0.05, min_gap_seconds: float = 5.0) -> tuple[bool, str | None]:
    """The sampled frames of one shot must lie inside it, carry strictly increasing frame indices in
    timestamp order (no duplicates, no reordering) and leave no gap longer than max(min_gap_seconds,
    3x the mean spacing)."""
    if not frames:
        return False, "no frames sampled for this shot"
    ordered = sorted(frames, key=lambda f: (f.timestamp, f.frame_index))
    problems: list[str] = []
    outside = [f for f in ordered if f.timestamp < start - tol or f.timestamp > end + tol]
    if outside:
        problems.append(f"{len(outside)} frame(s) outside the shot")
    idx = [f.frame_index for f in ordered]
    if len(set(idx)) != len(idx):
        problems.append("duplicate frame indices")
    if any(b <= a for a, b in zip(idx, idx[1:])):
        problems.append("frame indices not increasing with time")
    if len(ordered) >= 2:
        span = max(1e-6, end - start)
        gaps = [b.timestamp - a.timestamp for a, b in zip(ordered, ordered[1:])]
        limit = max(min_gap_seconds, 3.0 * span / len(ordered))
        worst = max(gaps)
        if worst > limit:
            problems.append(f"gap of {worst:.1f}s between sampled frames (limit {limit:.1f}s)")
    if problems:
        return False, "frame sequence: " + "; ".join(problems)
    return True, None


def check_clips(clips: list[Any], tolerance: float) -> tuple[list[str], list[str], list[str]]:
    """(missing files, unreadable / zero-length files, clips whose measured length is off by more than
    ``tolerance``) among the clip objects of one shot."""
    missing: list[str] = []
    corrupt: list[str] = []
    inexact: list[str] = []
    for c in clips:
        p = Path(c.clip_path)
        if not p.exists() or p.stat().st_size == 0:
            missing.append(c.clip_id)
            continue
        if c.media_duration is None or c.media_duration <= 0:
            corrupt.append(c.clip_id)
            continue
        if abs(float(c.media_duration) - float(c.end_time - c.start_time)) > tolerance:
            inexact.append(f"{c.clip_id} ({c.media_duration:.2f}s vs {c.end_time - c.start_time:.2f}s)")
    return missing, corrupt, inexact


def confidence_status(confidence: float | None, min_conf: float, review_conf: float) -> str:
    """accepted / review / rejected based on thresholds; None -> review (cannot be trusted or rejected)."""
    if confidence is None:
        return "review"
    if confidence >= min_conf:
        return "accepted"
    if confidence >= review_conf:
        return "review"
    return "rejected"
