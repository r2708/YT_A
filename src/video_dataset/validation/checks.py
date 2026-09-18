"""Structural checks. Each returns (ok, issue_message_or_None)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


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


def confidence_status(confidence: float | None, min_conf: float, review_conf: float) -> str:
    """accepted / review / rejected based on thresholds; None -> review (cannot be trusted or rejected)."""
    if confidence is None:
        return "review"
    if confidence >= min_conf:
        return "accepted"
    if confidence >= review_conf:
        return "review"
    return "rejected"
