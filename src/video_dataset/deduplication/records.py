"""Dedup for records that carry a time window (events, scene descriptions).

Two records are duplicates only when their text is (near-)identical AND their windows overlap enough.
The same description at a different time is a different record and is kept."""

from __future__ import annotations

from video_dataset.deduplication.near import NearDuplicateIndex
from video_dataset.utils.text import normalize_text
from video_dataset.utils.timecode import interval_iou


def dedupe_timed_texts(
    items: list[tuple[str, str, float, float]],
    text_threshold: float = 0.9,
    iou_threshold: float = 0.5,
    require_overlap: bool = True,
) -> dict[str, str]:
    """items = [(id, text, start, end)] -> {duplicate_id: canonical_id}."""
    index = NearDuplicateIndex(text_threshold)
    windows: dict[str, tuple[float, float]] = {}
    norm: dict[str, str] = {}
    dups: dict[str, str] = {}
    for item_id, text, start, end in sorted(items, key=lambda x: (x[2], x[3])):
        n = normalize_text(text)
        matches = [m for m in index.query(text) if m not in dups]
        exact = [m for m in windows if norm[m] == n and m not in dups]
        canon = None
        for m in exact + matches:
            if not require_overlap or interval_iou(start, end, *windows[m]) >= iou_threshold:
                canon = m
                break
        if canon is not None:
            dups[item_id] = canon
            continue
        index.add(item_id, text)
        windows[item_id] = (start, end)
        norm[item_id] = n
    return dups
