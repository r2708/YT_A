"""Stable identifier generation. IDs are deterministic so re-runs never produce duplicates."""

from __future__ import annotations

import hashlib
from pathlib import Path


def stable_hash(*parts: object, length: int = 12) -> str:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return h[:length]


def make_video_id(youtube_id: str | None = None, local_path: str | Path | None = None, url: str | None = None) -> str:
    """vid_<12 hex>. Stable across URL variants of the same YouTube video."""
    if youtube_id:
        return "vid_" + stable_hash("youtube", youtube_id)
    if local_path:
        p = Path(local_path)
        try:
            size = p.stat().st_size
        except OSError:
            size = -1
        return "vid_" + stable_hash("local", p.name, size)
    if url:
        return "vid_" + stable_hash("url", url)
    raise ValueError("Need youtube_id, local_path or url to derive a video id")


def scene_id(index: int) -> str:
    return f"scene_{index + 1:03d}"


def frame_id(scene_index: int, ordinal: int) -> str:
    return f"frame_{scene_index + 1:03d}_{ordinal:03d}"


def clip_id(scene_index: int, part: int = 0) -> str:
    return f"clip_{scene_index + 1:03d}" + (f"_{part:02d}" if part else "")


def event_id(n: int) -> str:
    return f"event_{n:04d}"


def relation_id(n: int) -> str:
    return f"rel_{n:05d}"


def qa_id(video_id: str, n: int) -> str:
    return f"qa_{video_id.removeprefix('vid_')}_{n:06d}"


def segment_id(n: int) -> str:
    return f"seg_{n:04d}"


def track_id(n: int) -> str:
    return f"ocr_{n:04d}"


def audio_event_id(n: int) -> str:
    return f"aud_{n:04d}"


def record_id(kind: str, video_id: str, local: str) -> str:
    return f"{kind}_{video_id.removeprefix('vid_')}_{local}"
