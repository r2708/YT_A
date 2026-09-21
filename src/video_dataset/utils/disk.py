"""Free-disk-space guard used before download, transcoding and frame extraction."""

from __future__ import annotations

import shutil
from pathlib import Path

from video_dataset.errors import InsufficientDiskSpace

__all__ = ["InsufficientDiskSpace", "ensure_free_disk", "free_bytes", "free_gb"]


def free_bytes(path: str | Path) -> int:
    """Free bytes on the filesystem holding `path` (walks up to the nearest existing parent)."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(str(p)).free


def free_gb(path: str | Path) -> float:
    return free_bytes(path) / 1e9


def ensure_free_disk(path: str | Path, min_free_gb: float, *, needed_bytes: int | None = None, label: str = "") -> float:
    """Raise InsufficientDiskSpace when the free space under `path` is below the floor.

    `needed_bytes` is an optional known requirement (e.g. the reported download size) that must fit
    on top of the floor. Returns the free space in GB.
    """
    free = free_bytes(path)
    floor = max(0.0, float(min_free_gb or 0.0)) * 1e9
    required = floor + (needed_bytes or 0)
    if free < required:
        what = f" for {label}" if label else ""
        need = f", need {(needed_bytes or 0) / 1e9:.1f} GB" if needed_bytes else ""
        raise InsufficientDiskSpace(
            f"only {free / 1e9:.1f} GB free on {Path(path)}{what} (floor {floor / 1e9:.1f} GB{need}); "
            "free space or lower limits.min_free_disk_gb"
        )
    return free / 1e9
