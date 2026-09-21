from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from video_dataset.errors import VideoRejected
from video_dataset.schemas.video import DownloadStatus, VideoMetadata
from video_dataset.utils.urls import InputItem


@dataclass
class FetchResult:
    status: DownloadStatus
    video_path: Path | None
    metadata: VideoMetadata | None
    error: str | None = None
    attempts: int = 1


class VideoSource(Protocol):
    name: str

    def fetch(self, item: InputItem, video_id: str, dest_dir: Path) -> FetchResult: ...


class VideoUnavailableError(RuntimeError):
    """The video cannot be fetched at all (private, removed, geo-blocked). Not retried."""


class VideoRejectedError(VideoRejected):
    """The video violates a configured limit (URL policy, duration, size, disk space). Not retried."""
