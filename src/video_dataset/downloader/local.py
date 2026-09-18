"""Treat a local video file as a source (used for tests and for videos already on disk)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from video_dataset.downloader.base import FetchResult
from video_dataset.schemas.video import DownloadStatus, SourceType, VideoMetadata
from video_dataset.utils.urls import InputItem


class LocalFileSource:
    name = "local"

    def fetch(self, item: InputItem, video_id: str, dest_dir: Path) -> FetchResult:
        if not item.local_path:
            return FetchResult(DownloadStatus.FAILED, None, None, error="no local path")
        src = Path(item.local_path)
        if not src.exists():
            return FetchResult(DownloadStatus.UNAVAILABLE, None, None, error=f"file not found: {src}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"source{src.suffix.lower()}"
        status = DownloadStatus.DONE
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            status = DownloadStatus.SKIPPED_EXISTING
        else:
            if dest.exists():
                dest.unlink()
            try:
                os.link(src, dest)  # instant when on the same filesystem
            except OSError:
                shutil.copy2(src, dest)
        metadata = VideoMetadata(
            video_id=video_id,
            youtube_id=None,
            url=src.resolve().as_uri(),
            source_type=SourceType.LOCAL,
            title=src.stem,
            format=src.suffix.lstrip(".").lower(),
        )
        return FetchResult(status, dest, metadata)
