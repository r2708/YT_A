"""Video-level schemas: metadata, probe results, download records."""

from __future__ import annotations

from pydantic import Field

from video_dataset.schemas.common import BaseSchema, StrEnum


class SourceType(StrEnum):
    YOUTUBE = "youtube"
    LOCAL = "local"


class VideoMetadata(BaseSchema):
    """Public, non-personal metadata about a source video.

    Deliberately excludes comments, commenter identities and other user PII.
    """

    video_id: str
    youtube_id: str | None = None
    url: str
    source_type: SourceType = SourceType.YOUTUBE
    title: str | None = None
    channel: str | None = None
    duration: float | None = Field(default=None, ge=0)
    upload_date: str | None = None  # YYYYMMDD as returned by yt-dlp
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    format: str | None = None
    license: str | None = None
    categories: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    language: str | None = None


class AudioStreamInfo(BaseSchema):
    codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bitrate: int | None = None


class MediaInfo(BaseSchema):
    """Technical properties measured with ffprobe/OpenCV on the local file."""

    path: str
    duration: float = Field(ge=0)
    fps: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_count: int = Field(ge=0)
    video_codec: str | None = None
    bitrate: int | None = None
    container: str | None = None
    has_audio: bool = False
    audio: AudioStreamInfo | None = None
    size_bytes: int | None = None


class DownloadStatus(StrEnum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
    SKIPPED_EXISTING = "skipped_existing"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"  # blocked by policy before download: URL/domain, duration, size or disk limits


class DownloadRecord(BaseSchema):
    video_id: str
    url: str
    status: DownloadStatus
    video_path: str | None = None
    metadata_path: str | None = None
    error: str | None = None
    attempts: int = 0
    sha256: str | None = None
