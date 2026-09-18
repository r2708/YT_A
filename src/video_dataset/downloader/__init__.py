"""Video acquisition: YouTube via yt-dlp, or local files."""

from video_dataset.downloader.base import FetchResult, VideoSource
from video_dataset.downloader.local import LocalFileSource
from video_dataset.downloader.youtube import YouTubeDownloader

__all__ = ["FetchResult", "LocalFileSource", "VideoSource", "YouTubeDownloader"]
