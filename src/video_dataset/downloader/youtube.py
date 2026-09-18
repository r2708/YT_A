"""yt-dlp based downloader with metadata extraction, existing-download detection and error classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from video_dataset.config import DownloadConfig
from video_dataset.downloader.base import FetchResult, VideoUnavailableError
from video_dataset.schemas.video import DownloadStatus, SourceType, VideoMetadata
from video_dataset.utils.ffmpeg import find_ffmpeg
from video_dataset.utils.logging import get_logger
from video_dataset.utils.retry import retry_call
from video_dataset.utils.urls import InputItem

log = get_logger("downloader.youtube")

_UNAVAILABLE_MARKERS = (
    "private video",
    "video unavailable",
    "this video is unavailable",
    "has been removed",
    "is not available",
    "not available in your country",
    "sign in to confirm your age",
    "members-only",
    "premieres in",
    "this live event",
    "no video formats found",
    "unsupported url",
    "is not a valid url",
    "incomplete youtube id",
)


class _QuietLogger:
    def debug(self, msg: str) -> None:
        if msg.startswith("[debug]"):
            return
        log.debug(msg)

    def info(self, msg: str) -> None:
        log.debug(msg)

    def warning(self, msg: str) -> None:
        log.debug("yt-dlp: %s", msg)

    def error(self, msg: str) -> None:
        log.error("yt-dlp: %s", msg)


def _parse_rate_limit(value: str | None) -> int | None:
    if not value:
        return None
    v = value.strip().upper()
    mult = 1
    if v.endswith("K"):
        mult, v = 1024, v[:-1]
    elif v.endswith("M"):
        mult, v = 1024 * 1024, v[:-1]
    try:
        return int(float(v) * mult)
    except ValueError:
        return None


class YouTubeDownloader:
    name = "yt_dlp"

    def __init__(self, config: DownloadConfig, ffmpeg_path: str | None = None):
        self.config = config
        self.ffmpeg_path = ffmpeg_path

    # -------------------------------------------------------------- options
    def build_options(self, dest_dir: Path) -> dict[str, Any]:
        h = int(self.config.max_resolution)
        fmt = (
            f"bestvideo[height<={h}][ext=mp4]+bestaudio[ext=m4a]/"
            f"bestvideo[height<={h}]+bestaudio/"
            f"best[height<={h}]/best"
        )
        opts: dict[str, Any] = {
            "format": fmt,
            "outtmpl": str(dest_dir / "source.%(ext)s"),
            "merge_output_format": self.config.format or "mp4",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "retries": int(self.config.retries),
            "fragment_retries": int(self.config.retries),
            "socket_timeout": int(self.config.socket_timeout),
            "continuedl": True,
            "overwrites": False,
            "restrictfilenames": True,
            "writesubtitles": False,
            "writeinfojson": False,
            "writethumbnail": False,
            "getcomments": False,
            "logger": _QuietLogger(),
        }
        try:
            opts["ffmpeg_location"] = str(Path(find_ffmpeg(self.ffmpeg_path)).parent)
        except Exception:  # pragma: no cover - ffmpeg checked earlier by `doctor`
            pass
        if self.config.cookies_file:
            opts["cookiefile"] = str(Path(self.config.cookies_file).expanduser())
        rl = _parse_rate_limit(self.config.rate_limit)
        if rl:
            opts["ratelimit"] = rl
        opts.update(self.config.extra_args or {})
        return opts

    # -------------------------------------------------------------- metadata
    @staticmethod
    def metadata_from_info(info: dict[str, Any], video_id: str, url: str, path: Path | None) -> VideoMetadata:
        """Public video metadata only - nothing about viewers/commenters."""
        fps = info.get("fps")
        return VideoMetadata(
            video_id=video_id,
            youtube_id=info.get("id"),
            url=info.get("webpage_url") or url,
            source_type=SourceType.YOUTUBE,
            title=info.get("title"),
            channel=info.get("channel") or info.get("uploader"),
            duration=float(info["duration"]) if info.get("duration") is not None else None,
            upload_date=info.get("upload_date"),
            width=info.get("width"),
            height=info.get("height"),
            fps=float(fps) if fps else None,
            format=(path.suffix.lstrip(".") if path else info.get("ext")),
            license=info.get("license"),
            categories=list(info.get("categories") or []),
            tags=list(info.get("tags") or [])[:30],
            language=info.get("language"),
        )

    # -------------------------------------------------------------- fetch
    def existing_download(self, dest_dir: Path) -> Path | None:
        if not dest_dir.exists():
            return None
        for p in sorted(dest_dir.glob("source.*")):
            if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".m4v"} and p.stat().st_size > 0:
                return p
        return None

    def fetch(self, item: InputItem, video_id: str, dest_dir: Path) -> FetchResult:
        import yt_dlp
        from yt_dlp.utils import DownloadError, ExtractorError

        dest_dir.mkdir(parents=True, exist_ok=True)
        url = item.url
        existing = self.existing_download(dest_dir) if self.config.skip_existing else None
        opts = self.build_options(dest_dir)
        if existing is not None:
            opts["skip_download"] = True  # still refresh metadata cheaply

        attempts = {"n": 0}

        def _run() -> dict[str, Any]:
            attempts["n"] += 1
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    raise RuntimeError("yt-dlp returned no info")
                if "entries" in info:  # playlist guard
                    entries = [e for e in info["entries"] if e]
                    if not entries:
                        raise VideoUnavailableError("playlist contained no downloadable entries")
                    info = entries[0]
                return dict(info)

        def _classify(exc: BaseException) -> None:
            msg = str(exc).lower()
            if any(m in msg for m in _UNAVAILABLE_MARKERS):
                raise VideoUnavailableError(str(exc)) from exc

        try:
            try:
                info = retry_call(
                    _run,
                    retries=max(0, int(self.config.retries) - 1),
                    base_delay=2.0,
                    retry_on=(DownloadError, ExtractorError, OSError, RuntimeError),
                    on_retry=lambda exc, _n: _classify(exc),
                    label=f"download {url}",
                )
            except (DownloadError, ExtractorError) as exc:
                _classify(exc)
                raise
        except VideoUnavailableError as exc:
            return FetchResult(DownloadStatus.UNAVAILABLE, None, None, error=str(exc)[:500], attempts=attempts["n"])
        except Exception as exc:
            return FetchResult(DownloadStatus.FAILED, None, None, error=f"{type(exc).__name__}: {exc}"[:500], attempts=attempts["n"])

        path = existing or self._locate_file(info, dest_dir)
        if path is None or not path.exists():
            return FetchResult(DownloadStatus.FAILED, None, None, error="download finished but no media file found", attempts=attempts["n"])
        metadata = self.metadata_from_info(info, video_id, url, path)
        status = DownloadStatus.SKIPPED_EXISTING if existing is not None else DownloadStatus.DONE
        return FetchResult(status, path, metadata, attempts=attempts["n"])

    @staticmethod
    def _locate_file(info: dict[str, Any], dest_dir: Path) -> Path | None:
        for rd in info.get("requested_downloads") or []:
            fp = rd.get("filepath") or rd.get("_filename")
            if fp and Path(fp).exists():
                return Path(fp)
        fp = info.get("filepath") or info.get("_filename")
        if fp and Path(fp).exists():
            return Path(fp)
        candidates = [p for p in dest_dir.glob("source.*") if p.suffix.lower() not in {".part", ".ytdl", ".json"}]
        return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


def probe_metadata_only(url: str, config: DownloadConfig) -> dict[str, Any]:
    """Fetch metadata without downloading (used by `video-dataset info`)."""
    import yt_dlp

    dl = YouTubeDownloader(config)
    opts = dl.build_options(Path("."))
    opts["skip_download"] = True
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return dict(info or {})


logging.getLogger("yt_dlp").setLevel(logging.WARNING)
