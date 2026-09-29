"""yt-dlp based downloader with metadata extraction, existing-download detection and error classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from video_dataset.config import DownloadConfig, LimitsConfig
from video_dataset.downloader.base import FetchResult, VideoRejectedError, VideoUnavailableError
from video_dataset.schemas.video import DownloadStatus, SourceType, VideoMetadata
from video_dataset.utils.disk import free_bytes
from video_dataset.utils.ffmpeg import find_ffmpeg
from video_dataset.utils.logging import get_logger
from video_dataset.utils.retry import retry_call
from video_dataset.utils.urls import InputItem

log = get_logger("downloader.youtube")

# Messages yt-dlp / its cookie extractor produce when the cookie source itself is broken (not the video).
# On Windows, Chrome cookies fail in two ways when yt-dlp runs from a background process / IDE:
#   * "Failed to decrypt with DPAPI" - the DPAPI key is only available to the user session / process
#     context that encrypted the cookies, so every cookie decrypts to garbage;
#   * "Could not copy Chrome cookie database" - Chrome holds an exclusive lock on its Cookies SQLite
#     file while it is open.
# Neither has anything to do with the video, so the download must continue without cookies.
_COOKIE_FAILURE_MARKERS = (
    "failed to load cookies",
    "failed to decrypt with dpapi",
    "could not copy chrome cookie database",
    "could not copy",  # generic "could not copy <browser> cookie database"
    "cookie database",
    "cookies database",
    "unsupported browser",
    "unknown browser",
    "unsupported keyring",
    "could not find the cookies database",
    "database is locked",
    "cookies file",
    "netscape format",
    "invalid netscape",
)

# Browser cookie extraction is probed once per process per browser spec (the result cannot change
# between videos, and re-extracting for every video would re-trigger the same DPAPI / lock errors).
_BROWSER_COOKIE_PROBE: dict[tuple[Any, ...], bool] = {}

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


class _CookieProbeLogger:
    """Logger handed to yt-dlp's cookie extractor during the preflight probe. Errors are demoted to
    debug because the probe's *outcome* is what gets reported (once), not every failed cookie row."""

    def debug(self, msg: str, *a: Any, **k: Any) -> None:
        log.debug("cookies: %s", msg)

    def info(self, msg: str, *a: Any, **k: Any) -> None:
        log.debug("cookies: %s", msg)

    def warning(self, msg: str, *a: Any, **k: Any) -> None:
        log.debug("cookies: %s", msg)

    def error(self, msg: str, *a: Any, **k: Any) -> None:
        log.debug("cookies: %s", msg)


def parse_browser_spec(spec: Any) -> tuple[str, str | None, str | None, str | None] | None:
    """Normalize a browser cookie spec into yt-dlp's (browser, profile, keyring, container) tuple.

    Accepts the CLI string form ``BROWSER[+KEYRING][:PROFILE][::CONTAINER]`` (e.g. ``chrome``,
    ``chrome:Profile 1``, ``firefox::personal``), or a list/tuple of 1-4 items as yt-dlp's Python API
    expects. Returns None for an empty spec. A bare string is *not* passed through unchanged because
    yt-dlp would unpack ``"chrome"`` character by character.
    """
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        items = [str(x) if x is not None else None for x in spec]
        if not items or not items[0]:
            return None
        items = (items + [None, None, None])[:4]
        return (str(items[0]).lower(), items[1] or None, items[2] or None, items[3] or None)
    text = str(spec).strip()
    if not text:
        return None
    container: str | None = None
    if "::" in text:
        text, container = text.split("::", 1)
    profile: str | None = None
    if ":" in text:
        text, profile = text.split(":", 1)
    keyring: str | None = None
    if "+" in text:
        text, keyring = text.split("+", 1)
    return (text.strip().lower(), profile or None, (keyring or None) and keyring.upper(), container or None)


def is_cookie_failure(exc: BaseException | str) -> bool:
    """True when an error is about loading cookies rather than about the video."""
    msg = str(exc).lower()
    return any(m in msg for m in _COOKIE_FAILURE_MARKERS)


def probe_browser_cookies(spec: tuple[str, str | None, str | None, str | None]) -> tuple[bool, str]:
    """Try to extract cookies from the browser once. Returns (ok, detail).

    ``ok`` is False when yt-dlp raises (locked database, missing profile, unsupported browser) *or*
    when the extraction yields no usable cookies (the Windows DPAPI case: every row fails to decrypt
    and yt-dlp silently returns an empty jar, which would then cause bot-check failures anyway).
    """
    try:
        from yt_dlp.cookies import extract_cookies_from_browser

        browser, profile, keyring, container = spec
        jar = extract_cookies_from_browser(browser, profile, _CookieProbeLogger(), keyring=keyring, container=container)
    except Exception as exc:  # any failure here means "no browser cookies"
        return False, f"{type(exc).__name__}: {exc}"
    try:
        count = len(jar)
    except TypeError:  # pragma: no cover - defensive: unexpected jar type
        count = 1
    if count == 0:
        return False, "browser returned 0 usable cookies (on Windows this usually means DPAPI decryption failed)"
    return True, f"{count} cookies"


def reset_cookie_probe_cache() -> None:
    _BROWSER_COOKIE_PROBE.clear()


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

    def __init__(self, config: DownloadConfig, ffmpeg_path: str | None = None, limits: LimitsConfig | None = None):
        self.config = config
        self.ffmpeg_path = ffmpeg_path
        self.limits = limits or LimitsConfig()

    # -------------------------------------------------------------- limits
    def check_limits(self, info: dict[str, Any], dest_dir: Path | None = None) -> str | None:
        """Reason the video must not be downloaded (duration / size / disk), or None. Uses yt-dlp's
        metadata so the decision is made before any media bytes are transferred."""
        lim = self.limits
        duration = info.get("duration")
        if lim.max_duration_seconds and duration and float(duration) > float(lim.max_duration_seconds):
            return f"duration {float(duration) / 60:.1f} min exceeds limits.max_duration_seconds ({float(lim.max_duration_seconds) / 60:.0f} min)"
        size = info.get("filesize") or info.get("filesize_approx")
        if size:
            size = int(size)
            if lim.max_file_size_gb and size > float(lim.max_file_size_gb) * 1e9:
                return f"reported size {size / 1e9:.2f} GB exceeds limits.max_file_size_gb ({lim.max_file_size_gb})"
            if dest_dir is not None and lim.min_free_disk_gb is not None:
                free = free_bytes(dest_dir)
                if free < size + float(lim.min_free_disk_gb) * 1e9:
                    return f"not enough disk: {free / 1e9:.1f} GB free, download needs {size / 1e9:.2f} GB plus a {lim.min_free_disk_gb} GB floor"
        return None

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
        except Exception as exc:  # pragma: no cover - ffmpeg checked earlier by `doctor`
            log.warning("ffmpeg not found (%s); yt-dlp will look for it on PATH and cannot merge streams without it", exc)
        rl = _parse_rate_limit(self.config.rate_limit)
        if rl:
            opts["ratelimit"] = rl
        opts.update(self.config.extra_args or {})
        self._apply_cookie_options(opts)
        return opts

    # -------------------------------------------------------------- cookies
    def _cookies_optional(self) -> bool:
        return bool(getattr(self.config, "cookies_optional", True))

    def _apply_cookie_options(self, opts: dict[str, Any]) -> None:
        """Attach cookie options that are known to work; drop the ones that cannot be loaded.

        ``download.cookies_from_browser`` (or a raw ``cookiesfrombrowser`` in ``extra_args``) is probed
        once per process. If the browser's cookies cannot be read - on Windows Chrome keeps its
        cookie database locked and DPAPI refuses to decrypt for a background process - a warning is
        logged and the download proceeds *without* cookies, because public videos do not need them.
        Set ``download.cookies_optional: false`` to make that a hard error instead.
        """
        # --- cookies file --------------------------------------------------------------------
        cookie_file = self.config.cookies_file or opts.pop("cookiefile", None)
        if cookie_file:
            path = Path(str(cookie_file)).expanduser()
            if path.is_file():
                opts["cookiefile"] = str(path)
            else:
                msg = f"download.cookies_file {path} does not exist or is not a file"
                if not self._cookies_optional():
                    raise FileNotFoundError(msg)
                log.warning("%s; continuing without cookies (set download.cookies_optional: false to make this fatal)", msg)

        # --- cookies from browser ------------------------------------------------------------
        raw_spec = getattr(self.config, "cookies_from_browser", None) or opts.pop("cookiesfrombrowser", None)
        opts.pop("cookiesfrombrowser", None)
        try:
            spec = parse_browser_spec(raw_spec)
        except Exception as exc:
            spec = None
            log.warning("invalid download.cookies_from_browser %r (%s); continuing without browser cookies", raw_spec, exc)
        if spec is None:
            return

        if spec not in _BROWSER_COOKIE_PROBE:
            ok, detail = probe_browser_cookies(spec)
            _BROWSER_COOKIE_PROBE[spec] = ok
            if ok:
                log.info("using cookies from %s (%s)", spec[0], detail)
            else:
                msg = f"could not load cookies from browser '{spec[0]}': {detail}"
                if not self._cookies_optional():
                    raise RuntimeError(msg)
                log.warning(
                    "%s. Continuing WITHOUT cookies - public videos still download; age-restricted / "
                    "members-only ones will be recorded as unavailable. On Windows, close Chrome or export "
                    "cookies to a file and set download.cookies_file instead.",
                    msg,
                )
        if _BROWSER_COOKIE_PROBE[spec]:
            opts["cookiesfrombrowser"] = spec
        elif not self._cookies_optional():  # cached failure on a later video
            raise RuntimeError(f"could not load cookies from browser '{spec[0]}'")

    @staticmethod
    def _strip_cookie_options(opts: dict[str, Any]) -> bool:
        """Remove every cookie-related option. Returns True if anything was removed."""
        removed = False
        for key in ("cookiesfrombrowser", "cookiefile"):
            if key in opts:
                opts.pop(key, None)
                removed = True
        return removed

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
        rejected: dict[str, str] = {}

        def _match_filter(info: dict[str, Any], *, incomplete: bool = False) -> str | None:
            # yt-dlp calls this with the metadata before downloading; a string skips the download.
            if incomplete:
                return None
            reason = self.check_limits(info, dest_dir)
            if reason:
                rejected["reason"] = reason
            return reason

        if existing is None:
            opts["match_filter"] = _match_filter

        def _extract() -> Any:
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=True)

        def _run() -> dict[str, Any]:
            attempts["n"] += 1
            try:
                info = _extract()
            except Exception as exc:  # re-raised below unless it is a cookie problem
                # Second line of defence: the preflight probe passed (or was bypassed) but yt-dlp
                # still failed while loading cookies. Drop them and try again without.
                if not (is_cookie_failure(exc) and self._cookies_optional() and self._strip_cookie_options(opts)):
                    raise
                log.warning(
                    "yt-dlp could not load cookies (%s: %s); retrying %s without cookies",
                    type(exc).__name__, str(exc)[:200], url,
                )
                info = _extract()
            if info is None:
                raise RuntimeError("yt-dlp returned no info")
            if "entries" in info:  # playlist guard
                entries = [e for e in info["entries"] if e]
                if not entries:
                    raise VideoUnavailableError("playlist contained no downloadable entries")
                info = entries[0]
            if rejected.get("reason"):
                raise VideoRejectedError(rejected["reason"])
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
                    no_retry_on=(VideoRejectedError, VideoUnavailableError),
                    on_retry=lambda exc, _n: _classify(exc),
                    label=f"download {url}",
                )
            except (DownloadError, ExtractorError) as exc:
                _classify(exc)
                raise
        except VideoRejectedError as exc:
            log.warning("rejected %s: %s", url, exc)
            return FetchResult(DownloadStatus.REJECTED, None, None, error=str(exc)[:500], attempts=attempts["n"])
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
