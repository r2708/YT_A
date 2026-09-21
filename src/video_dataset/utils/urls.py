"""YouTube URL parsing and input collection (single URL, URL file, directory of URL files, local video)."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".mpg", ".mpeg", ".ts"}
URL_FILE_EXTENSIONS = {".txt", ".urls", ".list", ".csv"}

_YT_ID = r"([A-Za-z0-9_-]{11})"
_PATTERNS = [
    re.compile(r"(?:youtube\.com|youtube-nocookie\.com)/watch\?(?:.*&)?v=" + _YT_ID),
    re.compile(r"youtu\.be/" + _YT_ID),
    re.compile(r"youtube\.com/(?:shorts|embed|live|v)/" + _YT_ID),
    re.compile(r"youtube\.com/attribution_link\?.*v(?:%3D|=)" + _YT_ID),
]
_BARE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_youtube_id(url: str) -> str | None:
    url = url.strip()
    for pat in _PATTERNS:
        m = pat.search(url)
        if m:
            return m.group(1)
    if _BARE_ID.match(url):
        return url
    return None


def canonical_youtube_url(youtube_id: str) -> str:
    return f"https://www.youtube.com/watch?v={youtube_id}"


def is_url(text: str) -> bool:
    return bool(re.match(r"^https?://", text.strip(), re.IGNORECASE))


MAX_URL_LENGTH = 2048
_BLOCKED_HOSTS = {"localhost", "localhost.localdomain", "0.0.0.0", "metadata.google.internal"}


class InvalidURL(ValueError):
    """The URL is not something the downloader should ever be handed."""


def validate_url(url: str, allowed_domains: list[str] | None = None) -> str:
    """Return the stripped URL or raise InvalidURL.

    Rejects: non-http(s) schemes, missing/blocked hosts, embedded credentials, private/loopback/link-local
    IP targets, control characters and over-long strings. With `allowed_domains`, the host must equal one
    of them or be a subdomain of one (so "youtube.com" also allows "www.youtube.com").
    """
    url = (url or "").strip()
    if not url:
        raise InvalidURL("empty URL")
    if len(url) > MAX_URL_LENGTH:
        raise InvalidURL(f"URL longer than {MAX_URL_LENGTH} characters")
    if any(ord(c) < 32 or ord(c) == 127 for c in url) or " " in url:
        raise InvalidURL("URL contains whitespace or control characters")
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise InvalidURL(f"unparseable URL: {exc}") from exc
    if parts.scheme.lower() not in ("http", "https"):
        raise InvalidURL(f"unsupported scheme '{parts.scheme or '(none)'}'; only http/https are accepted")
    if parts.username is not None or parts.password is not None:
        raise InvalidURL("URLs with embedded credentials are not accepted")
    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        raise InvalidURL("URL has no host")
    if host in _BLOCKED_HOSTS:
        raise InvalidURL(f"host '{host}' is not allowed")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
        raise InvalidURL(f"IP address {host} is not a public address")
    if ip is None and "." not in host:
        raise InvalidURL(f"host '{host}' is not a fully qualified domain")
    if allowed_domains:
        allowed = [d.strip().lower().lstrip(".") for d in allowed_domains if d and d.strip()]
        if allowed and not any(host == d or host.endswith("." + d) for d in allowed):
            raise InvalidURL(f"host '{host}' is not in download.allowed_domains {allowed}")
    return url


def url_problem(url: str, allowed_domains: list[str] | None = None) -> str | None:
    """Non-raising variant of validate_url: returns the reason or None when the URL is fine."""
    try:
        validate_url(url, allowed_domains)
    except InvalidURL as exc:
        return str(exc)
    return None


def is_local_video(text: str) -> bool:
    p = Path(text.strip()).expanduser()
    return p.suffix.lower() in VIDEO_EXTENSIONS and p.exists() and p.is_file()


@dataclass(frozen=True)
class InputItem:
    raw: str
    kind: str  # "youtube" | "url" | "local"
    youtube_id: str | None = None
    local_path: str | None = None

    @property
    def url(self) -> str:
        if self.kind == "youtube" and self.youtube_id:
            return canonical_youtube_url(self.youtube_id)
        if self.kind == "local" and self.local_path:
            return Path(self.local_path).resolve().as_uri()
        return self.raw


def classify_input(text: str) -> InputItem | None:
    text = text.strip()
    if not text or text.startswith("#"):
        return None
    yt = extract_youtube_id(text)
    if yt and _BARE_ID.match(text):
        return InputItem(raw=text, kind="youtube", youtube_id=yt)
    if yt and is_url(text) and url_problem(text) is None:
        return InputItem(raw=text, kind="youtube", youtube_id=yt)
    if is_local_video(text):
        return InputItem(raw=text, kind="local", local_path=str(Path(text).expanduser().resolve()))
    if is_url(text):
        if url_problem(text) is None:
            return InputItem(raw=text, kind="url")
        return None  # structurally invalid URL (bad host, credentials, private IP...) - never handed to yt-dlp
    return None


def item_from_stored_url(url: str) -> InputItem | None:
    """Rebuild the input for a video registered earlier (videos.url in state.db), so a batch can be
    resumed without the original URL file. Local files were stored as file:// URIs."""
    url = (url or "").strip()
    if url.startswith("file://"):
        from urllib.parse import unquote, urlsplit
        from urllib.request import url2pathname

        path = url2pathname(unquote(urlsplit(url).path))
        p = Path(path)
        if p.exists() and p.is_file():
            return InputItem(raw=str(p), kind="local", local_path=str(p.resolve()))
        return None
    return classify_input(url)


def read_url_file(path: str | Path) -> list[InputItem]:
    items: list[InputItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        # allow simple CSV: first column is the URL
        candidate = line.split(",")[0] if "," in line and is_url(line.split(",")[0]) else line
        item = classify_input(candidate)
        if item:
            items.append(item)
    return items


def collect_inputs(source: str) -> list[InputItem]:
    """Accepts a URL, a bare YouTube id, a local video, a URL file or a directory of URL files."""
    item = classify_input(source)
    if item is not None and item.kind in ("youtube", "local", "url"):
        return [item]
    p = Path(source).expanduser()
    if p.is_dir():
        items: list[InputItem] = []
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in URL_FILE_EXTENSIONS:
                items.extend(read_url_file(f))
            elif f.suffix.lower() in VIDEO_EXTENSIONS:
                items.append(InputItem(raw=str(f), kind="local", local_path=str(f.resolve())))
        return dedupe_inputs(items)
    if p.is_file():
        return dedupe_inputs(read_url_file(p))
    raise FileNotFoundError(f"Input '{source}' is not a URL, a video file, a URL file or a directory")


def dedupe_inputs(items: list[InputItem]) -> list[InputItem]:
    seen: set[str] = set()
    out: list[InputItem] = []
    for it in items:
        key = it.youtube_id or it.local_path or it.raw
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out
